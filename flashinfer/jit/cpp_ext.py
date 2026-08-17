# Adapted from https://github.com/pytorch/pytorch/blob/v2.7.0/torch/utils/cpp_extension.py

import functools
import hashlib
import logging
import os
import platform
import re
import subprocess
import sys
import sysconfig
from packaging.version import Version
from pathlib import Path
from typing import List, Optional

import tvm_ffi
import torch

from . import env as jit_env
from ..compilation_context import CompilationContext
from .utils import write_if_different

is_windows = platform.system() == "Windows"
logger = logging.getLogger(__name__)
_WINDOWS_SAFE_DEPFILE_PATH_LENGTH = 240


def get_windows_cuda_arch_dir() -> str:
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    if machine in ("amd64", "x86_64"):
        return "x64"
    raise RuntimeError(f"Unsupported Windows architecture for CUDA: {machine}")


def get_windows_cuda_bin_path(cuda_home: str) -> str:
    arch_bin_path = os.path.join(cuda_home, "bin", get_windows_cuda_arch_dir())
    if os.path.exists(arch_bin_path):
        return arch_bin_path
    return os.path.join(cuda_home, "bin")


def get_windows_arm64_cuda_compat_include(cuda_home: str) -> Optional[Path]:
    if not is_windows or get_windows_cuda_arch_dir() != "arm64":
        return None
    if get_cuda_version().major != 13:
        return None

    # CUDA 13 emits ARM64 MSVC stubs that pass CUtensorMap by value, but MSVC
    # rejects its 128-byte alignment. Shadow cuda.h instead of modifying the toolkit.
    cuda_header = Path(cuda_home) / "include" / "cuda.h"
    if not cuda_header.exists():
        raise FileNotFoundError(f"CUDA header not found: {cuda_header}")

    header_content = cuda_header.read_text(encoding="utf-8")
    tensor_map_start = header_content.find("typedef struct CUtensorMap_st {")
    tensor_map_end = header_content.find("} CUtensorMap;", tensor_map_start)
    if tensor_map_start < 0 or tensor_map_end < 0:
        raise RuntimeError(f"Could not find CUtensorMap definition in {cuda_header}")

    tensor_map_end += len("} CUtensorMap;")
    tensor_map_definition = header_content[tensor_map_start:tensor_map_end]
    patched_definition = tensor_map_definition.replace("alignas(128)", "alignas(64)")
    patched_definition = patched_definition.replace("_Alignas(128)", "_Alignas(64)")
    if patched_definition == tensor_map_definition:
        raise RuntimeError(
            f"Could not patch CUtensorMap alignment in {cuda_header}"
        )

    compat_dir = jit_env.FLASHINFER_GEN_SRC_DIR / "windows_arm64_cuda_compat"
    write_if_different(
        compat_dir / "cuda.h",
        header_content[:tensor_map_start]
        + patched_definition
        + header_content[tensor_map_end:],
    )
    return compat_dir


def parse_env_flags(env_var_name) -> List[str]:
    env_flags = os.environ.get(env_var_name)
    if env_flags:
        try:
            import shlex

            return shlex.split(env_flags)
        except ValueError as e:
            logger.warning(
                "Could not parse %s with shlex: %s. Falling back to simple split.",
                env_var_name,
                e,
            )
            return env_flags.split()
    return []


def _get_glibcxx_abi_build_flags() -> List[str]:
    glibcxx_abi_cflags = [] if is_windows else [
        "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))
    ]
    return glibcxx_abi_cflags


@functools.cache
def get_cuda_path() -> str:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is not None:
        return cuda_home
    # get output of "which nvcc"
    nvcc_path = subprocess.run(["which", "nvcc"], capture_output=True)
    if nvcc_path.returncode == 0:
        cuda_home = os.path.dirname(
            os.path.dirname(nvcc_path.stdout.decode("utf-8").strip())
        )
    else:
        cuda_home = "/usr/local/cuda"  # This default value is from: https://github.com/pytorch/pytorch/blob/ceb11a584d6b3fdc600358577d9bf2644f88def9/torch/utils/cpp_extension.py#L115
        if not os.path.exists(cuda_home):
            raise RuntimeError(
                f"Could not find nvcc and default {cuda_home=} doesn't exist"
            )
    return cuda_home


@functools.cache
def get_cuda_version() -> Version:
    # Try to query nvcc for CUDA version; if nvcc is unavailable, fall back to torch.version.cuda
    try:
        cuda_home = get_cuda_path()
        nvcc = os.path.join(cuda_home, "bin/nvcc")
        txt = subprocess.check_output([nvcc, "--version"], text=True)
        matches = re.findall(r"release (\d+\.\d+),", txt)
        if not matches:
            raise RuntimeError(
                f"Could not parse CUDA version from nvcc --version output: {txt}"
            )
        return Version(matches[0])
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as e:
        # NOTE(Zihao): when nvcc is unavailable, fall back to torch.version.cuda
        if torch.version.cuda is None:
            raise RuntimeError(
                "nvcc not found and PyTorch is not built with CUDA support. "
                "Could not determine CUDA version."
            ) from e
        return Version(torch.version.cuda)


def is_cuda_version_at_least(version_str: str) -> bool:
    return get_cuda_version() >= Version(version_str)


def get_nvcc_parallelism_flags() -> List[str]:
    """Build nvcc flags controlled by FlashInfer parallelism environment variables."""
    env_var_name = "FLASHINFER_NVCC_THREADS"
    default = 1
    value = os.environ.get(env_var_name, str(default))

    try:
        threads = int(value)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; using %s.", env_var_name, value, default
        )
        threads = default

    if threads < 1:
        logger.warning("Ignoring %s=%r; value must be >= 1.", env_var_name, value)
        threads = default

    return [f"--threads={threads}"]


def join_multiline(vs: List[str]) -> str:
    return " $\n    ".join(vs)


def get_object_file_name(source: Path, output_dir: Path) -> str:
    object_suffix = ".cuda.o" if source.suffix == ".cu" else ".o"
    object_name = f"{source.parent.name}_{source.stem}{object_suffix}"
    depfile_path = (output_dir / f"{object_name}.d").resolve()
    if is_windows and len(str(depfile_path)) >= _WINDOWS_SAFE_DEPFILE_PATH_LENGTH:
        source_hash = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:16]
        object_name = f"obj_{source_hash}{object_suffix}"
    return object_name


def get_cccl_includes() -> List:
    """Get vendored CCCL include directories (added with -I for CTK override precedence)."""
    return [p.resolve() for p in jit_env.CCCL_INCLUDE_DIRS]


def get_system_includes(cuda_home: str) -> List:
    """Get list of system include directories."""
    system_includes = [
        sysconfig.get_path("include"),
        "$cuda_home/include",
        tvm_ffi.libinfo.find_include_path(),
        tvm_ffi.libinfo.find_dlpack_include_path(),
        jit_env.FLASHINFER_INCLUDE_DIR.resolve(),
        jit_env.FLASHINFER_CSRC_DIR.resolve(),
    ]
    system_includes += [p.resolve() for p in jit_env.CUTLASS_INCLUDE_DIRS]
    system_includes.append(jit_env.SPDLOG_INCLUDE_DIR.resolve())

    compat_include = get_windows_arm64_cuda_compat_include(cuda_home)
    if compat_include is not None:
        system_includes.insert(0, compat_include.resolve())

    if cuda_home == "/usr":
        # NOTE: this will resolve to /usr/include, which will mess up includes. See #1793
        system_includes.remove("$cuda_home/include")

    return system_includes


def build_common_cflags(
    cuda_home: str,
    extra_include_dirs: Optional[List[Path]] = None,
) -> List[str]:
    """Build common compilation flags."""
    cccl_includes = get_cccl_includes()
    system_includes = get_system_includes(cuda_home)

    common_cflags = []
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        common_cflags.append("-DPy_LIMITED_API=0x03090000")
    common_cflags += _get_glibcxx_abi_build_flags()
    if extra_include_dirs is not None:
        for extra_dir in extra_include_dirs:
            common_cflags.append(f"-I{extra_dir.resolve()}")
    # Vendored CCCL headers use -I (not -isystem) so they take precedence
    # over the CTK-bundled copy. CCCL headers use #pragma system_header
    # internally to suppress warnings. See https://github.com/NVIDIA/cccl/issues/527
    if is_windows:
        for cccl_dir in cccl_includes:
            common_cflags.append(f'-I"{str(cccl_dir)}"')
        for sys_dir in system_includes:
            common_cflags.append(f'-I"{str(sys_dir)}"')
    else:
        for cccl_dir in cccl_includes:
            common_cflags.append(f"-I{cccl_dir}")
        for sys_dir in system_includes:
            common_cflags.append(f"-isystem {sys_dir}")

    return common_cflags


def build_cflags(
    common_cflags: List[str],
    extra_cflags: Optional[List[str]] = None,
) -> List[str]:
    """Build C++ compilation flags."""
    cflags = [
        "$common_cflags",
    ]

    if not is_windows:
        cflags.append("-fPIC")
    else:
        cflags.append("/std:c++20")
        cflags.append("/DNOMINMAX")
        cflags.append("/Zc:preprocessor")
        cflags.append("/bigobj")

    if extra_cflags is not None:
        cflags += extra_cflags

    env_extra_cflags = parse_env_flags("FLASHINFER_EXTRA_CFLAGS")
    if env_extra_cflags is not None:
        cflags += env_extra_cflags

    cflags = list(set(cflags))

    return cflags


def build_cuda_cflags(
    common_cflags: List[str],
    extra_cuda_cflags: Optional[List[str]] = None,
) -> List[str]:
    """Build CUDA compilation flags."""
    cuda_cflags: List[str] = []
    cc_env = os.environ.get("CC")
    if cc_env is not None:
        cuda_cflags += ["-ccbin", cc_env]
    common_cuda_flags = common_cflags.copy()

    if is_windows:
        common_cuda_flags = [
            "-DTORCH_EXTENSION_NAME=$name",
            "--std=c++20",
            "-Xcompiler /Zc:__cplusplus",
            "-Xcompiler /Zc:preprocessor",
            "-Xcompiler /bigobj",
        ] + common_cuda_flags[1:]

    cuda_cflags += [
        "$common_cuda_flags",
        "--expt-relaxed-constexpr",
    ]

    if not is_windows:
        cuda_cflags.append("--compiler-options=-fPIC")
    cuda_version = get_cuda_version()
    # enable -static-global-template-stub when cuda version >= 12.8
    if cuda_version >= Version("12.8"):
        cuda_cflags += [
            "-static-global-template-stub=false",
        ]

    cpp_ext_initial_compilation_context = CompilationContext()
    global_flags = cpp_ext_initial_compilation_context.get_nvcc_flags_list()
    if extra_cuda_cflags is not None:
        # Check if module provides architecture flags
        module_has_gencode = any(
            flag.startswith("-gencode=") for flag in extra_cuda_cflags
        )

        if module_has_gencode:
            # Use module's architecture flags, but keep global non-architecture flags
            global_non_arch_flags = [
                flag for flag in global_flags if not flag.startswith("-gencode=")
            ]
            cuda_cflags += global_non_arch_flags + extra_cuda_cflags
        else:
            # No module architecture flags, use both global and module flags
            cuda_cflags += global_flags + extra_cuda_cflags
    else:
        # No module flags, use global flags
        cuda_cflags += global_flags

    env_extra_cuda_cflags = parse_env_flags("FLASHINFER_EXTRA_CUDAFLAGS")
    if env_extra_cuda_cflags is not None:
        cuda_cflags += env_extra_cuda_cflags

    return cuda_cflags, common_cuda_flags


def generate_ninja_build_for_op(
    name: str,
    sources: List[Path],
    extra_cflags: Optional[List[str]],
    extra_cuda_cflags: Optional[List[str]],
    extra_ldflags: Optional[List[str]],
    extra_include_dirs: Optional[List[Path]],
    needs_device_linking: bool = False,
) -> str:
    cuda_home = get_cuda_path()
    common_cflags = build_common_cflags(cuda_home, extra_include_dirs)
    cflags = build_cflags(common_cflags, extra_cflags)
    cuda_cflags, common_cuda_flags = build_cuda_cflags(common_cflags, extra_cuda_cflags)

    if is_windows:
        python_path = os.path.dirname(sys.executable)
        if python_path.endswith("\\Scripts"):
            python_path = os.path.dirname(python_path)
        python_lib_path = os.path.join(sys.base_exec_prefix, "libs")
        cuda_arch_dir = get_windows_cuda_arch_dir()
        ldflags = [
            f'"/LIBPATH:{python_lib_path}"',
            f'"/LIBPATH:$cuda_home\\lib\\{cuda_arch_dir}"',
            f'"/LIBPATH:{python_path}\\Lib\\site-packages\\tvm_ffi\\lib"',
            f'"/LIBPATH:{python_path}\\Lib\\site-packages\\torch\\lib"',
            "c10.lib",
            "c10_cuda.lib",
            "torch.lib",
            "torch_cuda.lib",
            "cudart.lib",
            "cuda.lib",
            "tvm_ffi.lib",
            "torch_python.lib"
        ]
    else:
        ldflags = [
            "-shared",
            "-L$cuda_home/lib64",
            "-L$cuda_home/lib64/stubs",
            "-lcudart",
            "-lcuda",
        ]

    env_extra_ldflags = parse_env_flags("FLASHINFER_EXTRA_LDFLAGS")
    if env_extra_ldflags is not None:
        ldflags += env_extra_ldflags

    if extra_ldflags is not None:
        if is_windows:
            for ldflag in extra_ldflags:
                if ldflag.startswith("-l"):
                    ldflag = ldflag[2:] + ".lib"
                ldflags.append(ldflag)
        else:
            ldflags += extra_ldflags

    cxx = os.environ.get("CXX", "c++")
    nvcc = os.environ.get("FLASHINFER_NVCC", "$cuda_home/bin/nvcc")
    if is_windows:
        nvcc = f'"{nvcc}"'
    # Compiler launchers (e.g., sccache, ccache) — empty string when unset
    cxx_launcher = os.environ.get("FLASHINFER_CXX_LAUNCHER", "")
    nvcc_launcher = os.environ.get("FLASHINFER_NVCC_LAUNCHER", "")

    if is_windows:
        rule_compile = [
            "rule compile",
            "  command = cl.exe $cflags -c $in /Fo$out $post_cflags",
            "  deps = msvc",
        ]
        rule_cuda_compile = [
            "rule cuda_compile",
            "  command = $nvcc --generate-dependencies-with-compile -MF $out.d $cuda_cflags -c $in -o $out $cuda_post_cflags",
            "  depfile = $out.d",
            "  deps = msvc",
        ]
        rule_link = [
            "rule link",
            "  command = link.exe /DLL $in /nologo $ldflags /out:$out",
        ]
    else:
        rule_compile = [
            "rule compile",
            "  command = $cxx_launcher $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags",
            "  depfile = $out.d",
            "  deps = gcc",
        ]
        rule_cuda_compile = [
            "rule cuda_compile",
            "  command = $nvcc_launcher $nvcc --generate-dependencies-with-compile -MF $out.d $cuda_cflags -c $in -o $out $cuda_post_cflags",
            "  depfile = $out.d",
            "  deps = gcc",
        ]
        rule_link = [
            "rule link",
            "  command = $cxx $in $ldflags -o $out",
        ]

    lines = [
        "ninja_required_version = 1.3",
        f"name = {name}",
        f"cuda_home = {cuda_home}",
        f"cxx = {cxx}",
        f"nvcc = {nvcc}",
        f"cxx_launcher = {cxx_launcher}",
        f"nvcc_launcher = {nvcc_launcher}",
        "",
        "common_cflags = " + join_multiline(common_cflags),
        "common_cuda_flags = " + join_multiline(common_cuda_flags),
        "cflags = " + join_multiline(cflags),
        "post_cflags =",
        "cuda_cflags = " + join_multiline(cuda_cflags),
        "cuda_post_cflags =",
        "ldflags = " + join_multiline(ldflags),
        "",
        *rule_compile,
        "",
        *rule_cuda_compile,
        "",

    ]

    # Add nvcc linking rule for device code
    if needs_device_linking:
        lines.extend(
            [
                "rule nvcc_link",
                "  command = $nvcc -shared $in $ldflags -o $out",
                "",
            ]
        )
    else:
        lines.extend(
            [
                *rule_link,
                "",
            ]
        )

    # Use absolute paths for outputs so ninja files work with any workdir
    # This enables isolated workdirs for runtime JIT (avoiding .ninja_log races)
    # while still supporting subninja for parallel AOT builds
    output_dir = jit_env.FLASHINFER_JIT_DIR / name

    objects = []
    for source in sources:
        is_cuda = source.suffix == ".cu"
        cmd = "cuda_compile" if is_cuda else "compile"
        obj_name = get_object_file_name(source, output_dir)
        obj = str((output_dir / obj_name).resolve()).replace(":\\", "$:\\")
        objects.append(obj)
        source_path = source.resolve()
        if is_windows:
            source_path = str(source_path).replace(":\\", "$:\\")
        lines.append(f"build {obj}: {cmd} {source_path}")

    lines.append("")
    link_rule = "nvcc_link" if needs_device_linking else "link"
    if is_windows:
        output_so = str((output_dir / f"{name}.dll").resolve()).replace(":\\", "$:\\")
        lines.append(f"build {output_so}: {link_rule} " + " ".join(objects))
        lines.append(f"default {output_so}")
    else:
        output_so = str((output_dir / f"{name}.so").resolve())
        lines.append(f"build {output_so}: {link_rule} " + " ".join(objects))
        lines.append(f"default {output_so}")
    lines.append("")

    return "\n".join(lines)


def _get_num_workers() -> Optional[int]:
    max_jobs = os.environ.get("MAX_JOBS")
    if max_jobs is not None and max_jobs.isdigit():
        return int(max_jobs)
    return None


def _get_ninja_env() -> dict:
    env = os.environ.copy()
    if is_windows and env.get("FLASHINFER_JIT_WARNINGS", "0") != "1":
        env["_CL_"] = f"{env.get('_CL_', '')} /w".strip()
        env["NVCC_APPEND_FLAGS"] = (
            f"{env.get('NVCC_APPEND_FLAGS', '')} -w -Xcompiler=/w".strip()
        )
    return env


def run_ninja(workdir: Path, ninja_file: Path, verbose: bool) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    command = [
        "ninja",
        "-v",
        "-C",
        str(workdir.resolve()),
        "-f",
        str(ninja_file.resolve()),
    ]
    num_workers = _get_num_workers()
    if num_workers is not None:
        command += ["-j", str(num_workers)]

    sys.stdout.flush()
    sys.stderr.flush()
    try:
        subprocess.run(
            command,
            stdout=None if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(workdir.resolve()),
            check=True,
            text=True,
            env=_get_ninja_env(),
        )
    except subprocess.CalledProcessError as e:
        msg = "Ninja build failed."
        if e.output:
            msg += " Ninja output:\n" + e.output
        raise RuntimeError(msg) from e