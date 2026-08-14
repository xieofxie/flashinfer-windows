import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from flashinfer.jit import core, cpp_ext


def test_build_jit_specs_escapes_windows_drive_colon(monkeypatch, tmp_path):
    captured = {}
    spec = SimpleNamespace(
        aot_path=tmp_path / "missing",
        ninja_path=Path(r"C:\_fib\aot\cached_ops\spdlog\build.ninja"),
        lock_path=tmp_path / "spec.lock",
        write_ninja=lambda: None,
    )
    monkeypatch.setattr(core.platform, "system", lambda: "Windows")
    monkeypatch.setattr(core, "FileLock", lambda *_args, **_kwargs: core.nullcontext())
    monkeypatch.setattr(core, "get_tmpdir", lambda: tmp_path)
    monkeypatch.setattr(
        core,
        "write_if_different",
        lambda _path, content: captured.setdefault("content", content),
    )
    monkeypatch.setattr(core, "run_ninja", lambda *_args, **_kwargs: None)

    core.build_jit_specs([spec], skip_prebuilt=False)

    assert (
        r"subninja C$:\_fib\aot\cached_ops\spdlog\build.ninja"
        in captured["content"]
    )


def test_nvcc_parallelism_flags_use_flashinfer_nvcc_threads(monkeypatch):
    monkeypatch.setenv("FLASHINFER_NVCC_THREADS", "4")

    assert cpp_ext.get_nvcc_parallelism_flags() == ["--threads=4"]


def test_nvcc_parallelism_flags_ignore_sccache_launcher(monkeypatch):
    monkeypatch.setenv("FLASHINFER_NVCC_THREADS", "4")
    monkeypatch.setenv("FLASHINFER_NVCC_LAUNCHER", "sccache")

    assert cpp_ext.get_nvcc_parallelism_flags() == ["--threads=4"]


@pytest.mark.parametrize(
    ("machine", "expected"),
    [
        ("AMD64", "x64"),
        ("x86_64", "x64"),
        ("ARM64", "arm64"),
        ("aarch64", "arm64"),
    ],
)
def test_windows_cuda_arch_dir(monkeypatch, machine, expected):
    monkeypatch.setattr(cpp_ext.platform, "machine", lambda: machine)

    assert cpp_ext.get_windows_cuda_arch_dir() == expected


def test_windows_cuda_bin_path_uses_process_architecture(monkeypatch, tmp_path):
    (tmp_path / "bin" / "x64").mkdir(parents=True)
    (tmp_path / "bin" / "arm64").mkdir()
    monkeypatch.setattr(cpp_ext.platform, "machine", lambda: "ARM64")

    assert cpp_ext.get_windows_cuda_bin_path(str(tmp_path)) == str(
        tmp_path / "bin" / "arm64"
    )


def test_windows_arm64_cuda_compat_header_lowers_tensor_map_alignment(
    monkeypatch, tmp_path
):
    cuda_home = tmp_path / "cuda"
    cuda_header = cuda_home / "include" / "cuda.h"
    cuda_header.parent.mkdir(parents=True)
    cuda_header.write_text(
        """\
typedef struct CUtensorMap_st {
#if defined(__cplusplus)
    alignas(128)
#else
    _Alignas(128)
#endif
    unsigned long long opaque[16];
} CUtensorMap;
"""
    )
    monkeypatch.setattr(cpp_ext, "is_windows", True)
    monkeypatch.setattr(cpp_ext.platform, "machine", lambda: "ARM64")
    monkeypatch.setattr(cpp_ext, "get_cuda_version", lambda: cpp_ext.Version("13.4"))
    monkeypatch.setattr(
        cpp_ext.jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path / "generated"
    )

    compat_dir = cpp_ext.get_windows_arm64_cuda_compat_include(str(cuda_home))
    compat_header = (compat_dir / "cuda.h").read_text()

    assert "alignas(64)" in compat_header
    assert "_Alignas(64)" in compat_header
    assert "alignas(128)" not in compat_header
    original_header = cuda_header.read_text()
    assert "alignas(128)" in original_header
    assert "_Alignas(128)" in original_header


def test_generate_ninja_uses_arm64_cuda_libraries(monkeypatch, tmp_path):
    monkeypatch.setattr(cpp_ext, "is_windows", True)
    monkeypatch.setattr(cpp_ext.platform, "machine", lambda: "ARM64")
    monkeypatch.setattr(cpp_ext, "get_cuda_path", lambda: "C:\\CUDA\\v13.4")
    monkeypatch.setattr(
        cpp_ext,
        "get_windows_arm64_cuda_compat_include",
        lambda _cuda_home: tmp_path / "compat",
    )
    monkeypatch.setattr(cpp_ext.jit_env, "FLASHINFER_JIT_DIR", tmp_path / "jit")
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "12.1a")

    ninja = cpp_ext.generate_ninja_build_for_op(
        name="test_module",
        sources=[tmp_path / "generated" / "kernel.cu"],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_dirs=None,
    )

    assert '"/LIBPATH:$cuda_home\\lib\\arm64"' in ninja
    assert f'-I"{tmp_path / "compat"}"' in ninja
    assert "/bigobj" in ninja
    assert "-Xcompiler /bigobj" in ninja


def test_windows_object_name_is_shortened_for_long_depfile_path(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cpp_ext, "is_windows", True)
    output_dir = tmp_path / ("m" * 40)
    source = tmp_path / ("s" * 140) / "selective_state_update_kernel_inst.cu"

    object_name = cpp_ext.get_object_file_name(source, output_dir)

    assert object_name.startswith("obj_")
    assert object_name.endswith(".cuda.o")
    assert len(str(output_dir / f"{object_name}.d")) < 240


def test_windows_object_name_stays_readable_when_path_is_short(monkeypatch, tmp_path):
    monkeypatch.setattr(cpp_ext, "is_windows", True)
    source = tmp_path / "generated" / "kernel.cu"

    assert (
        cpp_ext.get_object_file_name(source, tmp_path / "jit")
        == "generated_kernel.cuda.o"
    )


def test_generate_ninja_uses_sccache_compatible_nvcc_depfile_flag(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cpp_ext, "get_cuda_path", lambda: "/usr/local/cuda")
    monkeypatch.setattr(
        cpp_ext, "get_windows_arm64_cuda_compat_include", lambda _cuda_home: None
    )
    monkeypatch.setattr(cpp_ext.jit_env, "FLASHINFER_JIT_DIR", tmp_path / "jit")
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "7.5")

    ninja = cpp_ext.generate_ninja_build_for_op(
        name="test_module",
        sources=[tmp_path / "generated" / "kernel.cu"],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_dirs=None,
    )

    assert "--generate-dependencies-with-compile -MF $out.d" in ninja
    assert "--dependency-output" not in ninja


def test_debug_jit_uses_sccache_compatible_nvcc_device_debug_flag(monkeypatch):
    monkeypatch.setenv("FLASHINFER_JIT_DEBUG", "1")
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(core, "get_nvcc_parallelism_flags", lambda: ["--threads=1"])

    spec = core.gen_jit_spec(
        name="test_module",
        sources=[],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_paths=None,
    )

    assert "--device-debug" in spec.extra_cuda_cflags
    assert "-G" not in spec.extra_cuda_cflags


def test_release_jit_propagates_ndebug_to_host_cflags(monkeypatch):
    monkeypatch.delenv("FLASHINFER_JIT_DEBUG", raising=False)
    monkeypatch.delenv("FLASHINFER_JIT_VERBOSE", raising=False)
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(core, "get_nvcc_parallelism_flags", lambda: ["--threads=1"])

    spec = core.gen_jit_spec(
        name="test_module",
        sources=[],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_paths=None,
    )

    assert "-DNDEBUG" in spec.extra_cflags
    assert "-DNDEBUG" in spec.extra_cuda_cflags


def test_debug_jit_does_not_propagate_ndebug(monkeypatch):
    monkeypatch.setenv("FLASHINFER_JIT_DEBUG", "1")
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(core, "get_nvcc_parallelism_flags", lambda: ["--threads=1"])

    spec = core.gen_jit_spec(
        name="test_module",
        sources=[],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_paths=None,
    )

    assert "-DNDEBUG" not in spec.extra_cflags
    assert "-DNDEBUG" not in spec.extra_cuda_cflags


def test_run_ninja_uses_max_jobs(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setenv("MAX_JOBS", "8")
    monkeypatch.setattr(cpp_ext.subprocess, "run", fake_run)

    cpp_ext.run_ninja(tmp_path, tmp_path / "build.ninja", verbose=False)

    assert commands == [
        [
            "ninja",
            "-v",
            "-C",
            str(tmp_path.resolve()),
            "-f",
            str((tmp_path / "build.ninja").resolve()),
            "-j",
            "8",
        ]
    ]


def test_jit_spec_build_rewrites_ninja_before_build(monkeypatch):
    writes = []
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)

    spec = core.JitSpec(
        name="test_module",
        sources=[],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_dirs=None,
    )

    monkeypatch.setattr(spec, "write_ninja", lambda: writes.append(True))
    monkeypatch.setattr(core, "run_ninja", lambda *_args, **_kwargs: None)

    spec.build(verbose=False, need_lock=False)

    assert writes == [True]
