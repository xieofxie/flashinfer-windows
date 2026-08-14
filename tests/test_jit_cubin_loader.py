from flashinfer.jit.cubin_loader import _patch_windows_trtllm_header


def test_patch_windows_trtllm_common_utils_uses_64_bit_constants():
    content = b"""\
constexpr unsigned long LargeN = 1UL << 30;
constexpr unsigned long XLargeN = 1UL << 35;
"""

    patched = _patch_windows_trtllm_header("trtllm/gen/CommonUtils.h", content)

    assert b"constexpr unsigned long long LargeN = 1ULL << 30;" in patched
    assert b"constexpr unsigned long long XLargeN = 1ULL << 35;" in patched
