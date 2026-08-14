from types import SimpleNamespace

from flashinfer import aot


def test_sm103_aot_includes_sm10x_modules(monkeypatch):
    def spec(name):
        return SimpleNamespace(name=name)

    def gen_attention(*args):
        assert args[7] is True
        return []

    monkeypatch.setattr(aot, "gen_spdlog_module", lambda: spec("spdlog"))
    monkeypatch.setattr(aot, "gen_attention", gen_attention)
    generators = {
        "gen_gemm_module": "gemm",
        "gen_gemm_sm100_module": "gemm_sm100",
        "gen_gemm_sm100_module_cutlass_fp8": "fp8_gemm_cutlass",
        "gen_gemm_sm100_module_cutlass_mxfp8": "mxfp8_gemm_cutlass",
        "gen_mxfp8_quantization_sm100_module": "mxfp8_quantization_sm100",
        "gen_trtllm_gen_gemm_module": "trtllm_gemm",
        "gen_trtllm_low_latency_gemm_module": "trtllm_low_latency_gemm",
        "gen_trtllm_gen_fused_moe_sm100_module": "fused_moe_trtllm_sm100",
        "gen_moe_utils_module": "moe_utils",
        "gen_mm_bf16_cublaslt_module": "mm_bf16_cublaslt",
        "gen_fp4_quantization_sm103_module": "fp4_sm103",
        "gen_cutlass_fused_moe_sm103_module": "fused_moe_sm103",
        "gen_gemm_sm103_module_cutlass_fp4": "fp4_gemm_cutlass_sm103",
    }
    for generator, module_name in generators.items():
        monkeypatch.setattr(aot, generator, lambda name=module_name: spec(name))

    tgv_feature_flags = []

    def gen_tgv(dtype, use_sm_100f):
        tgv_feature_flags.append(use_sm_100f)
        return spec(f"tgv_{dtype}")

    monkeypatch.setattr(aot, "gen_tgv_gemm_sm10x_module", gen_tgv)
    monkeypatch.setattr(aot, "gen_cudnn_fmha_module", lambda: spec("cudnn_fmha"))

    specs = aot.gen_all_modules(
        f16_dtype_=[],
        f8_dtype_=[],
        fa2_head_dim_=[],
        fa3_head_dim_=[],
        use_sliding_window_=[],
        use_logits_soft_cap_=[],
        sm_capabilities={"sm103": True},
        add_comm=False,
        add_gemma=False,
        add_oai_oss=False,
        add_moe=True,
        add_act=False,
        add_misc=False,
        add_xqa=False,
    )

    names = {item.name for item in specs}
    assert {
        "gemm_sm100",
        "fp8_gemm_cutlass",
        "mxfp8_gemm_cutlass",
        "mxfp8_quantization_sm100",
        "trtllm_gemm",
        "trtllm_low_latency_gemm",
        "fused_moe_trtllm_sm100",
        "moe_utils",
        "mm_bf16_cublaslt",
        "fp4_sm103",
        "fused_moe_sm103",
        "fp4_gemm_cutlass_sm103",
    } <= names
    assert tgv_feature_flags == [True, True]
