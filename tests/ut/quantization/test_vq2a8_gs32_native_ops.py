# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of registration and constructor ordering, not NPU execution."""

import ast
import builtins
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
NATIVE_MODULE = "vllm_ascend.models.deepseek_v4.native_ops"
EXTENSION_MODULE = "vllm_ascend.vllm_ascend_C"
REQUIRED_OPS = (
    "npu_hc_pre_v2",
    "npu_hc_post",
    "moe_gating_top_k_hash",
    "inplace_partial_rotary_mul",
    "compressor",
    "compressor_metadata",
    "compressor_metadata_out",
    "npu_quant_lightning_indexer_v2",
    "npu_quant_lightning_indexer_v2_metadata",
    "npu_kv_quant_sparse_attn_sharedkv",
    "npu_kv_quant_sparse_attn_sharedkv_metadata",
    "kv_compress_epilog",
    "indexer_compress_epilog_v2",
)


def load_native_helper():
    """Isolate this helper from the real vLLM model registry on CPU hosts."""
    path = REPO_ROOT / "vllm_ascend/models/deepseek_v4/native_ops.py"
    spec = importlib.util.spec_from_file_location(NATIVE_MODULE, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_registration(monkeypatch, *, missing_schema=None, missing_backend=None, import_error=None):
    events = []
    registered = {name for name in REQUIRED_OPS if name != missing_schema}

    def has_backend(schema, backend):
        assert schema.removeprefix("_C_ascend::") in registered
        assert backend == "PrivateUse1"
        events.append((schema, backend))
        return schema != f"_C_ascend::{missing_backend}"

    torch = ModuleType("torch")
    torch.ops = SimpleNamespace(_C_ascend=SimpleNamespace(**dict.fromkeys(registered, object())))
    torch._C = SimpleNamespace(_dispatch_has_kernel_for_dispatch_key=has_backend)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    utils = ModuleType("vllm_ascend.utils")
    utils.bootstrap_custom_op_env = lambda: events.append("bootstrap")

    def disallow_generic_enable():
        raise AssertionError("GS32 registration must not enable generic A5 custom-op replacements")

    utils.enable_custom_op = disallow_generic_enable
    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", utils)
    extension = SimpleNamespace(__file__="/installed/vllm_ascend/vllm_ascend_C.so")
    original_import = importlib.import_module

    def import_module(name, package=None):
        if name == EXTENSION_MODULE:
            events.append("extension")
            if import_error is not None:
                raise import_error
            return extension
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)
    return events, extension


def test_gs32_native_helper_import_is_lazy(monkeypatch):
    original_import = builtins.__import__

    def guard_import(name, *args, **kwargs):
        if name in {"torch", "torch_npu", "vllm_ascend.utils", EXTENSION_MODULE}:
            raise AssertionError(f"Import initialized the runtime prematurely: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard_import)
    assert callable(load_native_helper().require_deepseek_v4_native_ops)


def test_gs32_native_registration_bootstraps_and_checks_device_bindings(monkeypatch):
    events, extension = fake_registration(monkeypatch)
    result = load_native_helper().require_deepseek_v4_native_ops()
    assert result == extension.__file__
    assert events[:2] == ["bootstrap", "extension"]
    assert set(events[2:]) == {(f"_C_ascend::{name}", "PrivateUse1") for name in REQUIRED_OPS}


@pytest.mark.parametrize("missing", ["missing_schema", "missing_backend"])
@pytest.mark.parametrize("name", ["npu_hc_pre_v2", "moe_gating_top_k_hash", "compressor_metadata_out"])
def test_gs32_native_registration_rejects_incomplete_main_extension(monkeypatch, missing, name):
    fake_registration(monkeypatch, **{missing: name})
    with pytest.raises(RuntimeError, match=name):
        load_native_helper().require_deepseek_v4_native_ops()


@pytest.mark.parametrize("error_type", [ImportError, OSError])
def test_gs32_native_registration_preserves_extension_load_failure(monkeypatch, error_type):
    original_error = error_type("libcust_opapi.so could not be loaded")
    fake_registration(monkeypatch, import_error=original_error)
    with pytest.raises((ImportError, OSError, RuntimeError)) as failure:
        load_native_helper().require_deepseek_v4_native_ops()
    assert failure.value is original_error or failure.value.__cause__ is original_error


def model_constructor(monkeypatch, check, *, compressed_cache=False):
    """Execute the actual constructor body, isolating upstream model allocation.

    Extracting the method avoids importing unavailable vLLM dependencies. The
    method's control flow and call ordering are executed, not matched as text.
    """
    path = REPO_ROOT / "vllm_ascend/models/deepseek_v4/model.py"
    owner = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "AscendDeepseekV4ForCausalLM"
    )
    constructor = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    definition = ast.ClassDef(
        name="Model", bases=[ast.Name(id="object", ctx=ast.Load())], keywords=[], body=[constructor], decorator_list=[]
    )
    helper = ModuleType(NATIVE_MODULE)
    helper.require_deepseek_v4_native_ops = check
    monkeypatch.setitem(sys.modules, NATIVE_MODULE, helper)
    compressed_cache_capability = object()

    def supports(capability):
        assert capability is compressed_cache_capability
        return compressed_cache

    scope = {
        "VllmConfig": object,
        "HardwareCapability": SimpleNamespace(DSV4_COMPRESSED_CACHE=compressed_cache_capability),
        "get_current_hardware_profile": lambda: SimpleNamespace(supports=supports),
        "require_deepseek_v4_native_ops": check,
        "maybe_prefix": lambda prefix, name: f"{prefix}.{name}" if prefix else name,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])), str(path), "exec"), scope)
    return scope["Model"]


@pytest.mark.parametrize(
    ("compressed_cache", "is_gs32"),
    [
        pytest.param(True, False, id="a5-native-baseline"),
        pytest.param(True, True, id="a5-gs32"),
        pytest.param(False, True, id="non-a5-gs32"),
    ],
)
def test_gs32_worker_model_rejects_missing_bindings_before_constructing_layers(monkeypatch, compressed_cache, is_gs32):
    events = []

    def check():
        events.append("check")
        raise RuntimeError("missing npu_hc_pre_v2")

    def allocate_model(**kwargs):
        events.append("allocate")
        raise AssertionError("Model allocation must not start with missing NPU bindings")

    model = model_constructor(monkeypatch, check, compressed_cache=compressed_cache)
    model.model_cls = staticmethod(allocate_model)
    quant_config = SimpleNamespace(is_gs32=True) if is_gs32 else SimpleNamespace()
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=object()), quant_config=quant_config)
    with pytest.raises(RuntimeError, match="missing npu_hc_pre_v2"):
        model(vllm_config=config)
    assert events == ["check"]


@pytest.mark.parametrize("compressed_cache", [True, False], ids=["compressed-cache", "no-compressed-cache"])
@pytest.mark.parametrize("quantization", ["native-baseline", "gs32", "non-gs32", "none"])
def test_gs32_worker_model_registration_is_scoped_and_precedes_layers(monkeypatch, compressed_cache, quantization):
    events = []

    class ReachedModelAllocation(Exception):
        pass

    def allocate_model(**kwargs):
        events.append("allocate")
        raise ReachedModelAllocation

    model = model_constructor(monkeypatch, lambda: events.append("check"), compressed_cache=compressed_cache)
    model.model_cls = staticmethod(allocate_model)
    quant_config = {
        "native-baseline": SimpleNamespace(),
        "gs32": SimpleNamespace(is_gs32=True),
        "non-gs32": SimpleNamespace(is_gs32=False),
        "none": None,
    }[quantization]
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=object()), quant_config=quant_config)
    with pytest.raises(ReachedModelAllocation):
        model(vllm_config=config)
    assert events == (["check", "allocate"] if compressed_cache or quantization == "gs32" else ["allocate"])
