import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from opd_rag.lora import precise_adapter_config, targets_from_weight_keys


def test_numeric_linear_targets_do_not_match_vision_blocks():
    names = ['model.visual.blocks.0', 'model.visual.merger.mlp.0', 'model.visual.merger.mlp.2',
             'model.language_model.layers.0.self_attn.q_proj']
    keys = [f'base_model.model.{name}.lora_{side}.weight' for name in names[1:] for side in ('A', 'B')]
    assert targets_from_weight_keys(keys, names) == sorted(names[1:])


def test_wrong_base_fails_explicitly():
    with pytest.raises(ValueError, match='BASE_MODEL'):
        targets_from_weight_keys(['base_model.model.model.visual.merger.mlp.0.lora_A.weight'], [])


def test_legacy_config_repaired_without_modifying_checkpoint(tmp_path):
    metadata = {'target_modules': ['0', '2'], 'r': 64, 'lora_alpha': 128}
    config_file = tmp_path / 'adapter_config.json'
    config_file.write_text(json.dumps(metadata))
    (tmp_path / 'adapter_model.safetensors').touch()
    original = config_file.read_bytes()
    name = 'model.visual.merger.mlp.0'
    handle = Mock()
    handle.keys.return_value = [f'base_model.model.{name}.lora_A.weight', f'base_model.model.{name}.lora_B.weight']
    context = Mock()
    context.__enter__ = Mock(return_value=handle)
    context.__exit__ = Mock(return_value=False)
    config = SimpleNamespace(**metadata)
    modules = {'safetensors': SimpleNamespace(safe_open=Mock(return_value=context)),
               'peft': SimpleNamespace(PeftConfig=SimpleNamespace(from_pretrained=Mock(return_value=config)))}
    model = SimpleNamespace(config=SimpleNamespace(model_type='qwen2_5_vl'), named_modules=lambda: [(name, object()), ('model.visual.blocks.0', object())])
    with patch.dict('sys.modules', modules):
        result = precise_adapter_config(model, tmp_path)
    assert result.target_modules == {name}
    assert (result.r, result.lora_alpha) == (64, 128)
    assert config_file.read_bytes() == original


def test_unambiguous_config_unchanged(tmp_path):
    (tmp_path / 'adapter_config.json').write_text(json.dumps({'target_modules': ['q_proj']}))
    assert precise_adapter_config(None, tmp_path) is None


@pytest.mark.parametrize('model_type', ['qwen3_vl', 'qwen3_vl_moe'])
def test_qwen3_never_rewrites_adapter_config(tmp_path, model_type):
    (tmp_path / 'adapter_config.json').write_text(json.dumps({'target_modules': ['0', 'q_proj']}))
    model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
    assert precise_adapter_config(model, tmp_path) is None


@pytest.mark.parametrize('model_type', ['qwen2_5_vl', 'qwen3_vl', 'qwen3_vl_moe'])
def test_exporter_only_changes_qwen25_targets(model_type):
    # Execute the exporter's actual extraction loop without loading CUDA dependencies.
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / 'verl-runtime/verl/model_merger/base_model_merger.py'
    tree = ast.parse(source.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'save_lora_adapter')
    loop = next(node for node in method.body if isinstance(node, ast.For))
    name = 'base_model.model.model.visual.merger.mlp.0.lora_A.default.weight'
    namespace = {'self': SimpleNamespace(model_config=SimpleNamespace(model_type=model_type)),
                 'lora_params_names': [name], 'state_dict': {name: 'tensor'},
                 'lora_params': {}, 'target_modules': set()}
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(source), 'exec'), namespace)
    expected = 'model.visual.merger.mlp.0' if model_type == 'qwen2_5_vl' else '0'
    assert namespace['target_modules'] == {expected}
    assert list(namespace['lora_params']) == [name.replace('.default.weight', '.weight')]
