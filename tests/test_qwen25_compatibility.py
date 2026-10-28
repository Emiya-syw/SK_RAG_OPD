"""Compatibility regressions, without allocating GPUs or loading weights."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from jinja2 import Environment

from opd_rag.thinking import (
    chat_template_kwargs, resolve_model_type, validate_distillation_pair,
)

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('auto_config_qwen25', ROOT / 'scripts/pipeline_auto_config.py')
auto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto)


@pytest.mark.parametrize('nested', [False, True])
def test_qwen25_config_and_presets(tmp_path, nested):
    text = {'hidden_size': 3584, 'num_hidden_layers': 28}
    config = {'model_type': 'qwen2_5_vl', **({'text_config': text} if nested else text)}
    (tmp_path / 'config.json').write_text(json.dumps(config))
    assert resolve_model_type(tmp_path) == 'qwen2_5_vl'
    assert auto.model_size(tmp_path) == 7
    for count in (4, 8):
        result = auto.choose({'DISTILLATION_LOSS_MODE': 'entropy_jsd'}, 7, 7,
                             [{'index': str(i)} for i in range(count)])
        assert int(result['TRAIN_BATCH_SIZE']) == count * 2
        assert result['USE_CHUNKED_TOPK'] == 'true'
        assert result['GENERATION_BATCH_SIZE'] == '4'
        assert result['TEACHER_MAX_NUM_SEQS'] == '2'


@pytest.mark.parametrize('thinking', [False, True])
def test_qwen25_does_not_override_native_template(thinking):
    assert chat_template_kwargs(thinking, 'qwen2_5_vl') == {}


def test_local_qwen25_template_renders_image_without_think():
    path = Path('/home/sunyw/video_safety/weights/Qwen2.5-VL-7B-Instruct')
    if not path.is_dir():
        pytest.skip('Local model not installed')
    template = json.loads((path / 'chat_template.json').read_text())['chat_template']
    rendered = Environment().from_string(template).render(
        messages=[{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'Describe.'}]}],
        add_generation_prompt=True,
        **chat_template_kwargs(False, resolve_model_type(path)),
    )
    assert '<|vision_start|><|image_pad|><|vision_end|>' in rendered
    assert rendered.endswith('<|im_start|>assistant\n')
    assert '<think>' not in rendered


@pytest.mark.parametrize('matching', [False, True])
def test_teacher_vocabulary_validation(matching):
    first = SimpleNamespace(get_vocab=lambda: {'a': 0, 'b': 1}, bos_token_id=0, eos_token_id=1, pad_token_id=1)
    second = SimpleNamespace(get_vocab=lambda: {'a': 0, 'b': 1} if matching else {'a': 1, 'b': 0},
                             bos_token_id=0, eos_token_id=1, pad_token_id=1)
    tokenizer = SimpleNamespace(from_pretrained=Mock(side_effect=[first, second]))
    with patch.dict('sys.modules', {'transformers': SimpleNamespace(AutoTokenizer=tokenizer)}):
        if matching:
            validate_distillation_pair('/student', '/teacher')
        else:
            with pytest.raises(ValueError, match='token-to-ID'):
                validate_distillation_pair('/student', '/teacher')


@pytest.mark.parametrize('model_type,thinking,custom_template', [
    ('qwen2_5_vl', 'false', False), ('qwen2_5_vl', 'true', False),
    ('qwen3_vl', 'false', True), ('qwen3_vl', 'true', False),
])
def test_training_launcher_template_routing(tmp_path, model_type, thinking, custom_template):
    import os
    import subprocess
    import sys

    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text(json.dumps({'model_type': model_type}))
    dataset = tmp_path / 'train.jsonl'
    dataset.write_text('{}\n')
    captured = tmp_path / 'arguments.json'
    shim = tmp_path / 'python-shim'
    shim.write_text(f'''#!{sys.executable}
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[:2] == ['-m', 'opd_rag.thinking']:
    os.execv(sys.executable, [sys.executable, *args])
elif args[:2] == ['-m', 'verl.trainer.main_ppo']:
    Path(os.environ['CAPTURE_ARGS']).write_text(json.dumps(args[2:]))
elif args[0] == '-c' and 'teacher_manager.py' in args[1]:
    print(os.environ['TEACHER_MANAGER'])
elif args[0] != '-c':
    raise RuntimeError(args)
''')
    shim.chmod(0o755)
    env = {**os.environ, 'PYTHON_BIN': str(shim), 'MODEL_PATH': str(model),
           'TEACHER_MODEL_PATH': str(model), 'ENABLE_THINKING': thinking,
           'CAPTURE_ARGS': str(captured), 'PRINT_TRAIN_CONFIG': 'false',
           'SAVE_TRAIN_CONFIG': 'false', 'PLOT_TRAIN_METRICS': 'false',
           'TEACHER_MANAGER': str(ROOT / 'verl-runtime/verl/experimental/teacher_loop/teacher_manager.py')}
    subprocess.run(['bash', str(ROOT / 'scripts/run_opd.sh'), str(dataset), str(tmp_path / 'output')],
                   env=env, check=True, capture_output=True, text=True)
    args = json.loads(captured.read_text())
    assert any('custom_chat_template=' in arg for arg in args) == custom_template
    assert ('+data.apply_chat_template_kwargs.enable_thinking=true' in args) == (
        model_type == 'qwen3_vl' and thinking == 'true')
