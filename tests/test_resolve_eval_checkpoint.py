import json

import pytest

from scripts.resolve_eval_checkpoint import experiment, latest_actor, exported_adapter, base_model


def test_v2_alias_and_numeric_latest_checkpoint(tmp_path):
    directory = tmp_path / 'outputs/run'
    for step in (9, 100):
        (directory / f'global_step_{step}/actor').mkdir(parents=True)
    assert experiment(tmp_path, 'run_v2') == directory
    assert latest_actor(directory) == directory / 'global_step_100/actor'
    exact = tmp_path / 'outputs/run_v2'
    exact.mkdir()
    assert experiment(tmp_path, 'run_v2') == exact


def test_relocated_adapter_base_wins_over_teacher(tmp_path):
    directory = tmp_path / 'outputs/run'
    adapter = directory / 'global_step_100/actor/huggingface_merged/lora_adapter'
    adapter.mkdir(parents=True)
    (adapter / 'adapter_model.safetensors').write_bytes(b'weights')
    (adapter / 'adapter_config.json').write_text(json.dumps({'base_model_name_or_path': '/old/models/Qwen3-VL-2B-Thinking'}))
    base = tmp_path / 'models/Qwen3-VL-2B-Thinking'
    base.mkdir(parents=True)
    (base / 'config.json').write_text('{}')
    (directory / 'training_config.txt').write_text('TEACHER_MODEL_PATH = /teacher-8b\n')
    assert exported_adapter(latest_actor(directory)) == adapter
    assert base_model(tmp_path, adapter, directory) == base


def test_training_config_fallback_and_no_size_guess(tmp_path):
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    (adapter / 'adapter_config.json').write_text('{}')
    (tmp_path / 'training_config.txt').write_text('TEACHER_MODEL_PATH = /teacher\n')
    with pytest.raises(FileNotFoundError, match='student base'):
        base_model(tmp_path, adapter, tmp_path)
    base = tmp_path / 'models/Qwen3-VL-4B-Instruct'
    base.mkdir(parents=True)
    (base / 'config.json').write_text('{}')
    (tmp_path / 'training_config.txt').write_text('MODEL_PATH = Qwen/Qwen3-VL-4B-Instruct\nTEACHER_MODEL_PATH = /teacher\n')
    assert base_model(tmp_path, adapter, tmp_path) == base


def test_latest_checkpoint_never_silently_uses_older_export(tmp_path):
    (tmp_path / 'global_step_10/actor').mkdir(parents=True)
    (tmp_path / 'global_step_20').mkdir()
    with pytest.raises(FileNotFoundError, match='Latest checkpoint'):
        latest_actor(tmp_path)
