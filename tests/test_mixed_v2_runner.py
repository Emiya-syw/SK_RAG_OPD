import json
from pathlib import Path

import pytest

from scripts import eval_mixed_v2_runner as runner
from scripts.eval_mixed_v2_profiles import detect_profile, settings


@pytest.mark.parametrize('kind,hidden,profile,batch', [
    ('qwen3_vl', 2048, 'qwen3vl-2b', 8),
    ('qwen3_vl', 2560, 'qwen3vl-4b', 4),
    ('qwen3_vl', 4096, 'qwen3vl-8b', 2),
    ('qwen2_5_vl', 3584, 'qwen25vl-7b', 2),
])
def test_profiles_use_config_for_renamed_models(tmp_path, kind, hidden, profile, batch):
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': kind, 'text_config': {'hidden_size': hidden}}))
    assert detect_profile(tmp_path) == profile
    assert settings(profile, 3, 'flash_attention_2') == batch
    assert settings(profile, 3, 'sdpa') == max(1, batch // 2)
    assert settings(profile, 10, 'flash_attention_2') == max(1, batch // 2)


def test_per_model_override(tmp_path):
    path = tmp_path / 'models.json'
    path.write_text(json.dumps([dict(run_name='test', base_model='/base', adapter_path='',
                                    profile='qwen3vl-4b', generation_batch_size=3)]))
    env = runner.model_environments(path, {})[0]
    assert env['GENERATION_BATCH_SIZE'] == '3'
    assert env['EVAL_PROFILE'] == 'qwen3vl-4b'


def test_model_pairs_have_separate_outputs(tmp_path):
    path = tmp_path / 'models.json'
    models = [dict(run_name=name, adapter_path=adapter, base_model=base)
              for name, adapter, base in [('A', '/a', '/base-a'), ('B', '/b', '/base-a'), ('C', '/c', '/base-b')]]
    path.write_text(json.dumps(models))
    envs = runner.model_environments(path, {'MODELS_FILE': str(path), 'RESULTS_ROOT': str(tmp_path)})
    assert [e['BASE_MODEL'] for e in envs] == ['/base-a', '/base-a', '/base-b']
    assert [e['ADAPTER_PATH'] for e in envs] == ['/a', '/b', '/c']
    assert len({e['OUTPUT_DIR'] for e in envs}) == 3
    assert all('MODELS_FILE' not in e for e in envs)
    path.write_text(json.dumps([models[0], models[0]]))
    with pytest.raises(ValueError, match='duplicate'):
        runner.model_environments(path, {})


def test_stream_scores_before_generation_finishes_and_drains_all(monkeypatch, tmp_path):
    datasets = ['MM-SafetyBench', 'VLGuard', 'MSSBench', 'SPA-VL', 'SIUO-Gen', 'VLSU', 'VLSBench']
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    for key, value in dict(DATASETS=' '.join(datasets), OUTPUT_DIR=str(tmp_path / 'results'),
                           DATA_ROOT=str(tmp_path / 'data'), RUN_NAME='test',
                           STREAM_POLL_SECONDS='0.01', LIMIT='0').items():
        monkeypatch.setenv(key, value)
    for name in datasets:
        directory = tmp_path / 'data' / name
        directory.mkdir(parents=True)
        (directory / 'test.jsonl').write_text('{"id":"a"}\n{"id":"b"}\n')
    scripts = tmp_path / 'rag_eval'
    scripts.mkdir()
    (scripts / 'run_rag_eval.sh').write_text('''#!/bin/bash
python3 - <<'PY'
import os, json, time
from pathlib import Path
root = Path(os.environ['OUTPUT_DIR'])
for name in os.environ['DATASETS'].split():
    p = root/name/'answers.rank0of1.jsonl'
    p.parent.mkdir(parents=True)
    p.write_text('{"id":"a","prediction":"first"}\\n')
# Generation can only finish after the judge has seen the first answers.
deadline = time.monotonic() + 10
while not (root/'judge_started').exists():
    if time.monotonic() > deadline: raise RuntimeError('Judge did not overlap')
    time.sleep(.01)
for name in os.environ['DATASETS'].split():
    with (root/name/'answers.rank0of1.jsonl').open('a') as f:
        f.write('{"id":"b","prediction":"second"}\\n')
PY
''')
    (scripts / 'run_mssbench_protocol_eval.sh').write_text('''#!/bin/bash
python3 - <<'PY'
import os, json
from pathlib import Path
root = Path(os.environ['OUTPUT_DIR'])
counts = {name: len((Path(os.environ['INPUT_ROOT'])/name/'answers.jsonl').read_text().splitlines()) for name in os.environ['DATASETS'].split()}
with (root/'judge_calls.jsonl').open('a') as f: f.write(json.dumps(counts)+'\\n')
(root/'judge_started').touch()
PY
''')
    runner.stream()
    calls = [json.loads(line) for line in (tmp_path / 'results/judge_calls.jsonl').read_text().splitlines()]
    assert calls[0] == dict.fromkeys(datasets, 1)
    assert calls[-1] == dict.fromkeys(datasets, 2)
