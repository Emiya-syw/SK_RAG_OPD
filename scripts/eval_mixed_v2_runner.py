"""Model-list scheduling and streaming judging for eval_mixed_top10_v2.sh."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from rag_eval.process_utils import handle_termination, stop_processes
from scripts.evaluate_safe_models import available_answers

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / 'scripts/eval_mixed_top10_v2.sh'


def model_environments(path, environ):
    models = json.loads(path.read_text())
    if not isinstance(models, list) or not models:
        raise ValueError('Models file must be a nonempty JSON list')
    environments, names = [], set()
    for item in models:
        name = item['run_name']
        if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..') or name in names:
            raise ValueError(f'Invalid or duplicate run_name: {name}')
        names.add(name)
        for key in ('base_model', 'adapter_path'):
            if not isinstance(item[key], str):
                raise ValueError(f'{name}: {key} must be a string')
        if not item['base_model']:
            raise ValueError(f'{name}: base_model is empty')
        env = dict(environ)
        env.pop('MODELS_FILE', None)
        env.update(RUN_NAME=name, BASE_MODEL=item['base_model'], ADAPTER_PATH=item['adapter_path'],
                   OUTPUT_DIR=str(Path(env.get('RESULTS_ROOT', ROOT / 'rag_eval/results')) / name),
                   EVAL_OUTPUT_ROOT=str(Path(env.get('SCORES_ROOT', ROOT / 'rag_eval/evaluation')) / name / 'mssbench_protocol'))
        # Per-model overrides take priority over global defaults.
        if 'profile' in item:
            env['EVAL_PROFILE'] = str(item['profile'])
        if 'generation_batch_size' in item:
            batch = item['generation_batch_size']
            if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
                raise ValueError(f'{name}: generation_batch_size must be a positive integer')
            env['GENERATION_BATCH_SIZE'] = str(batch)
        environments.append(env)
    return environments


def run_models(path):
    environments = model_environments(path, os.environ)
    # Check every combination before starting expensive work on the first model.
    for env in environments:
        subprocess.run(['bash', str(WRAPPER)], env=dict(env, DRY_RUN='true'), check=True)
    if os.environ.get('DRY_RUN') == 'true':
        return
    for env in environments:
        print(f'[MODEL] Starting {env["RUN_NAME"]}', flush=True)
        process = subprocess.Popen(['bash', str(WRAPPER)], env=env, start_new_session=True)
        try:
            if process.wait():
                raise RuntimeError(f'Model evaluation failed: {env["RUN_NAME"]}')
        finally:
            stop_processes([process])


def stream():
    env = os.environ
    datasets = env['DATASETS'].split()
    results = Path(env['OUTPUT_DIR'])
    staging = results / 'stream_judge_inputs'
    expected = {}
    for name in datasets:
        rows = [json.loads(line) for line in (Path(env['DATA_ROOT']) / name / 'test.jsonl').read_text().splitlines() if line.strip()]
        limit = int(env.get('LIMIT', '0'))
        expected[name] = [row['id'] for row in (rows[:limit] if limit else rows)]
    interval = float(env.get('STREAM_POLL_SECONDS', '5'))
    if interval <= 0:
        raise ValueError('STREAM_POLL_SECONDS must be positive')
    generation = judge = None
    judged_counts = {name: 0 for name in datasets}
    try:
        generation = subprocess.Popen(['bash', str(ROOT / 'rag_eval/run_rag_eval.sh')],
                                      env=dict(env, STAGE='generate'), start_new_session=True)
        while True:
            status = generation.poll()
            if status not in (None, 0):
                raise RuntimeError(f'Generation failed: exit {status}')
            done = status == 0
            counts, active = {}, []
            for name in datasets:
                rows = available_answers(results / name, set(expected[name]))
                if done and len(rows) != len(expected[name]):
                    raise ValueError(f'{name}: incomplete generation ({len(rows)}/{len(expected[name])})')
                counts[name] = len(rows)
                if not rows:
                    continue
                active.append(name)
                target = staging / name / 'answers.jsonl'
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix('.tmp')
                temporary.write_text(''.join(json.dumps(rows[i], ensure_ascii=False) + '\n' for i in expected[name] if i in rows))
                temporary.replace(target)
            if active and (done or counts != judged_counts):
                print(f'[STREAM] {env["RUN_NAME"]}: available={counts}; generation_done={done}', flush=True)
                judge_env = dict(env, DATASETS=' '.join(active), INPUT_ROOT=str(staging), LIMIT='0',
                                 MAX_IMAGE_PIXELS=env.get('JUDGE_MAX_IMAGE_PIXELS', '786432'))
                judge = subprocess.Popen(['bash', str(ROOT / 'rag_eval/run_mssbench_protocol_eval.sh')],
                                         env=judge_env, start_new_session=True)
                while judge.poll() is None:
                    if generation.poll() not in (None, 0):
                        raise RuntimeError('Generation failed while scoring')
                    time.sleep(min(interval, 1))
                if judge.returncode:
                    raise RuntimeError(f'Judge failed: exit {judge.returncode}')
                judge = None
                judged_counts = counts
            if done:
                print(f'[DONE] {env["RUN_NAME"]}: generation and scoring finished', flush=True)
                return
            time.sleep(interval)
    finally:
        stop_processes([judge, generation])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--models', type=Path)
    mode.add_argument('--stream', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, handle_termination)
    if args.models:
        run_models(args.models)
    else:
        stream()


if __name__ == '__main__':
    main()
