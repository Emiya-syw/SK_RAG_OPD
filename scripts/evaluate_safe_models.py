"""Run five local safety baselines on frozen test splits with MSSBench judging."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path('/home/sunyw/safety_alignment_models')
MODELS = {
    'safeguard-vl-rl': 'tyodd__SafeGuard-VL-RL',
    'safeqwen2.5-vl-7b': 'etri-vilab__SafeQwen2.5-VL-7B',
    'safework-r1-qwen2.5vl-7b': 'AI45Research__SafeWork-R1-Qwen2.5VL-7B',
    'qwen2.5-vl-7b-mm-rlhf': 'HappyCorpse__Qwen2.5-VL-7B-MM-RLHF',
    'qwen2.5-vl-7b-spavl': 'Qwen2.5-VL-7B-SPAVL/checkpoint-5000',
}
DATASETS = ['SIUO-Gen', 'SPA-VL', 'VLSU', 'VLSBench']


def available_answers(directory, expected_ids):
    """Read completed JSONL records only, ignoring a writer's partial final line."""
    merged = directory / 'answers.jsonl'
    paths = [merged] if merged.exists() else sorted(directory.glob('answers.rank*.jsonl'))
    rows = {}
    for path in paths:
        payload = path.read_bytes()
        lines = payload.splitlines(keepends=True)
        for line in lines:
            if not line.endswith(b'\n'):
                continue
            row = json.loads(line)
            if row['id'] not in expected_ids:
                raise ValueError('Generated row does not belong to the frozen test split')
            if row['id'] in rows:
                raise ValueError('Duplicate ID across generation shards')
            rows[row['id']] = row
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['smoke', 'generate', 'judge', 'all', 'stream'], default='smoke')
    parser.add_argument('--output-root', type=Path, default=ROOT/'rag_eval/results/safe_models_20261028_3gpu')
    parser.add_argument('--gpu-ids', default='2,4,6')
    parser.add_argument('--models', nargs='+', choices=list(MODELS), default=list(MODELS))
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output/'data_snapshot'
    for name in DATASETS:
        directory = snapshot/name
        if not directory.exists():
            directory.mkdir(parents=True)
            for filename in ['test.jsonl', 'retrieval.jsonl', 'split_summary.json', 'preparation_manifest.json']:
                source = ROOT/'rag_eval/data'/name/filename
                # SIUO has an annotated pool instead of retrieval.jsonl.
                if filename == 'retrieval.jsonl' and not source.exists():
                    source = source.with_name('retrieval_qwen3.7_plus_merged.jsonl')
                if source.exists():
                    shutil.copy2(source, directory/filename)
    settings = {'models': {k: str(MODELS_ROOT/v) for k,v in MODELS.items()},
                'datasets': {name: {'sha256': hashlib.sha256((snapshot/name/'test.jsonl').read_bytes()).hexdigest(),
                                    'count': sum(1 for _ in (snapshot/name/'test.jsonl').open())} for name in DATASETS},
                'max_new_tokens': args.max_new_tokens, 'batch_size': args.batch_size,
                'max_image_pixels': 262144, 'do_sample': False, 'mode': 'direct',
                'gpu_ids': args.gpu_ids, 'attention': 'flash_attention_2',
                'schedule': 'sequential models, sequential datasets, multi-GPU shards per pair',
                'judge_protocol': 'MSSBench', 'judge_model': os.getenv('JUDGE_MODEL', 'qwen3.6-flash')}
    settings_path = output/'run_config.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('Run configuration changed; use a fresh --output-root')
    settings_path.write_text(json.dumps(settings, indent=2))
    logs = output/'logs'
    logs.mkdir(exist_ok=True)
    gpus = args.gpu_ids.split(',')
    if len(set(gpus)) != len(gpus):
        raise ValueError('GPU IDs must be distinct')
    if args.stage in {'judge', 'all', 'stream'} and not os.getenv('DASHSCOPE_API_KEY'):
        raise ValueError('Configure DASHSCOPE_API_KEY for online scoring')

    active = {}

    def run(name, dataset, stage, judge_input=None):
        env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4'}
        tuning_path = output/'performance_policy.json'
        tuning = json.loads(tuning_path.read_text()) if tuning_path.exists() else {}
        model_tuning = tuning.get('models', {}).get(name, {})
        if tuning.get('adaptive_judge'):
            env['JUDGE_ADAPTIVE'] = 'true'
        result = output/('smoke' if stage == 'smoke' else 'generation')/name
        if stage == 'generate':
            answers = result/dataset/'answers.jsonl'
            if answers.is_file():
                actual = [json.loads(line) for line in answers.read_text().splitlines() if line.strip()]
                expected = [json.loads(line) for line in (snapshot/dataset/'test.jsonl').read_text().splitlines() if line.strip()]
                if len(actual) == len(expected) and {r['id'] for r in actual} == {r['id'] for r in expected}:
                    print(f'Skip complete generation: {name}/{dataset}', flush=True)
                    return {'model': name, 'dataset': dataset, 'stage': stage, 'exit_code': 0, 'reused': True}
        if stage == 'judge':
            command = [sys.executable, '-m', 'rag_eval.evaluate_mssbench_protocol',
                       '--input-root', str(judge_input or result), '--output-root', str(output/'evaluation'/name),
                       '--datasets', dataset, '--judge-model', settings['judge_model'],
                       '--workers', str(tuning.get('judge_workers', 4)), '--flush-every', '4']
        else:
            env.update(PYTHON_BIN=sys.executable, BASE_MODEL=str(MODELS_ROOT/MODELS[name]),
                       ADAPTER_PATH='', DATASETS=dataset, DATA_ROOT=str(snapshot), OUTPUT_DIR=str(result),
                       STAGE='generate', DIRECT_GENERATION='true', MULTI_GPU='true', GPU_IDS=args.gpu_ids,
                       GENERATION_BATCH_SIZE=str(model_tuning.get('batch_size', args.batch_size)), MAX_NEW_TOKENS=str(args.max_new_tokens),
                       MAX_IMAGE_PIXELS='262144', EVAL_ATTN_IMPLEMENTATION='flash_attention_2',
                       EVAL_DEVICE_MAP='cuda:0',
                       DTYPE='bfloat16', PROMPT_MODE='guided', ENABLE_THINKING='false',
                       LIMIT=str(len(gpus)) if stage == 'smoke' else '0')
            if 'min_free_mib' in model_tuning:
                env['GPU_MIN_FREE_MIB'] = str(model_tuning['min_free_mib'])
            print(f'[TUNING] {name}/{dataset}: batch={env["GENERATION_BATCH_SIZE"]}, min_free_mib={env.get("GPU_MIN_FREE_MIB", "policy default")}', flush=True)
            command = ['bash', str(ROOT/'rag_eval/run_rag_eval.sh')]
        log_path = logs/f'{stage}-{name}-{dataset}.log'
        print(f'Start {stage}: {name}/{dataset}; log={log_path}', flush=True)
        with log_path.open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[stage] = process
            status = process.wait()
            active.pop(stage, None)
        print(f'Finished {name}/{dataset}: exit={status}', flush=True)
        return {'model': name, 'dataset': dataset, 'stage': stage, 'exit_code': status, 'log': str(log_path)}

    statuses = []
    for name in args.models:
        for dataset in args.datasets:
            if args.stage == 'stream':
                expected = [json.loads(line)['id'] for line in (snapshot/dataset/'test.jsonl').read_text().splitlines() if line.strip()]
                expected_ids = set(expected)
                existing_summary = output/'evaluation'/name/dataset/'summary.json'
                if existing_summary.exists():
                    summary = json.loads(existing_summary.read_text())
                    available = available_answers(output/'generation'/name/dataset, expected_ids)
                    if len(available) == len(expected) and summary.get('total') == len(expected):
                        print(f'[STREAM] Reuse completed pair: {name}/{dataset}; recorded judge errors={summary.get("errors", 0)}', flush=True)
                        continue
                judged_count = -1
                staging = output/'judge_inputs'/name
                target = staging/dataset/'answers.jsonl'
                target.parent.mkdir(parents=True, exist_ok=True)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    generation = executor.submit(run, name, dataset, 'generate')
                    try:
                        while True:
                            done = generation.done()
                            if done and generation.result()['exit_code']:
                                raise RuntimeError('Generation failed during concurrent scoring')
                            rows = available_answers(output/'generation'/name/dataset, expected_ids)
                            if done and len(rows) != len(expected):
                                raise ValueError('Completed generation does not cover the entire frozen test set')
                            if rows and (done or len(rows) >= judged_count + 64):
                                temporary = target.with_suffix('.tmp')
                                temporary.write_text(''.join(json.dumps(rows[i], ensure_ascii=False)+'\n' for i in expected if i in rows))
                                temporary.replace(target)
                                print(f'[STREAM] {name}/{dataset}: judge {len(rows)}/{len(expected)} available answers; generation continues', flush=True)
                                status = run(name, dataset, 'judge', staging)
                                statuses.append(status)
                                if status['exit_code']:
                                    raise RuntimeError('Concurrent scoring failed')
                                judged_count = len(rows)
                            if done:
                                statuses.append(generation.result())
                                break
                            time.sleep(5)
                    finally:
                        if 'generate' in active and active['generate'].poll() is None:
                            active['generate'].terminate()
                (output/'stream_status.json').write_text(json.dumps(statuses, indent=2))
                continue
            for stage in (['generate', 'judge'] if args.stage == 'all' else [args.stage]):
                statuses.append(run(name, dataset, stage))
                (output/f'{args.stage}_status.json').write_text(json.dumps(statuses, indent=2))
                if statuses[-1]['exit_code']:
                    raise SystemExit(1)


if __name__ == '__main__':
    main()
