"""Evaluate Qwen3-VL 2B and 8B on seven frozen no-mixed Top-3 benchmarks."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import evaluate_safe_models as suite

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['MSSBench', 'VLGuard', 'MM-SafetyBench', 'SIUO-Gen', 'SPA-VL', 'VLSBench', 'VLSU']
MODELS = {
    'qwen3-vl-2b-thinking-no-thinking': '/home/sunyw/SK_RAG/models/Qwen3-VL-2B-Thinking',
    'qwen3-vl-8b-thinking-no-thinking': '/home/sunyw/SK_RAG/models/Qwen3-VL-8B-Thinking',
}
OUT = ROOT/'rag_eval/results/qwen2b8b_top3_no_mixed_20261103'

suite.MODELS_ROOT = Path('/')
suite.MODELS = MODELS
suite.DATASETS = DATASETS


def freeze(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f'Frozen input changed: {path}')
    else:
        path.write_bytes(payload)


def prepare():
    if not os.environ.get('DASHSCOPE_API_KEY'):
        secret = Path('/home/sunyw/training_free_vl_bench/.secrets/dashscope_api_key')
        os.environ['DASHSCOPE_API_KEY'] = secret.read_text().strip()
    manifest = {'models': MODELS, 'top_k': 3, 'enable_thinking': False,
                'max_new_tokens': 1024, 'datasets': {}}
    for name, model_path in MODELS.items():
        if not (Path(model_path)/'config.json').is_file():
            raise FileNotFoundError(f'{name}: {model_path}')
    for dataset in DATASETS:
        directory = ROOT/'rag_eval/data'/dataset
        source = directory/'test_no_mixed_top10.jsonl'
        raw = source.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        original_raw = (directory/'test.jsonl').read_bytes()
        original = [json.loads(line) for line in original_raw.splitlines() if line.strip()]
        if len(rows) != len(original) or not rows:
            raise ValueError(f'{dataset}: wrong test count')
        for row, expected in zip(rows, original):
            if any(row.get(key) != value for key, value in expected.items()):
                raise ValueError(f'{dataset}: test row changed: {expected["id"]}')
            if len(row.get('retrieval', [])) != 10 or not row['retrieval_config'].get('scene_rules_removed'):
                raise ValueError(f'{dataset}: invalid no-mixed retrieval: {row["id"]}')
        freeze(OUT/'retrieval_snapshot'/dataset/'retrieval.jsonl', raw)
        freeze(OUT/'data_snapshot'/dataset/'test.jsonl', original_raw)
        manifest['datasets'][dataset] = {'rows': len(rows), 'retrieval_sha256': hashlib.sha256(raw).hexdigest(),
                                         'source': str(source)}
    freeze(OUT/'input_manifest.json', (json.dumps(manifest, indent=2)+'\n').encode())
    freeze(OUT/'performance_policy.json', (json.dumps({'judge_workers': 8, 'adaptive_judge': True,
             'models': {name: {'batch_size': 1} for name in MODELS}}, indent=2)+'\n').encode())
    for relative in ['scripts/evaluate_qwen2b8b_no_mixed.py', 'scripts/evaluate_safe_models.py',
                     'rag_eval/run_rag_eval.sh', 'rag_eval/generate.py',
                     'rag_eval/evaluate_mssbench_protocol.py', 'opd_rag/thinking.py',
                     'opd_rag/chat_templates/qwen3_vl_thinking_off.jinja']:
        target = OUT/'code_snapshot'/relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/relative, target)
    print(f'Frozen {sum(x["rows"] for x in manifest["datasets"].values())} samples per model', flush=True)


def main():
    prepare()
    gpu_ids = os.environ['GPU_IDS']
    os.environ.update(AUTO_GPU_POLICY='false', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                      TOKENIZERS_PARALLELISM='false', JUDGE_MODEL='qwen3.6-flash',
                      HF_HUB_OFFLINE='1', JUDGE_ADAPTIVE='true')
    status = 1
    try:
        suite_args = ['--stage', 'stream', '--models', *MODELS, '--gpu-ids', gpu_ids,
                      '--batch-size', '1', '--max-new-tokens', '1024',
                      '--retrieval-root', str(OUT/'retrieval_snapshot'), '--output-root', str(OUT)]
        sys.argv = [sys.argv[0], *suite_args]
        suite.main()
        for name in MODELS:
            command = [sys.executable, '-m', 'rag_eval.evaluate_mssbench_protocol',
                       '--input-root', str(OUT/'generation'/name),
                       '--output-root', str(OUT/'evaluation'/name),
                       '--datasets', *DATASETS, '--judge-model', 'qwen3.6-flash', '--workers', '8']
            subprocess.run(command, cwd=ROOT, check=True)
            results = json.loads((OUT/'evaluation'/name/'summary.json').read_text())['results']
            with (OUT/f'metrics_{name}.tsv').open('w', newline='') as handle:
                writer = csv.writer(handle, delimiter='\t')
                writer.writerow(['dataset', 'subset', 'safe_pct', 'unsafe_pct', 'judge_errors'])
                for result in results:
                    subsets = ['chat', 'embodied'] if result['dataset'] == 'MSSBench' else ['all']
                    for subset in subsets:
                        metrics = result['by_task_type'][subset]
                        writer.writerow([result['dataset'], subset, metrics['safe_accuracy_pct'],
                                         metrics['unsafe_accuracy_pct'], result['errors']])
        status = 0
    finally:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT/'pipeline.exit').write_text(str(status)+'\n')


if __name__ == '__main__':
    main()
