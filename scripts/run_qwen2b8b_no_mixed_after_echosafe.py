"""Wait for EchoSafe 4B scoring, then run both no-mixed baselines on free GPUs."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
ECHO = Path('/home/sunyw/training_free_vl_bench')
DATASETS = ['MSSBench', 'VLGuard', 'MM-SafetyBench', 'SIUO-Gen', 'SPA-VL', 'VLSBench', 'VLSU']
OUT = ROOT/'rag_eval/results/qwen2b8b_top3_no_mixed_20261103'
STATUS = ROOT/'reports/qwen2b8b_no_mixed_scheduler_status.json'


def update(**values):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(values, indent=2)+'\n')
    print(values, flush=True)


def echosafe_done():
    exits = ECHO/'runs/parallel_echosafe_qwen3_4b/exit_status.tsv'
    if not exits.exists():
        return False
    rows = [line.split('\t') for line in exits.read_text().splitlines()]
    if len(rows) != 5 or any(len(row) != 3 or row[2] != '0' for row in rows):
        return False
    for dataset in DATASETS:
        expected = sum(1 for _ in (ECHO/'data_snapshot'/dataset/'test.jsonl').open())
        answers = ECHO/'runs/full_echosafe_qwen3_4b'/dataset/'answers.jsonl'
        summary = ECHO/'scoring/full_echosafe_qwen3_4b'/dataset/'summary.json'
        if not answers.exists() or not summary.exists():
            return False
        if sum(1 for _ in answers.open()) != expected or json.loads(summary.read_text()).get('total') != expected:
            return False
    return True


def free_gpus():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.free',
                                      '--format=csv,noheader,nounits'], text=True)
    free = []
    for line in output.splitlines():
        index, used, available = [int(item.strip()) for item in line.split(',')]
        if used < 1024 and available >= 21000:
            free.append(index)
    return free


def main():
    while not echosafe_done():
        update(state='waiting_for_echosafe_4b_generation_and_scoring')
        time.sleep(60)
    print('EchoSafe 4B generation and scoring complete.', flush=True)
    while True:
        gpus = free_gpus()
        if len(gpus) >= 2:
            break
        update(state='waiting_for_multiple_free_gpus', available_gpus=gpus)
        time.sleep(60)
    gpu_ids = ','.join(map(str, gpus))
    update(state='running', gpu_ids=gpu_ids)
    log_path = ROOT/'reports/qwen2b8b_no_mixed_pipeline.log'
    with log_path.open('a') as log:
        process = subprocess.run(['docker', 'exec', '-e', 'GPU_IDS='+gpu_ids,
                                  '-e', 'OMP_NUM_THREADS=4', '-e', 'TOKENIZERS_PARALLELISM=false',
                                  'syw_clip2', '/opt/conda/envs/qwen3vl/bin/python', '-u',
                                  str(ROOT/'scripts/evaluate_qwen2b8b_no_mixed.py')],
                                 stdout=log, stderr=subprocess.STDOUT)
    update(state='complete' if process.returncode == 0 else 'failed', gpu_ids=gpu_ids,
           exit_code=process.returncode, log=str(log_path))
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
