"""Overlap GPU generation with one ordered dataset judging queue."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from rag_eval.process_utils import handle_termination, stop_processes

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', required=True)
    args = parser.parse_args()
    pending_generation = set(os.environ['DATASETS'].split())
    results = Path(os.environ['OUTPUT_DIR'])
    scores = Path(os.environ['EVAL_OUTPUT_ROOT'])
    python = os.environ.get('PYTHON_BIN', sys.executable)
    artifacts = [python, str(ROOT / 'scripts/pipeline_artifacts.py')]
    results.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, handle_termination)
    generation = judge = None
    with tempfile.TemporaryDirectory(prefix='.judge-queue-', dir=results) as folder:
        try:
            generation = subprocess.Popen(
                ['bash', str(ROOT / 'rag_eval/run_rag_eval.sh')],
                env=dict(os.environ, GENERATION_READY_DIR=folder), start_new_session=True,
            )

            def check_generation():
                status = generation.poll()
                if status is not None and status != 0:
                    raise RuntimeError(f'Generation failed with exit code {status}; stopping judge queue')

            for dataset in args.datasets:
                if dataset in pending_generation:
                    print(f'[QUEUE] Waiting for merged answers: {dataset}', flush=True)
                    marker = Path(folder) / f'{dataset}.ready'
                    while not marker.is_file():
                        check_generation()
                        if generation.poll() is not None:
                            raise RuntimeError(f'Generation exited without publishing {dataset}')
                        time.sleep(0.2)
                check_generation()
                complete = subprocess.run(
                    artifacts + ['judge', str(results / dataset / 'answers.jsonl'), str(scores / dataset)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                ).returncode == 0
                if complete:
                    print(f'[SKIP 4/4] {dataset} scoring complete', flush=True)
                    continue
                if not os.environ.get('DASHSCOPE_API_KEY'):
                    raise RuntimeError('Set DASHSCOPE_API_KEY to score unfinished datasets')
                print(f'[QUEUE] Scoring {dataset}; GPU generation continues independently', flush=True)
                judge_env = dict(os.environ, DATASETS=dataset, INPUT_ROOT=str(results),
                                 MAX_IMAGE_PIXELS=os.environ.get('JUDGE_MAX_IMAGE_PIXELS', '786432'))
                judge = subprocess.Popen(
                    ['bash', str(ROOT / 'rag_eval/run_mssbench_protocol_eval.sh')],
                    env=judge_env, start_new_session=True,
                )
                while judge.poll() is None:
                    check_generation()
                    time.sleep(0.2)
                if judge.returncode:
                    raise RuntimeError(f'Judging {dataset} failed with exit code {judge.returncode}')
                print(f'[QUEUE] Finished judging {dataset}', flush=True)
                judge = None
            while generation.poll() is None:
                time.sleep(0.2)
            check_generation()
        finally:
            stop_processes([judge, generation])


if __name__ == '__main__':
    main()
