"""Keep one model worker per GPU; publish each dataset as soon as its shards finish."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

from rag_eval.merge_answer_shards import merge_dataset
from rag_eval.process_utils import handle_termination, stop_processes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-ids', required=True)
    parser.add_argument('--datasets', nargs='+', required=True)
    args = parser.parse_args()
    gpu_ids = args.gpu_ids.split(',')
    if any(not x.strip() for x in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        parser.error('GPU IDs must be nonempty and unique')
    datasets = args.datasets
    if 'all' in [d.lower() for d in datasets]:
        from rag_eval.data import DEFAULT_DATA_ROOT, select_datasets
        datasets = [d.name for d in select_datasets(datasets, Path(os.environ.get('DATA_ROOT', DEFAULT_DATA_ROOT)))]
    if any(Path(d).name != d or d in ('.', '..') for d in datasets):
        parser.error('Invalid dataset name')
    ready_dir = os.environ.get('GENERATION_READY_DIR')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, handle_termination)
    workers = []
    # Fresh directory for each invocation: old shard files are never completion signals.
    with tempfile.TemporaryDirectory(prefix='.generation-shards-', dir=args.output_dir) as folder:
        completion_dir = Path(folder)
        try:
            for index, gpu in enumerate(gpu_ids):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, MULTI_GPU='false',
                           GENERATION_COMPLETION_DIR=folder, DATASETS=' '.join(datasets))
                workers.append(subprocess.Popen(
                    ['bash', str(args.script), '--internal-multi-gpu-shard', str(index), str(len(gpu_ids))],
                    env=env, start_new_session=True,
                ))
            pending = list(datasets)
            while pending or any(p.poll() is None for p in workers):
                for index, worker in enumerate(workers):
                    status = worker.poll()
                    if status is not None and status != 0:
                        raise RuntimeError(f'Generation worker {index} failed with exit code {status}')
                for dataset in pending[:]:
                    markers = [completion_dir / dataset / f'{i}.done' for i in range(len(workers))]
                    if all(marker.is_file() for marker in markers):
                        merge_dataset(args.output_dir / dataset, len(workers))
                        if ready_dir:
                            target = Path(ready_dir) / f'{dataset}.ready'
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.touch()
                        pending.remove(dataset)
                    elif any(p.poll() == 0 and not markers[i].is_file() for i, p in enumerate(workers)):
                        raise RuntimeError(f'A worker exited without completing dataset {dataset}')
                if pending or any(p.poll() is None for p in workers):
                    time.sleep(0.2)
        finally:
            stop_processes(workers)


if __name__ == '__main__':
    main()
