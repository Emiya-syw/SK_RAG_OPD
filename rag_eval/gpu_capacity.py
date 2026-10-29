"""Select GPUs at dataset boundaries while preserving partial shard assignments."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from datetime import datetime, timezone


def eligible_gpus(output, min_free_mib):
    selected = []
    for line in output.strip().splitlines():
        index, free = [part.strip() for part in line.split(',')]
        if int(free) >= min_free_mib:
            selected.append(index)
    return selected


def choose(directory, requested, policy):
    allocation = directory / 'gpu_allocation.json'
    partial = [p for p in directory.glob('answers.rank*of*.jsonl') if p.stat().st_size]
    if partial:
        sizes = {int(re.search(r'of(\d+)\.jsonl$', p.name)[1]) for p in partial}
        assigned = json.loads(allocation.read_text())['gpu_ids'] if allocation.exists() else requested
        if sizes != {len(assigned)}:
            raise ValueError('Existing shard count differs; refusing to mix partitions')
        return assigned, 'preserved partial dataset allocation'
    while True:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free',
                                 '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
        selected = eligible_gpus(result.stdout, policy['min_free_mib'])
        if len(selected) >= policy.get('min_gpus', 2):
            directory.mkdir(parents=True, exist_ok=True)
            record = {'time': datetime.now(timezone.utc).isoformat(), 'gpu_ids': selected,
                      'min_free_mib': policy['min_free_mib'], 'memory_snapshot': result.stdout.strip()}
            allocation.write_text(json.dumps(record, indent=2))
            return selected, 'fresh GPU capacity check'
        import sys
        print('Waiting for enough GPUs with free memory at dataset boundary', file=sys.stderr, flush=True)
        time.sleep(policy.get('poll_seconds', 30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--requested', required=True)
    args = parser.parse_args()
    if len(args.datasets) != 1:
        raise ValueError('Capacity scheduling requires one dataset per invocation')
    policy = json.loads(args.policy.read_text())
    if os.getenv('GPU_MIN_FREE_MIB'):
        policy['min_free_mib'] = int(os.environ['GPU_MIN_FREE_MIB'])
    selected, reason = choose(args.output_root / args.datasets[0], args.requested.split(','), policy)
    import sys
    print(f'GPU allocation: {",".join(selected)} ({reason})', file=sys.stderr, flush=True)
    print(','.join(selected))


if __name__ == '__main__':
    main()
