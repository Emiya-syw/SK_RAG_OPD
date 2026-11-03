"""Write paper-compatible S/U/Avg for the six 4B EJSD component runs."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {'MSSBench', 'VLGuard', 'MM-SafetyBench', 'SIUO-Gen', 'SPA-VL', 'VLSBench', 'VLSU'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores-root', type=Path, default=ROOT/'rag_eval/evaluation/ejsd_components_4b')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/ejsd_components_4b.tsv')
    args = parser.parse_args()
    output = []
    for method in ('opsd', 'opd'):
        for variant in ('wo_entropy', 'wo_coverage', 'wo_mass_gate'):
            name = f'{method}_s4b_ejsd_{variant}_stage2_controlled'
            path = args.scores_root/name/'mssbench_protocol/summary.json'
            row = {'method': method.upper(), 'variant': variant, 'setting': {
                'wo_entropy': 'E=1', 'wo_coverage': 'D=1', 'wo_mass_gate': 'rho=1'}[variant],
                'S': '', 'U': '', 'Avg': '', 'judged': '', 'judge_errors': '', 'status': 'pending'}
            if path.exists():
                results = json.loads(path.read_text())['results']
                by = {item['dataset']: item for item in results}
                if set(by) == DATASETS:
                    safe = [by['MSSBench']['by_task_type'][group]['safe_accuracy_pct']
                            for group in ('chat', 'embodied')]
                    safe += [by[dataset]['by_task_type']['all']['safe_accuracy_pct']
                             for dataset in ('VLGuard', 'VLSU')]
                    unsafe = [by['MSSBench']['by_task_type'][group]['unsafe_accuracy_pct']
                              for group in ('chat', 'embodied')]
                    unsafe += [by[dataset]['by_task_type']['all']['unsafe_accuracy_pct']
                               for dataset in ('VLGuard', 'MM-SafetyBench', 'SIUO-Gen',
                                               'SPA-VL', 'VLSBench', 'VLSU')]
                    if all(value is not None for value in (*safe, *unsafe)):
                        s = sum(safe)/len(safe)
                        u = sum(unsafe)/len(unsafe)
                        row.update(S=f'{s:.2f}', U=f'{u:.2f}', Avg=f'{(s+u)/2:.2f}',
                                   judged=sum(item['judged'] for item in results),
                                   judge_errors=sum(item['errors'] for item in results),
                                   status='scored')
            output.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(output)
    print(args.output)


if __name__ == '__main__':
    main()
