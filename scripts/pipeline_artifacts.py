"""Read-only artifact checks used by train_eval_pipeline.sh (stdlib only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys


class IncompatibleAnswers(ValueError):
    pass


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def adapter_at(path):
    path = Path(path)
    config = path / 'adapter_config.json'
    if not config.is_file() or not isinstance(json.loads(config.read_text()), dict):
        return None
    if any((path / name).is_file() and (path / name).stat().st_size > 0
           for name in ('adapter_model.safetensors', 'adapter_model.bin')):
        return path
    return None


def latest_step(root):
    steps = [p for p in Path(root).glob('global_step_*') if p.is_dir() and re.fullmatch(r'global_step_\d+', p.name)]
    return max(steps, key=lambda p: int(p.name.rsplit('_', 1)[1])) if steps else None


def exported(root):
    step = latest_step(root)
    if step:
        for config in sorted((step / 'actor').rglob('adapter_config.json')):
            if adapter_at(config.parent):
                return config.parent
    return None


def checkpoint(root):
    step = latest_step(root)
    if not step:
        return None
    if exported(root):
        return step
    shards = list((step / 'actor').glob('model_world_size_*_rank_*.pt'))
    matches = [re.fullmatch(r'model_world_size_(\d+)_rank_(\d+)\.pt', p.name) for p in shards]
    if matches:
        world = int(matches[0][1])
        if (len(shards) == world and all(int(m[1]) == world for m in matches)
                and {int(m[2]) for m in matches} == set(range(world))
                and all(p.stat().st_size > 0 for p in shards)):
            return step
    return None


def answers_complete(source, answers, adapter):
    expected = read_rows(source)
    rows = read_rows(answers)
    ids = {str(row['id']) for row in expected}
    for row in rows:
        if not isinstance(row.get('prediction'), str):
            return False
        if row.get('prompt_mode', 'guided') != os.environ.get('PROMPT_MODE', 'guided'):
            raise IncompatibleAnswers('Existing answers use a different PROMPT_MODE; choose a fresh OUTPUT_DIR')
        if row.get('base_model') and Path(row['base_model']).resolve() != Path(os.environ['MODEL_PATH']).resolve():
            raise IncompatibleAnswers('Existing answers use a different student model; choose a fresh OUTPUT_DIR')
        if adapter and (not row.get('adapter') or Path(row['adapter']).resolve() != Path(adapter).resolve()):
            raise IncompatibleAnswers('Existing answers use a different adapter; choose a fresh OUTPUT_DIR')
    return bool(ids) and len(ids) == len(expected) == len(rows) and {str(row['id']) for row in rows} == ids


def judged_complete(answers, directory):
    rows = read_rows(answers)
    limit = int(os.environ.get('LIMIT', '0'))
    rows = [r for r in rows if str(r.get('prediction') or r.get('raw_prediction') or r.get('response') or '').strip()]
    if limit:
        rows = rows[:limit]
    ids = {str(r['id']) for r in rows}
    summary = json.loads((Path(directory) / 'summary.json').read_text())
    judgments = {str(r['id']): r for r in read_rows(Path(directory) / 'judgments.jsonl')}
    if (summary.get('dry_run') or summary.get('judge_model') != os.environ.get('JUDGE_MODEL', 'qwen3.6-flash')
            or summary.get('total') != len(rows) or summary.get('judged') != len(rows) or summary.get('errors') != 0):
        return False
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rag_eval.evaluate_safety_alignment_outputs import MSS_CHAT_JUDGE_PROMPT, MSS_EMBODIED_JUDGE_PROMPT

    for row in rows:
        judgment = judgments.get(str(row['id']), {})
        answer = str(row.get('prediction') or row.get('raw_prediction') or row.get('response') or '')
        task_type = str((row.get('metadata') or {}).get('task_type') or row.get('subset') or 'chat')
        template = MSS_EMBODIED_JUDGE_PROMPT if task_type == 'embodied' else MSS_CHAT_JUDGE_PROMPT
        prompt = template.format(answer=answer.split('Response to instruction:')[-1])
        if (judgment.get('status') != 'valid' or (judgment.get('parsed') or {}).get('label') not in (0, 1, 2, 3)
                or judgment.get('judge_model') != summary['judge_model']
                or judgment.get('judge_prompt_sha256') != hashlib.sha256(prompt.encode()).hexdigest()):
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('checkpoint', 'adapter', 'valid-adapter', 'answers', 'judge', 'summarize'))
    parser.add_argument('paths', nargs='+')
    args = parser.parse_args()
    try:
        if args.action == 'summarize':
            root, *datasets = args.paths
            results = [json.loads((Path(root) / d / 'summary.json').read_text()) for d in datasets]
            target = Path(root) / 'summary.json'
            tmp = target.with_suffix('.json.tmp')
            tmp.write_text(json.dumps({'results': results}, ensure_ascii=False, indent=2))
            tmp.replace(target)
            return
        functions = {'checkpoint': checkpoint, 'adapter': exported, 'valid-adapter': adapter_at,
                     'answers': answers_complete, 'judge': judged_complete}
        result = functions[args.action](*args.paths)
        if not result:
            sys.exit(1)
        if isinstance(result, Path):
            print(result)
    except IncompatibleAnswers as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'Artifact incomplete: {exc}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
