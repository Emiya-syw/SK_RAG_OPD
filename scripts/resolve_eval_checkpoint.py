"""Resolve an evaluation run name to a local exported adapter and its base."""
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def experiment(root, name):
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('RUN_NAME must be a single experiment name')
    candidates = [root / 'outputs' / name]
    if name.endswith('_v2'):
        candidates.append(root / 'outputs' / name[:-3])
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f'No experiment directory found: {candidates}')


def latest_actor(directory):
    steps = [(int(p.name.removeprefix('global_step_')), p) for p in directory.glob('global_step_*')
             if p.is_dir() and re.fullmatch(r'global_step_\d+', p.name)]
    if not steps:
        raise FileNotFoundError(f'No global_step_* checkpoints in {directory}')
    actor = max(steps)[1] / 'actor'
    if not actor.is_dir():
        raise FileNotFoundError(f'Latest checkpoint has no actor directory: {actor}')
    return actor


def exported_adapter(actor):
    matches = [p.parent for p in actor.rglob('adapter_config.json')
               if any((p.parent / name).is_file() and (p.parent / name).stat().st_size
                      for name in ('adapter_model.safetensors', 'adapter_model.bin'))]
    preferred = actor / 'huggingface_merged/lora_adapter'
    if preferred in matches:
        return preferred
    if len(matches) > 1:
        raise ValueError(f'Multiple exported adapters found; set ADAPTER_PATH explicitly: {matches}')
    return matches[0] if matches else None


def base_model(root, adapter, directory, explicit=None):
    if explicit:
        if not Path(explicit).is_dir():
            raise FileNotFoundError(f'BASE_MODEL does not exist: {explicit}')
        return Path(explicit).resolve()
    metadata = json.loads((adapter / 'adapter_config.json').read_text())
    references = [metadata.get('base_model_name_or_path')]
    training = directory / 'training_config.txt'
    if training.is_file():
        for line in training.read_text().splitlines():
            match = re.match(r'^\s*MODEL_PATH\s*=\s*(.*?)\s*$', line)
            if match:
                values = shlex.split(match[1])
                if len(values) == 1:
                    references.append(values[0])
    for reference in references:
        if not isinstance(reference, str) or not reference:
            continue
        path = Path(reference)
        # Preserve exact existing paths; support a relocated identical model name.
        candidates = [path] if path.is_absolute() else [root / path]
        candidates.append(root / 'models' / path.name)
        for candidate in candidates:
            if (candidate / 'config.json').is_file():
                return candidate.resolve()
    raise FileNotFoundError(f'Cannot locate the student base model from {references}. '
                            'Place the recorded model under models/<model-name>, or set BASE_MODEL.')


def main():
    env = os.environ
    directory = experiment(ROOT, env['RUN_NAME'])
    actor = latest_actor(directory)
    adapter = exported_adapter(actor)
    if adapter is None:
        if env.get('DRY_RUN') == 'true':
            raise FileNotFoundError(f'{actor}: adapter requires export; rerun without DRY_RUN to export automatically')
        target = actor / 'huggingface_merged'
        print(f'[AUTO] Exporting {actor}', file=sys.stderr, flush=True)
        runtime_env = dict(env, PYTHONPATH=f'{ROOT / "verl-runtime"}:{ROOT}:{env.get("PYTHONPATH", "")}')
        subprocess.run([sys.executable, '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
                        '--local_dir', str(actor), '--target_dir', str(target)],
                       env=runtime_env, stdout=sys.stderr, check=True)
        adapter = exported_adapter(actor)
        if adapter is None:
            raise FileNotFoundError(f'Export produced no LoRA weights under {actor}')
    base = base_model(ROOT, adapter, directory, env.get('BASE_MODEL'))
    print(f'[AUTO] Experiment: {directory}\n[AUTO] LoRA: {adapter}\n[AUTO] Base: {base}', file=sys.stderr)
    for path in (base, adapter):
        if '\n' in str(path):
            raise ValueError('Paths containing newlines are unsupported')
        print(path)


if __name__ == '__main__':
    main()
