"""Recover precise LoRA targets from older veRL exports with numeric suffixes."""
from pathlib import Path
import re


def targets_from_weight_keys(keys, module_names):
    targets = set()
    for key in keys:
        match = re.fullmatch(r'(.+)\.lora_[AB](?:\.default)?\.weight', key)
        if match is None:
            continue
        name = match.group(1).removeprefix('base_model.model.')
        if name not in module_names:
            raise ValueError(f'Adapter target {name!r} does not exist in the base model; '
                             'check BASE_MODEL and ADAPTER_PATH.')
        targets.add(name)
    if not targets:
        raise ValueError('No LoRA A/B weights found in adapter')
    return sorted(targets)


def precise_adapter_config(model, adapter):
    """Return a corrected PEFT config only for ambiguous numeric target suffixes.

    The checkpoint is read-only. Rank, alpha and other adapter settings are preserved.
    """
    import json

    if getattr(getattr(model, 'config', None), 'model_type', None) != 'qwen2_5_vl':
        return None
    adapter = Path(adapter)
    config_file = adapter / 'adapter_config.json'
    if not config_file.is_file():
        return None  # Let PEFT report missing files or resolve a Hub adapter.
    metadata = json.loads(config_file.read_text())
    targets = metadata.get('target_modules')
    if not isinstance(targets, list) or not any(str(target).isdigit() for target in targets):
        return None
    weights = adapter / 'adapter_model.safetensors'
    if weights.is_file():
        from safetensors import safe_open

        with safe_open(str(weights), framework='pt', device='cpu') as handle:
            keys = list(handle.keys())
    else:
        import torch

        keys = torch.load(adapter / 'adapter_model.bin', map_location='cpu', weights_only=True).keys()
    from peft import PeftConfig

    config = PeftConfig.from_pretrained(str(adapter))
    config.target_modules = set(targets_from_weight_keys(keys, dict(model.named_modules())))
    print(f'Recovered {len(config.target_modules)} precise LoRA targets from adapter weights', flush=True)
    return config
