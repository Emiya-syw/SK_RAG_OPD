"""Initial HF generation settings for one model replica per 24 GiB RTX 4090."""
import importlib.util
import json
import os
from pathlib import Path
import re

BATCHES = {'qwen3vl-2b': 8, 'qwen3vl-4b': 4, 'qwen3vl-8b': 2, 'qwen25vl-7b': 2, 'generic': 1}


def detect_profile(model_path):
    path = Path(model_path)
    config_path = path / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    kind = config.get('model_type', '')
    text = config.get('text_config', config)
    hidden = text.get('hidden_size')
    if kind == 'qwen3_vl':
        return {2048: 'qwen3vl-2b', 2560: 'qwen3vl-4b', 4096: 'qwen3vl-8b'}.get(hidden, 'generic')
    if kind == 'qwen2_5_vl' and hidden == 3584:
        return 'qwen25vl-7b'
    name = re.sub(r'[^a-z0-9]', '', path.name.lower())
    for size in (2, 4, 8):
        if f'qwen3vl{size}b' in name:
            return f'qwen3vl-{size}b'
    return 'qwen25vl-7b' if 'qwen25vl7b' in name else 'generic'


def settings(profile, top_k, attention):
    if profile not in BATCHES:
        raise ValueError(f'Unknown EVAL_PROFILE {profile}; choose auto or {list(BATCHES)}')
    batch = BATCHES[profile]
    # SDPA and longer retrieval prompts get more headroom. Runtime OOM recovery
    # can halve the batch further without dropping saved answers.
    if attention != 'flash_attention_2':
        batch = max(1, batch // 2)
    if top_k > 3:
        batch = max(1, batch // 2)
    return batch


def main():
    env = os.environ
    profile = env.get('EVAL_PROFILE', 'auto')
    if profile == 'auto':
        profile = detect_profile(env['BASE_MODEL'])
    attention = env.get('EVAL_ATTN_IMPLEMENTATION', env.get('ATTN_IMPLEMENTATION', 'auto'))
    if attention == 'auto':
        attention = 'flash_attention_2' if importlib.util.find_spec('flash_attn') else 'sdpa'
    batch = settings(profile, int(env.get('TOP_K', '3')), attention)
    print(profile, batch, attention)


if __name__ == '__main__':
    main()
