"""Select Qwen3-VL and Qwen2.5-VL-7B LoRA presets; no model loading or GPU allocation.

These are starting presets, not measured throughput optima or OOM guarantees.
Model signatures match the official Qwen3-VL 2B/4B/8B config.json files.
"""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

SIGNATURES = {(2048, 28): 2, (2560, 36): 4, (4096, 36): 8}
# token budget, rollout memory fraction, generation batch (per GPU)
STUDENT = {2: (65536, '0.72', 16), 4: (49152, '0.65', 8), 7: (21505, '0.55', 4), 8: (21505, '0.55', 8)}
TEACHER = {2: ('0.60', 16), 4: ('0.65', 8), 7: ('0.50', 2), 8: ('0.50', 2)}
DISTRIBUTION_LOSSES = {'forward_kl', 'jsd', 'entropy_kl', 'entropy_jsd', 'k3_jsd', 'k3_entropy_jsd'}


def model_size(path):
    config = json.loads((Path(path) / 'config.json').read_text())
    text = config.get('text_config', config)
    signature = (text.get('hidden_size'), text.get('num_hidden_layers'))
    if config.get('model_type') == 'qwen2_5_vl' and signature == (3584, 28):
        return 7
    if config.get('model_type') != 'qwen3_vl' or signature not in SIGNATURES:
        raise ValueError(f'Unsupported model config at {path}: expected Qwen2.5-VL 7B or dense Qwen3-VL 2B/4B/8B')
    return SIGNATURES[signature]


def visible_gpus(env):
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total', '--format=csv,noheader,nounits'], text=True
    )
    devices = [dict(zip(('index', 'uuid', 'name', 'memory'), [s.strip() for s in row]))
               for row in csv.reader(output.splitlines())]
    if 'CUDA_VISIBLE_DEVICES' in env:
        selected = []
        for token in env['CUDA_VISIBLE_DEVICES'].split(','):
            token = token.strip()
            matches = [d for d in devices if d['index'] == token or
                       (token.startswith('GPU-') and d['uuid'].startswith(token))]
            if len(matches) != 1 or matches[0] in selected:
                raise ValueError(f'Invalid or duplicate CUDA_VISIBLE_DEVICES entry: {token!r}')
            selected.append(matches[0])
        devices = selected
    if len(devices) not in (4, 8):
        raise ValueError(f'Auto presets require 4 or 8 visible GPUs, detected {len(devices)}')
    if any('A100' not in d['name'] or float(d['memory']) < 79000 for d in devices):
        raise ValueError('Auto presets require A100 80GB on all selected GPUs; use AUTO_CONFIG=false for manual settings')
    return devices


def choose(env, student, teacher, devices):
    result = {}

    def setting(key, value):
        result[key] = str(env.get(key, value))
        return result[key]

    setting('CUDA_VISIBLE_DEVICES', ','.join(d['index'] for d in devices))
    setting('GPU_IDS', result['CUDA_VISIBLE_DEVICES'])
    total = len(devices)
    trainer = int(setting('TRAINER_GPUS_PER_NODE', total - int(env.get('TEACHER_GPUS_PER_NODE', total // 2))))
    teacher_gpus = int(setting('TEACHER_GPUS_PER_NODE', total - trainer))
    if trainer <= 0 or teacher_gpus <= 0 or trainer + teacher_gpus != total:
        raise ValueError('Student and teacher GPU counts must be positive and sum to visible GPU count')
    tp = int(setting('TEACHER_TP', 1))
    if tp <= 0 or teacher_gpus % tp:
        raise ValueError('TEACHER_TP must divide TEACHER_GPUS_PER_NODE')
    replicas = int(setting('TEACHER_NUM_REPLICAS', teacher_gpus // tp))
    if replicas <= 0 or replicas * tp != teacher_gpus:
        raise ValueError('TEACHER_NUM_REPLICAS * TEACHER_TP must equal teacher GPU count')
    # 2B throughput trial: sixteen rollout samples and eight update samples per student GPU.
    # 4B throughput trial: eight rollout samples and four update samples per student GPU.
    # 8B throughput trial: four rollout samples and two update samples per student GPU.
    # Scale global batches with GPU count, keeping per-GPU memory budgets fixed.
    batch = int(setting('TRAIN_BATCH_SIZE', trainer * {2: 16, 4: 8, 7: 4, 8: 4}[student]))
    default_mini = trainer * {2: 8, 4: 4, 7: 2, 8: 2}[student]
    # An explicit training batch may not divide the preset mini-batch.
    if default_mini <= 0 or batch % default_mini:
        default_mini = batch
    mini = int(setting('PPO_MINI_BATCH_SIZE', default_mini))
    if mini <= 0 or batch <= 0 or mini % trainer or batch % mini:
        raise ValueError('Batch must be divisible by mini-batch, and mini-batch by student GPU count')
    tokens, rollout_memory, generation_batch = STUDENT[student]
    context = int(env.get('MAX_PROMPT_LENGTH', 20480)) + int(env.get('MAX_RESPONSE_LENGTH', 1024)) + 1
    # Distribution objectives keep gradients through a vocabulary-sized logit
    # tensor. The aggressive k3 token budget can exceed 80 GiB at backward.
    distribution_loss = env.get('DISTILLATION_LOSS_MODE') in DISTRIBUTION_LOSSES
    default_budget = max(min(tokens, 24576), context) if distribution_loss else max(tokens, context)
    budget = int(setting('MAX_TOKEN_LEN_PER_GPU', default_budget))
    if budget < context - 1:
        raise ValueError('MAX_TOKEN_LEN_PER_GPU is smaller than prompt + response length')
    if distribution_loss:
        setting('USE_CHUNKED_TOPK', 'true')
        setting('CHUNKED_TOPK_CHUNK_SIZE', 512)
    teacher_memory, teacher_max_seqs = TEACHER[teacher]
    if student == teacher == 8:
        # The larger 8B rollout batch can keep more independent teacher replicas busy.
        teacher_memory, teacher_max_seqs = '0.65', 4
    for key, value in {
        'MICRO_BATCH_SIZE': 1, 'USE_DYNAMIC_BSZ': 'true',
        'PARAM_OFFLOAD': 'false', 'OPTIMIZER_OFFLOAD': 'false',
        'ENABLE_GRADIENT_CHECKPOINTING': 'true',
        'ROLLOUT_GPU_MEMORY_UTILIZATION': rollout_memory,
        'ROLLOUT_MAX_NUM_BATCHED_TOKENS': {2: 32768, 4: 16384, 7: 16384, 8: 16384}[student],
        'ROLLOUT_MAX_NUM_SEQS': (
            max({2: 16, 4: 8}[student], (batch + trainer - 1) // trainer)
            if student in (2, 4) else max(8, batch)
        ),
        'TEACHER_GPU_MEMORY_UTILIZATION': teacher_memory,
        'TEACHER_MAX_NUM_SEQS': teacher_max_seqs,
        'GENERATION_BATCH_SIZE': generation_batch,
    }.items():
        setting(key, value)
    return result


def main():
    env = os.environ
    student = model_size(env['MODEL_PATH'])
    teacher = model_size(env['TEACHER_MODEL_PATH'])
    devices = visible_gpus(env)
    result = choose(env, student, teacher, devices)
    print(f'Auto preset: student={student}B teacher={teacher}B, {len(devices)} x A100 80GB; '
          'starting presets, not a benchmarked optimum or OOM guarantee.', file=sys.stderr)
    for key, value in result.items():
        if '\n' in value or '\t' in value:
            raise ValueError(f'Invalid whitespace in {key}')
        print(f'{key}\t{value}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        sys.exit(f'Auto configuration failed: {exc}')
