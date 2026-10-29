import json
from types import SimpleNamespace

import torch

from rag_eval import generate


def test_oom_halves_batch_and_keeps_existing_answers(tmp_path, monkeypatch):
    rows = [{'id': str(i), 'prompt': 'Question', 'images': []} for i in range(3)]
    source = tmp_path/'test.jsonl'
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    target = tmp_path/'answers.jsonl'
    target.write_text(json.dumps({**rows[0], 'prediction': 'cached', 'prompt_mode': 'guided'})+'\n')
    monkeypatch.setattr(generate, 'resolve_model_type', lambda _: 'qwen2_5_vl')
    calls = []

    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return 'prompt'

        def __call__(self, **kwargs):
            size = len(kwargs['text'])
            calls.append(size)
            if size > 1:
                raise torch.OutOfMemoryError('test OOM')
            return {'input_ids': torch.tensor([[1]])}

        def batch_decode(self, *args, **kwargs):
            return ['answer']

    model = SimpleNamespace(parameters=lambda: iter([torch.nn.Parameter(torch.zeros(1))]),
                            generate=lambda **kwargs: torch.tensor([[1, 2, 3]]),
                            generation_config=SimpleNamespace(eos_token_id=3))
    generate.generate_dataset(model, Processor(), source, target, tmp_path, None, 10, 100,
                              True, 1, batch_size=2, direct_generation=True)
    result = [json.loads(line) for line in target.read_text().splitlines()]
    assert [r['id'] for r in result] == ['0', '1', '2']
    assert result[0]['prediction'] == 'cached'
    assert calls == [2, 1, 1]
    assert all(r['generation_batch_size'] == 1 for r in result[1:])
