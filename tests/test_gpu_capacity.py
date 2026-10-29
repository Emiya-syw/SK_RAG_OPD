import json
from pathlib import Path
from unittest.mock import patch

from rag_eval.gpu_capacity import eligible_gpus, choose


def test_only_cards_with_enough_headroom_are_selected():
    assert eligible_gpus('0, 17000\n1, 22000\n2, 24000\n', 20000) == ['1', '2']


def test_partial_shards_keep_their_assignment_without_new_query(tmp_path):
    (tmp_path/'answers.rank0of3.jsonl').write_text('{}\n')
    (tmp_path/'gpu_allocation.json').write_text(json.dumps({'gpu_ids': ['4','5','6']}))
    with patch('rag_eval.gpu_capacity.subprocess.run') as query:
        selected, _ = choose(tmp_path, ['0','1','2'], {'min_free_mib': 20000})
    assert selected == ['4','5','6']
    query.assert_not_called()
