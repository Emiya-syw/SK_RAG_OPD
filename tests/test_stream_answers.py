import json
import pytest
from scripts.evaluate_safe_models import available_answers


def test_partial_writer_line_is_not_sent_to_judge(tmp_path):
    path = tmp_path/'answers.rank0of2.jsonl'
    path.write_bytes(b'{"id":"a"}\n{"id":')
    assert available_answers(tmp_path, {'a', 'b'}) == {'a': {'id': 'a'}}


def test_wrong_split_and_duplicate_shards_are_rejected(tmp_path):
    for i in range(2):
        (tmp_path/f'answers.rank{i}of2.jsonl').write_text(json.dumps({'id':'a'})+'\n')
    with pytest.raises(ValueError, match='Duplicate'):
        available_answers(tmp_path, {'a'})
    with pytest.raises(ValueError, match='frozen'):
        available_answers(tmp_path, {'b'})
