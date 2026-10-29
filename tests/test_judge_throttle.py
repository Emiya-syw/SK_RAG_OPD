import hashlib
import json
import time
import pytest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rag_eval.judge_throttle import RequestGate, terminal_error
from rag_eval import evaluate_safety_alignment_outputs as judge


def test_ramp_and_rate_limit_reduce_capacity_and_honor_cooldown():
    gate = RequestGate()
    for _ in range(32):
        gate.active = 1
        gate.release(success=True)
    assert gate.capacity == 10
    gate.active = 1
    before = time.monotonic()
    gate.release(limited=True, retry_after=30)
    assert gate.capacity == 5 and gate.active == 0
    assert gate.cooldown >= before + 30


def test_content_rejection_is_not_retried_but_remains_an_error():
    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError('data_inspection_failed')
    task = SimpleNamespace(id='a', model='model', dataset='mssbench', prompt='rubric', image=None)
    result = judge.judge_one(client, task, 'judge', 3, 0, 1000, False)
    assert client.chat.completions.create.call_count == 1
    assert result['status'] == 'error' and result['retryable'] is False
    assert not terminal_error('429 Too Many Requests')


def test_cached_rejection_reused_only_for_matching_prompt(tmp_path):
    task = SimpleNamespace(id='a', model='model', dataset='mssbench', prompt='rubric', image=None)
    p = tmp_path/'judgments.jsonl'
    p.write_text(json.dumps({'id':'a','status':'error','error':'data_inspection_failed',
                            'judge_model':'judge','judge_prompt_sha256':hashlib.sha256(b'rubric').hexdigest()})+'\n')
    args = SimpleNamespace(judge_model='judge',dry_run=False,api_key='test',base_url='https://example.invalid')
    with patch.object(judge, 'make_client') as client:
        result = judge.run_judge_tasks([task],p,args)
        client.assert_not_called()
    assert result[0]['status']=='error'
    task.prompt = 'changed rubric or response'
    with patch.object(judge, 'make_client', side_effect=RuntimeError('new request required')):
        with pytest.raises(RuntimeError, match='new request required'):
            judge.run_judge_tasks([task],p,args)


def test_transient_failure_is_still_retried():
    client = Mock()
    client.chat.completions.create.side_effect = [TimeoutError('timeout'),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='<answer>1</answer>'))])]
    task = SimpleNamespace(id='a',model='model',dataset='mssbench',prompt='rubric',image=None)
    with patch.object(judge.time, 'sleep'):
        result = judge.judge_one(client,task,'judge',3,0,1000,False)
    assert client.chat.completions.create.call_count == 2
    assert result['status']=='valid' and result['parsed']['label']==1
