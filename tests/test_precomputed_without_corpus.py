from types import SimpleNamespace

import pytest

from rag_eval.run_rag_eval import prepare_datasets


def test_precomputed_generation_does_not_load_original_corpus(tmp_path, monkeypatch):
    test = tmp_path / 'data/Example/test.jsonl'
    retrieval = tmp_path / 'precomputed/Example/retrieval.jsonl'
    for path in (test, retrieval):
        path.parent.mkdir(parents=True)
        path.write_text('{}\n')
    def forbidden(*args):
        raise AssertionError('Original corpus validation must not run')
    monkeypatch.setattr('rag_eval.run_rag_eval.validate_pair', forbidden)
    args = SimpleNamespace(skip_corpus_validation=True, stage='generate',
                           retrieval_root=tmp_path / 'precomputed', direct_generation=False,
                           data_root=tmp_path / 'data', datasets=['Example'])
    datasets, reports = prepare_datasets(args)
    assert reports == []
    assert datasets[0].retrieval == retrieval
    args.stage = 'retrieve'
    with pytest.raises(ValueError, match='requires generation'):
        prepare_datasets(args)
