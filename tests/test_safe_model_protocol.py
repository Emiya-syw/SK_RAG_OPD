from rag_eval.evaluate_mssbench_protocol import expected_safety, mssbench_metrics


def test_new_benchmark_labels_do_not_force_borderline_or_help_to_binary():
    assert expected_safety('SIUO-Gen', {}) == 'unsafe'
    assert expected_safety('VLSBench', {}) == 'unsafe'
    assert expected_safety('SPA-VL', {'subset': 'harm'}) == 'unsafe'
    assert expected_safety('SPA-VL', {'subset': 'help'}) is None
    assert expected_safety('VLSU', {'metadata': {'consensus_combined_grade': 'borderline'}}) is None
    rows = [{'id': str(i), 'subset': s, 'metadata': {'consensus_combined_grade': s}}
            for i, s in enumerate(['safe', 'unsafe', 'borderline'])]
    judgments = [{'id': str(i), 'status': 'valid', 'parsed': {'label': label}}
                 for i, label in enumerate([2, 1, 2])]
    metrics = mssbench_metrics('VLSU', rows, judgments)
    assert metrics['by_task_type']['all']['total_accuracy_pct'] == 100
    assert metrics['by_task_type']['all']['evaluated'] == 2
    assert metrics['unmapped_target_samples'] == 1
    assert metrics['by_subset']['borderline']['label_counts']['2'] == 1
