"""Exercise pipeline orchestration without GPUs, training, or paid API calls."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ("scripts", "rag_eval", "models/student", "models/teacher", "data"):
            (self.root / name).mkdir(parents=True)
        source = Path(__file__).resolve().parents[1] / "scripts/train_eval_pipeline.sh"
        shutil.copy(source, self.root / "scripts/train_eval_pipeline.sh")
        shutil.copy(source.with_name("pipeline_auto_config.py"), self.root / "scripts/pipeline_auto_config.py")
        shutil.copy(source.with_name("pipeline_artifacts.py"), self.root / "scripts/pipeline_artifacts.py")
        for name, hidden, layers in (("student", 2560, 36), ("teacher", 4096, 36)):
            (self.root / "models" / name / "config.json").write_text(json.dumps({
                "model_type": "qwen3_vl", "text_config": {"hidden_size": hidden, "num_hidden_layers": layers}
            }))
        (self.root / "bin").mkdir()
        nvidia = self.root / "bin/nvidia-smi"
        nvidia.write_text("#!/bin/bash\nprintf '%s\\n' " + " ".join(
            f"'{i}, GPU-fake-{i}, NVIDIA A100-SXM4-80GB, 81920'" for i in range(8)
        ) + "\n")
        nvidia.chmod(0o755)
        shutil.copy(source.parents[1] / "rag_eval/evaluate_safety_alignment_outputs.py", self.root / "rag_eval/evaluate_safety_alignment_outputs.py")
        (self.root / "data/train.jsonl").write_text('{}\n')
        for name in ("MSSBench", "VLGuard", "MM-SafetyBench"):
            folder = self.root / "rag_eval/data" / name
            folder.mkdir(parents=True)
            (folder / "test_mixed_top10.jsonl").write_text('{"id": "one"}\n')
        self.env = {
            **os.environ, "PYTHON_BIN": shutil.which("python3"), "DASHSCOPE_API_KEY": "test-only",
            "MODEL_PATH": str(self.root / "models/student"),
            "TEACHER_MODEL_PATH": str(self.root / "models/teacher"),
            "EXPERIMENT_NAME": "test", "START_STAGE": "train", "DRY_RUN": "false",
            "LOG": str(self.root / "stages.jsonl"),
            "PATH": str(self.root / "bin") + os.pathsep + os.environ['PATH'],
            "CUDA_VISIBLE_DEVICES": "0,1,2,3", "AUTO_CONFIG": "true",
        }
        for key in ("OUTPUT_DIR", "EVAL_OUTPUT_ROOT", "DATA_ROOT", "ADAPTER_PATH", "TRAIN_SCRIPT"):
            self.env.pop(key, None)
        for script, stage in (("scripts/train_consistent.sh", "train"),
                              ("scripts/latest_adapter.sh", "export"),
                              ("rag_eval/run_rag_eval.sh", "generate"),
                              ("rag_eval/run_mssbench_protocol_eval.sh", "judge")):
            (self.root / script).write_text(
                '#!/usr/bin/env bash\nset -eu\n'
                f'"$PYTHON_BIN" -c \'import json,os; '
                f'open(os.environ["LOG"],"a").write(json.dumps({{"stage":"{stage}","env":dict(os.environ)}})+"\\n")\'\n'
                + ('[[ "${FAIL_TRAIN:-false}" != true ]]\n' if stage == 'train' else '')
                + ('mkdir -p "$1/global_step_2/actor/lora_adapter"\n'
                   'echo \'{}\' > "$1/global_step_2/actor/lora_adapter/adapter_config.json"\n'
                   'echo weights > "$1/global_step_2/actor/lora_adapter/adapter_model.safetensors"\n'
                   'echo "merger diagnostic output"\n'
                   'echo "$1/global_step_2/actor/lora_adapter"\n' if stage == 'export' else '')
                + ('"$PYTHON_BIN" "' + str(self.root / 'fake_outputs.py') + '" ' + stage + '\n'
                   if stage in ('generate', 'judge') else '')
            )
        (self.root / 'fake_outputs.py').write_text('''import os,sys,json,hashlib
from rag_eval.evaluate_safety_alignment_outputs import MSS_CHAT_JUDGE_PROMPT
from pathlib import Path
for dataset in os.environ['DATASETS'].split():
    if sys.argv[1] == 'generate':
        directory = Path(os.environ['OUTPUT_DIR']) / dataset
        directory.mkdir(parents=True, exist_ok=True)
        row = {'id': 'one', 'prediction': 'answer', 'adapter': os.environ['ADAPTER_PATH'],
               'base_model': os.environ['BASE_MODEL'], 'prompt_mode': os.environ['PROMPT_MODE']}
        (directory / 'answers.jsonl').write_text(json.dumps(row) + '\\n')
    else:
        directory = Path(os.environ['EVAL_OUTPUT_ROOT']) / dataset
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'summary.json').write_text(json.dumps({'total': 1, 'judged': 1, 'errors': 0,
            'judge_model': 'qwen3.6-flash', 'dataset': dataset}))
        (directory / 'judgments.jsonl').write_text(json.dumps({'id': 'one', 'status': 'valid',
            'parsed': {'label': 1}, 'judge_model': 'qwen3.6-flash',
            'judge_prompt_sha256': hashlib.sha256(MSS_CHAT_JUDGE_PROMPT.format(answer='answer').encode()).hexdigest()}) + '\\n')
''')

    def run_pipeline(self, **overrides):
        return subprocess.run(
            ["bash", str(self.root / "scripts/train_eval_pipeline.sh"), "data/train.jsonl"],
            env={**self.env, **overrides}, capture_output=True, text=True,
        )

    def records(self):
        return [json.loads(line) for line in (self.root / 'stages.jsonl').read_text().splitlines()]

    def test_full_pipeline_paths_and_stage_specific_settings(self):
        result = self.run_pipeline(MAX_IMAGE_PIXELS="999", LR="5e-7")
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = self.records()
        self.assertEqual([row['stage'] for row in rows], ['train', 'export', 'generate', 'judge'])
        train, _, generation, judge = [row['env'] for row in rows]
        self.assertEqual(train['MAX_IMAGE_PIXELS'], '50176')
        self.assertEqual(train['LR'], '5e-7')
        self.assertEqual(train['PROMPT_MODE'], 'guided')
        self.assertEqual(generation['PROMPT_MODE'], 'guided')
        self.assertEqual(train['TRAINER_GPUS_PER_NODE'], '2')
        self.assertEqual(train['TEACHER_GPU_MEMORY_UTILIZATION'], '0.50')
        self.assertEqual(generation['MAX_IMAGE_PIXELS'], '262144')
        self.assertEqual(judge['MAX_IMAGE_PIXELS'], '786432')
        self.assertEqual(generation['BASE_MODEL'], self.env['MODEL_PATH'])
        self.assertEqual(generation['OUTPUT_DIR'], judge['INPUT_ROOT'])
        self.assertTrue(generation['ADAPTER_PATH'].endswith('global_step_2/actor/lora_adapter'))
        for dataset in ('MSSBench', 'VLGuard', 'MM-SafetyBench'):
            link = Path(generation['OUTPUT_DIR']) / dataset / 'retrieval.jsonl'
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), self.root / 'rag_eval/data' / dataset / 'test_mixed_top10.jsonl')

    def test_existing_artifacts_skip_all_without_key_or_gpu(self):
        first = self.run_pipeline()
        self.assertEqual(first.returncode, 0, first.stderr)
        before = len(self.records())
        second = self.run_pipeline(DASHSCOPE_API_KEY='', CUDA_VISIBLE_DEVICES='')
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(len(self.records()), before)
        for stage in ('1/4', '2/4', '3/4', '4/4'):
            self.assertIn('[SKIP ' + stage + ']', second.stdout)

    def test_incomplete_judge_only_reruns_one_dataset(self):
        self.assertEqual(self.run_pipeline().returncode, 0)
        (self.root / 'rag_eval/evaluation/test/mssbench_protocol/VLGuard/judgments.jsonl').write_text('')
        result = self.run_pipeline(CUDA_VISIBLE_DEVICES='')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.records()[-1]['stage'], 'judge')
        self.assertEqual(self.records()[-1]['env']['DATASETS'], 'VLGuard')
        root_summary = json.loads((self.root / 'rag_eval/evaluation/test/mssbench_protocol/summary.json').read_text())
        self.assertEqual(len(root_summary['results']), 3)

    def test_partial_answers_only_generate_missing_dataset(self):
        self.assertEqual(self.run_pipeline().returncode, 0)
        before = len(self.records())
        (self.root / 'rag_eval/results/test/VLGuard/answers.jsonl').write_text('')
        result = self.run_pipeline()
        self.assertEqual(result.returncode, 0, result.stderr)
        new = self.records()[before:]
        self.assertEqual([r['stage'] for r in new], ['generate'])
        self.assertEqual(new[0]['env']['DATASETS'], 'VLGuard')

    def test_changed_answer_invalidates_scoring(self):
        self.assertEqual(self.run_pipeline().returncode, 0)
        path = self.root / 'rag_eval/results/test/VLGuard/answers.jsonl'
        row = json.loads(path.read_text())
        row['prediction'] = 'changed answer'
        path.write_text(json.dumps(row) + '\n')
        before = len(self.records())
        result = self.run_pipeline(CUDA_VISIBLE_DEVICES='')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([r['stage'] for r in self.records()[before:]], ['judge'])

    def test_adapter_change_does_not_silently_resume_old_answers(self):
        self.assertEqual(self.run_pipeline().returncode, 0)
        other = self.root / 'another_adapter'
        other.mkdir()
        (other / 'adapter_config.json').write_text('{}')
        (other / 'adapter_model.safetensors').write_text('weights')
        before = len(self.records())
        result = self.run_pipeline(ADAPTER_PATH=str(other))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('different adapter', result.stderr)
        self.assertEqual(len(self.records()), before)

    def test_training_failure_stops_downstream(self):
        self.assertNotEqual(self.run_pipeline(FAIL_TRAIN='true').returncode, 0)
        self.assertEqual([r['stage'] for r in self.records()], ['train'])

    def test_eight_cards_and_override(self):
        result = self.run_pipeline(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', TRAIN_BATCH_SIZE='16')
        self.assertEqual(result.returncode, 0, result.stderr)
        train = self.records()[0]['env']
        self.assertEqual(train['TRAINER_GPUS_PER_NODE'], '4')
        self.assertEqual(train['TEACHER_NUM_REPLICAS'], '4')
        self.assertEqual(train['TRAIN_BATCH_SIZE'], '16')
        self.assertEqual(train['PPO_MINI_BATCH_SIZE'], '16')

    def test_detection_failure_stops_before_training(self):
        result = self.run_pipeline(CUDA_VISIBLE_DEVICES='0,1')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / 'stages.jsonl').exists())

    def test_judge_only(self):
        result = self.run_pipeline(START_STAGE='judge')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([r['stage'] for r in self.records()], ['judge'])

    def test_dry_run_does_not_require_key_or_run_stages(self):
        result = self.run_pipeline(DRY_RUN='true', DASHSCOPE_API_KEY='')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / 'stages.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
