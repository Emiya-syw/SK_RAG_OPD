"""Run real coordinators with fake generation/judging; no GPUs or API calls."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]

WORKER = r'''
import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--datasets',nargs='+')
p.add_argument('--output-dir',type=Path)
p.add_argument('--completion-dir',type=Path)
p.add_argument('--shard-index',type=int)
p.add_argument('--output-suffix',default='')
a,_=p.parse_known_args()
events=Path(os.environ['EVENTS'])
i=a.shard_index
(events/f'worker{i}.pid').write_text(str(os.getpid()))
def wait(path):
    until=time.monotonic()+8
    while not path.exists():
        if time.monotonic()>until: raise RuntimeError('timeout '+str(path))
        time.sleep(.02)
for dataset in a.datasets:
    if dataset=='VLGuard':
        if os.environ.get('FAIL_GENERATION')=='true' and i==0:
            wait(events/'MSSBench.start')
            raise RuntimeError('injected generation failure')
        if os.environ.get('STALL')=='true' or os.environ.get('FAIL_JUDGE')=='true' or os.environ.get('FAIL_GENERATION')=='true':
            time.sleep(60)
        if os.environ.get('BARRIER','true')=='true': wait(events/'MSSBench.start')
    out=a.output_dir/dataset
    out.mkdir(parents=True,exist_ok=True)
    (out/f'answers{a.output_suffix}.jsonl').write_text(json.dumps({'id':str(i),'prediction':'answer'})+'\n')
    marker=a.completion_dir/dataset/f'{i}.done'
    marker.parent.mkdir(parents=True,exist_ok=True)
    marker.touch()
    (events/f'{dataset}.{i}.generated').touch()
'''
JUDGE = r'''
import os,time,json
from pathlib import Path
name=os.environ['DATASETS']
events=Path(os.environ['EVENTS'])
assert ' ' not in name
assert os.environ['MAX_IMAGE_PIXELS']=='786432'
answers=Path(os.environ['INPUT_ROOT'])/name/'answers.jsonl'
rows=[json.loads(line) for line in answers.read_text().splitlines()]
assert len(rows)==int(os.environ.get('TEST_WORKERS','2'))
if name=='VLGuard': assert (events/'MSSBench.end').exists()
(events/f'{name}.start').touch()
(events/'judge.pid').write_text(str(os.getpid()))
if os.environ.get('FAIL_JUDGE')=='true': raise RuntimeError('injected judge failure')
if os.environ.get('FAIL_GENERATION')=='true' or os.environ.get('STALL')=='true': time.sleep(60)
if name=='MSSBench':
    until=time.monotonic()+8
    while not all((events/f'VLGuard.{i}.generated').exists() for i in range(int(os.environ.get('TEST_WORKERS','2')))):
        if time.monotonic()>until: raise RuntimeError('next dataset did not generate while judging')
        time.sleep(.02)
(events/f'{name}.end').touch()
'''


class OverlapTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for directory in ('rag_eval', 'scripts', 'events', 'answers'):
            (self.root / directory).mkdir()
        for file in ('run_rag_eval.sh', 'overlap_pipeline.py', 'multi_gpu_generate.py',
                     'process_utils.py', 'merge_answer_shards.py', 'data.py'):
            shutil.copy(ROOT / 'rag_eval' / file, self.root / 'rag_eval' / file)
        (self.root / 'rag_eval/run_rag_eval.py').write_text(WORKER)
        (self.root / 'judge.py').write_text(JUDGE)
        (self.root / 'rag_eval/run_mssbench_protocol_eval.sh').write_text(
            '#!/bin/bash\nexec "$PYTHON_BIN" "' + str(self.root / 'judge.py') + '"\n')
        (self.root / 'scripts/pipeline_artifacts.py').write_text('raise SystemExit(1)\n')
        self.env = dict(os.environ, PYTHONPATH=str(self.root), PYTHON_BIN=sys.executable,
                        OUTPUT_DIR=str(self.root / 'answers'), EVAL_OUTPUT_ROOT=str(self.root / 'scores'),
                        DATASETS='MSSBench VLGuard', GPU_IDS='0,1', EVENTS=str(self.root / 'events'),
                        MULTI_GPU='true', DASHSCOPE_API_KEY='fake', MAX_IMAGE_PIXELS='262144')
        self.command = [sys.executable, '-m', 'rag_eval.overlap_pipeline', '--datasets', 'MSSBench', 'VLGuard']

    def run_pipeline(self, **env):
        return subprocess.run(self.command, cwd=self.root, env={**self.env, **env},
                              capture_output=True, text=True, timeout=20)

    def assert_children_stopped(self):
        for path in (self.root / 'events').glob('*.pid'):
            pid = int(path.read_text())
            stat = Path(f'/proc/{pid}/stat')
            if stat.exists():
                self.assertEqual(stat.read_text().split()[2], 'Z', f'Child {pid} still running')

    def test_generation_overlaps_but_judging_is_serial(self):
        result = self.run_pipeline()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.root / 'events/VLGuard.end').exists())
        self.assert_children_stopped()

    def test_single_gpu_uses_mergeable_shard_name(self):
        result = self.run_pipeline(GPU_IDS='0', TEST_WORKERS='1')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_judge_failure_cancels_generation_workers(self):
        result = self.run_pipeline(FAIL_JUDGE='true')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / 'events/VLGuard.start').exists())
        self.assert_children_stopped()

    def test_generation_failure_cancels_active_judge(self):
        result = self.run_pipeline(FAIL_GENERATION='true')
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.root / 'events/MSSBench.start').exists())
        self.assertFalse((self.root / 'events/VLGuard.start').exists())
        self.assert_children_stopped()

    def test_old_answers_are_not_readiness_signals(self):
        directory = self.root / 'answers/MSSBench'
        directory.mkdir()
        (directory / 'answers.jsonl').write_text('{"id":"old"}\n')
        result = self.run_pipeline()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_termination_cleans_children(self):
        with tempfile.TemporaryFile(mode='w+') as log:
            process = subprocess.Popen(self.command, cwd=self.root, env={**self.env, 'STALL': 'true'},
                                       stdout=log, stderr=log, start_new_session=True)
            try:
                deadline = time.monotonic() + 10
                while not (self.root / 'events/MSSBench.start').exists():
                    if process.poll() is not None or time.monotonic() > deadline:
                        self.fail('Judge did not start')
                    time.sleep(.05)
                process.send_signal(signal.SIGTERM)
                self.assertNotEqual(process.wait(timeout=15), 0)
                self.assert_children_stopped()
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


if __name__ == '__main__':
    unittest.main()
