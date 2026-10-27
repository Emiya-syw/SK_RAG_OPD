"""CPU-only tests of all model/GPU presets and visibility constraints."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('auto_config', Path(__file__).resolve().parents[1] / 'scripts/pipeline_auto_config.py')
auto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto)


class PresetTest(unittest.TestCase):
    def test_4b_scaling_mixed_teacher_and_overrides(self):
        for count, batch, mini in ((4, 16, 8), (8, 32, 16)):
            devices = [{'index': str(i)} for i in range(count)]
            for teacher in (2, 4, 8):
                config = auto.choose({}, 4, teacher, devices)
                self.assertEqual(config['TRAIN_BATCH_SIZE'], str(batch))
                self.assertEqual(config['PPO_MINI_BATCH_SIZE'], str(mini))
                self.assertEqual(config['MAX_TOKEN_LEN_PER_GPU'], '49152')
                self.assertEqual(config['ROLLOUT_GPU_MEMORY_UTILIZATION'], '0.65')
                self.assertEqual(config['ROLLOUT_MAX_NUM_BATCHED_TOKENS'], '16384')
                self.assertEqual(config['ROLLOUT_MAX_NUM_SEQS'], '8')
                self.assertEqual(config['TEACHER_MAX_NUM_SEQS'], str(auto.TEACHER[teacher][1]))
            config = auto.choose({}, 8, 4, devices)
            self.assertEqual(config['TEACHER_GPU_MEMORY_UTILIZATION'], '0.65')
            self.assertEqual(config['TEACHER_MAX_NUM_SEQS'], '8')
            overrides = {'TRAIN_BATCH_SIZE': '4', 'PPO_MINI_BATCH_SIZE': '4',
                         'MAX_TOKEN_LEN_PER_GPU': '24576', 'ROLLOUT_GPU_MEMORY_UTILIZATION': '0.55',
                         'ROLLOUT_MAX_NUM_BATCHED_TOKENS': '8192', 'TEACHER_MAX_NUM_SEQS': '2'}
            config = auto.choose(overrides, 4, 4, devices)
            for key, value in overrides.items():
                self.assertEqual(config[key], value)

    def test_2b_scaling_and_overrides(self):
        for count, batch, mini in ((4, 32, 16), (8, 64, 32)):
            devices = [{'index': str(i)} for i in range(count)]
            config = auto.choose({}, 2, 2, devices)
            self.assertEqual(config['TRAIN_BATCH_SIZE'], str(batch))
            self.assertEqual(config['PPO_MINI_BATCH_SIZE'], str(mini))
            self.assertEqual(config['MAX_TOKEN_LEN_PER_GPU'], '65536')
            self.assertEqual(config['ROLLOUT_MAX_NUM_BATCHED_TOKENS'], '32768')
            self.assertEqual(config['ROLLOUT_MAX_NUM_SEQS'], '16')
            self.assertEqual(config['TEACHER_MAX_NUM_SEQS'], '16')
            self.assertEqual(config['ROLLOUT_GPU_MEMORY_UTILIZATION'], '0.72')
            self.assertEqual(config['TEACHER_GPU_MEMORY_UTILIZATION'], '0.60')
            overrides = {'TRAIN_BATCH_SIZE': '4', 'PPO_MINI_BATCH_SIZE': '4',
                         'MAX_TOKEN_LEN_PER_GPU': '24576',
                         'ROLLOUT_GPU_MEMORY_UTILIZATION': '0.60',
                         'ROLLOUT_MAX_NUM_BATCHED_TOKENS': '8192',
                         'ROLLOUT_MAX_NUM_SEQS': '4', 'TEACHER_MAX_NUM_SEQS': '4'}
            config = auto.choose(overrides, 2, 2, devices)
            for key, value in overrides.items():
                self.assertEqual(config[key], value)
            # Overriding only the batch must still produce a valid mini-batch.
            config = auto.choose({'TRAIN_BATCH_SIZE': '12'}, 2, 2, devices)
            self.assertEqual(config['PPO_MINI_BATCH_SIZE'], '12')

    def test_8b_throughput_preset_and_overrides(self):
        for count, batch, mini in ((4, 8, 4), (8, 16, 8)):
            devices = [{'index': str(i)} for i in range(count)]
            config = auto.choose({'DISTILLATION_LOSS_MODE': 'entropy_jsd'}, 8, 8, devices)
            self.assertEqual(config['TRAIN_BATCH_SIZE'], str(batch))
            self.assertEqual(config['PPO_MINI_BATCH_SIZE'], str(mini))
            self.assertEqual(config['MAX_TOKEN_LEN_PER_GPU'], '21505')
            self.assertEqual(config['ROLLOUT_GPU_MEMORY_UTILIZATION'], '0.55')
            self.assertEqual(config['ROLLOUT_MAX_NUM_BATCHED_TOKENS'], '16384')
            self.assertEqual(config['ROLLOUT_MAX_NUM_SEQS'], str(max(8, batch)))
            self.assertEqual(config['GENERATION_BATCH_SIZE'], '8')
            self.assertEqual(config['TEACHER_GPU_MEMORY_UTILIZATION'], '0.65')
            self.assertEqual(config['TEACHER_MAX_NUM_SEQS'], '4')
            self.assertEqual(config['USE_CHUNKED_TOPK'], 'true')
            overrides = {'TRAIN_BATCH_SIZE': '4', 'PPO_MINI_BATCH_SIZE': '4',
                         'ROLLOUT_GPU_MEMORY_UTILIZATION': '0.40',
                         'TEACHER_GPU_MEMORY_UTILIZATION': '0.50'}
            overridden = auto.choose(overrides, 8, 8, devices)
            for key, value in overrides.items():
                self.assertEqual(overridden[key], value)

    def test_distribution_losses_limit_backward_peak(self):
        devices = [{'index': str(i)} for i in range(4)]
        for mode in ('forward_kl', 'jsd', 'entropy_kl', 'entropy_jsd', 'k3_jsd', 'k3_entropy_jsd'):
            with self.subTest(mode=mode):
                config = auto.choose({'DISTILLATION_LOSS_MODE': mode}, 2, 2, devices)
                self.assertEqual(config['MAX_TOKEN_LEN_PER_GPU'], '24576')
                self.assertEqual(config['USE_CHUNKED_TOPK'], 'true')
                self.assertEqual(config['CHUNKED_TOPK_CHUNK_SIZE'], '512')
                overrides = {'DISTILLATION_LOSS_MODE': mode, 'MAX_TOKEN_LEN_PER_GPU': '32768',
                             'USE_CHUNKED_TOPK': 'false', 'CHUNKED_TOPK_CHUNK_SIZE': '256'}
                config = auto.choose(overrides, 2, 2, devices)
                for key in ('MAX_TOKEN_LEN_PER_GPU', 'USE_CHUNKED_TOPK', 'CHUNKED_TOPK_CHUNK_SIZE'):
                    self.assertEqual(config[key], overrides[key])
        self.assertNotIn('USE_CHUNKED_TOPK', auto.choose({'DISTILLATION_LOSS_MODE': 'k3'}, 2, 2, devices))

    def test_all_18_combinations(self):
        for count in (4, 8):
            for student in (2, 4, 8):
                for teacher in (2, 4, 8):
                    with self.subTest(count=count, student=student, teacher=teacher):
                        config = auto.choose({}, student, teacher, [{'index': str(i)} for i in range(count)])
                        self.assertEqual(int(config['TRAINER_GPUS_PER_NODE']), count // 2)
                        self.assertEqual(int(config['TEACHER_NUM_REPLICAS']), count // 2)
                        self.assertEqual(int(config['TRAIN_BATCH_SIZE']) % (count // 2), 0)
                        self.assertGreaterEqual(int(config['MAX_TOKEN_LEN_PER_GPU']), 21504)
                        self.assertEqual(config['GENERATION_BATCH_SIZE'], str(auto.STUDENT[student][2]))
                        expected_teacher_memory = '0.65' if student == teacher == 8 else auto.TEACHER[teacher][0]
                        self.assertEqual(config['TEACHER_GPU_MEMORY_UTILIZATION'], expected_teacher_memory)

    def test_constraints_and_context(self):
        devices = [{'index': str(i)} for i in range(4)]
        for env in ({'TRAIN_BATCH_SIZE': '3'}, {'TEACHER_TP': '3'},
                    {'TRAINER_GPUS_PER_NODE': '4'}, {'MAX_TOKEN_LEN_PER_GPU': '1024'}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                auto.choose(env, 4, 8, devices)
        result = auto.choose({'MAX_PROMPT_LENGTH': '65536', 'ROLLOUT_GPU_MEMORY_UTILIZATION': '0.42'}, 4, 8, devices)
        self.assertEqual(result['MAX_TOKEN_LEN_PER_GPU'], '66561')
        self.assertEqual(result['ROLLOUT_GPU_MEMORY_UTILIZATION'], '0.42')

    @patch.object(auto.subprocess, 'check_output')
    def test_visibility_and_hardware(self, query):
        query.return_value = '\n'.join(f'{i}, GPU-uuid-{i}, NVIDIA A100-SXM4-80GB, 81920' for i in range(8))
        self.assertEqual(len(auto.visible_gpus({})), 8)
        for mask in ('4,5,6,7', 'GPU-uuid-4,GPU-uuid-5,GPU-uuid-6,GPU-uuid-7'):
            self.assertEqual([d['index'] for d in auto.visible_gpus({'CUDA_VISIBLE_DEVICES': mask})], ['4','5','6','7'])
        for mask in ('', '-1', '0,0,1,2', '0,1'):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                auto.visible_gpus({'CUDA_VISIBLE_DEVICES': mask})
        query.return_value = query.return_value.replace('81920', '40960')
        with self.assertRaises(ValueError):
            auto.visible_gpus({})


if __name__ == '__main__':
    unittest.main()
