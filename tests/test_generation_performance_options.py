"""Verify inference options without loading a model or allocating GPUs."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from rag_eval.generate import load_generator

ROOT = Path(__file__).resolve().parents[1]


class InferenceOptionsTest(unittest.TestCase):
    def test_shell_attention_precedence_and_merge(self):
        with tempfile.TemporaryDirectory() as folder:
            python = Path(folder) / 'capture'
            python.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            python.chmod(0o755)
            env = dict(os.environ, PYTHON_BIN=str(python), MULTI_GPU='false', MERGE_LORA_FOR_INFERENCE='false')
            env.pop('ATTN_IMPLEMENTATION', None)
            env.pop('EVAL_ATTN_IMPLEMENTATION', None)
            for options, expected in (({}, 'sdpa'), ({'ATTN_IMPLEMENTATION': 'flash_attention_2'}, 'flash_attention_2'),
                                      ({'ATTN_IMPLEMENTATION': 'flash_attention_2', 'EVAL_ATTN_IMPLEMENTATION': 'eager',
                                        'MERGE_LORA_FOR_INFERENCE': 'true'}, 'eager')):
                result = subprocess.run(['bash', str(ROOT / 'rag_eval/run_rag_eval.sh')], env={**env, **options},
                                        text=True, capture_output=True, check=True)
                args = json.loads(result.stdout)
                self.assertEqual(args[args.index('--attn-implementation') + 1], expected)
                self.assertEqual('--merge-lora' in args, options.get('MERGE_LORA_FOR_INFERENCE') == 'true')

    def test_loader_merges_only_when_requested(self):
        for adapter, merge in ((None, False), (None, True), (Path('/adapter'), False), (Path('/adapter'), True)):
            with self.subTest(adapter=adapter, merge=merge):
                base = Mock()
                wrapped = Mock()
                merged = Mock()
                wrapped.merge_and_unload.return_value = merged
                processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token=None, eos_token='eos'))
                auto_model = SimpleNamespace(from_pretrained=Mock(return_value=base))
                auto_processor = SimpleNamespace(from_pretrained=Mock(return_value=processor))
                peft = SimpleNamespace(from_pretrained=Mock(return_value=wrapped))
                modules = {'transformers': SimpleNamespace(AutoModelForImageTextToText=auto_model, AutoProcessor=auto_processor),
                           'peft': SimpleNamespace(PeftModel=peft)}
                with patch.dict('sys.modules', modules), patch('rag_eval.generate._dtype', return_value='bf16'):
                    result, _ = load_generator(Path('/base'), adapter, None, 'bfloat16', 'flash_attention_2', merge_lora=merge)
                self.assertEqual(auto_model.from_pretrained.call_args.kwargs['attn_implementation'], 'flash_attention_2')
                self.assertEqual(processor.tokenizer.padding_side, 'left')
                expected = base if adapter is None else (merged if merge else wrapped)
                self.assertIs(result, expected.eval.return_value)
                if adapter is not None and merge:
                    wrapped.merge_and_unload.assert_called_once_with(safe_merge=True)
                else:
                    wrapped.merge_and_unload.assert_not_called()


if __name__ == '__main__':
    unittest.main()
