# SK-RAG-OPD

基于检索增强视觉语言模型的 On-Policy Distillation（OPD）训练实现。项目从 `SK_RAG` 中整理出可复用的核心算法：student 先根据普通检索上下文采样回答，teacher 再使用 privileged knowledge / demonstration 对同一段采样 token 打分，最后只在 on-policy completion 上进行分布匹配。

## 算法

OPD 有三种 loss：

- `reverse_kl`：标准 full-vocabulary token-level OPD，在 student 的 on-policy rollout 上计算 `KL(student || teacher)`。
- `jsd`：teacher 与 student 的 generalized JSD，支持 `--top_k_loss` 降低大词表显存。
- `sampled_pg`：只计算采样 token 的 log-prob advantage，适合显存有限的训练。

teacher 默认使用冻结的 base model（LoRA 模式下通过 `disable_adapter()`），student 使用可训练 LoRA adapter。prompt token 不参与 loss，避免把检索上下文当作监督答案。

## 安装

```bash
git clone <your-repository-url> SK_RAG_OPD
cd SK_RAG_OPD
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU 训练需要与 CUDA 匹配的 PyTorch；如果使用 Qwen-VL，建议安装官方支持的 `transformers` 版本。

## 数据格式

每行一个 JSON 对象，最少需要 `image_path`、`question` 和 `reference_answer`。可选字段包括 `retrieved_examples`、`safety_knowledge`、`teacher_demonstration`。检索样本可以是：

```json
{"image_path":"images/001.jpg","question":"...","reference_answer":"...","safety_knowledge":"...","retrieved_examples":[{"image_path":"images/002.jpg","question":"...","answer":"...","score":0.91}]}
```

已有 SK-RAG / BeaverTails / VLGuard JSONL 可先执行：

```bash
python scripts/prepare_jsonl.py old.jsonl data/train.jsonl --limit 100
```

## 训练

单卡示例：

```bash
python train_opd.py \
  --train_file data/train.jsonl \
  --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
  --output_dir outputs/opd \
  --use_lora true --bf16 true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing true \
  --loss_type reverse_kl --max_new_tokens 64
```

多卡使用 `accelerate launch`：

```bash
accelerate launch --num_processes 4 train_opd.py --train_file data/train.jsonl --output_dir outputs/opd --use_lora true
```

更完整的参数可运行 `python train_opd.py --help` 查看。不要把模型权重、训练输出和原始数据提交到 GitHub；`.gitignore` 已默认忽略这些目录。

## 验证

不需要 GPU 或模型即可运行核心 loss 测试：

```bash
python -m pytest -q
```

## 项目结构

```text
opd_rag/             OPD trainer 与多模态 collator
train_opd.py         Qwen-VL + LoRA 训练入口
scripts/prepare_jsonl.py  数据格式转换
tests/               CPU 算法 smoke tests
```

## License

建议在公开仓库发布前，根据原始 `SK_RAG` 代码、Qwen 模型和数据集的许可证补充对应声明。
