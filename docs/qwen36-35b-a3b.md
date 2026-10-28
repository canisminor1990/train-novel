# Qwen3.6-35B-A3B 用 QLoRA 接进训练台

这份说明只定做法，代码还没改。4B / 9B 仍是现在的两条路：bf16 LoRA。35B-A3B 是以后要加的第三项，而且只能走 4bit QLoRA。

本机是一张 RTX 4090（24GB），内存约 128GB，模型缓存在 `E:/train-novel/.cache/huggingface`，下载走 `https://hf-mirror.com`。

## 结论

可以在这张卡上做 QLoRA，不能做全量，也不能做 bf16 LoRA。

Qwen3.6-35B-A3B 是 MoE：总参数约 35B，每个 token 激活约 3B（256 个专家，路由 8 个，再加 1 个共享专家）。Unsloth 给同类 MoE 的 QLoRA 大约 17.5GB 显存。bf16 LoRA 大约要 74GB，27B 的 bf16 LoRA 大约要 56GB，这两条都不要加。

Unsloth 官方不建议对 Qwen3.5 / 3.6 的 MoE 做 4bit QLoRA，原因是 BitsAndBytes 的量化误差比稠密小模型大。这张卡装不下它推荐的 bf16，所以仍然用 QLoRA，接受这个精度损失。不要在实现时改回 16bit。

## 权重

训练时加载的仓库是 `unsloth/Qwen3.6-35B-A3B`。

镜像上这个仓库是 26 个 safetensors 分片的 bf16 权重，Unsloth 的硬件表写 BF16 约 70GB。没有官方的 `unsloth/Qwen3.6-35B-A3B-bnb-4bit`。`load_in_4bit=True` 时由 Unsloth 在加载时做 BitsAndBytes 量化，所以要先把这份 bf16 下到本地。本机内存够做这一步。

不要下这些来训练：

- `unsloth/Qwen3.6-35B-A3B-NVFP4`：只要 Blackwell（RTX 50 系、B200）。4090 跑不了。
- `unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit`：给 Mac 的 MLX。
- 第三方 `*-bnb-4bit`（例如 rwitz、lainlives）：不是这条训练链路要加载的底座。

4bit 文件大约 23GB，那是 GGUF / 量化后的体积，不是训练要提前下载的仓库。训练显存按大约 20GB 做开训检查：空闲显存低于这个数就警告，不拦死。

9B 是另一件事，仓库 `unsloth/Qwen3.5-9B`，bf16，索引里的权重大约 18.0GB。这次只补 9B，35B-A3B 先不下。

## 训练设置

和现有预设对齐，不要为这个模型再开一套学习率表。

| 项 | 取值 | 原因 |
| --- | --- | --- |
| 加载 | `load_in_4bit=true`，`load_in_16bit=false` | QLoRA。4B / 9B 保持相反 |
| batch | 1 | 已经是 `configs/sft_4090.json` 的值 |
| 优化器 | `adamw_8bit` | 训练循环里已经是这个 |
| 梯度检查点 | `unsloth` | 已经打开 |
| 序列 | 6144 | 一段正文大约 4000 到 6500 字。再短会把模型训回写一小段就停。爆显存时先降 rank，不先改短序列 |
| rank | 试跑 8，标准 16，细读 32 | 沿用三个预设。标准档落在 16–32 |
| LoRA 层 | 只挂 `q_proj` `k_proj` `v_proj` `o_proj` | 见下一节 |

4B / 9B 继续挂现在这 7 个模块：上面四个，再加上 `gate_proj` `up_proj` `down_proj`。

### 为什么不挂专家层

这个模型 40 层的排列是 10 组「3 层 Gated DeltaNet + 1 层完整注意力」，每层后面都是 MoE。`gate_proj` / `up_proj` / `down_proj` 会命中全部 256 个专家。

Unsloth 在这条路上有一个已知错误：Qwen3.6-35B-A3B 把专家层也放进 LoRA 时，第一步就会在 `torch._grouped_mm` 上报形状对不上（unsloth-zoo 里修过同类问题）。路由器层 Unsloth 默认也不训。

所以第一版只给完整注意力的 `q_proj` `k_proj` `v_proj` `o_proj` 加 LoRA。DeltaNet 的 `in_proj_qkv`、`out_proj`、`conv` 先不挂。这意味着 40 层里只有大约 10 层完整注意力被适配，文风会弱于「所有层都挂」。那是第一版的取舍，等试跑能跑完再考虑加 DeltaNet，不要和专家层一起加。

## 要改的地方

模型表加在 `novel_train/models.py`。建议 id 用 `35b`，显示名 `Qwen3.6-35B-A3B`，仓库名 `unsloth/Qwen3.6-35B-A3B`。表里写明量化方式和这份模型自己的 `target_modules`，不要让训练脚本猜。

现在有三处把所有模型锁死成 bf16，要改成按模型表判断：

- `novel_train/runs.py` 的 `locked()`：看到 `load_in_4bit` 就拒绝，然后强制写成 16bit。
- `novel_train/train_sft.py`：同样拒绝 4bit，LoRA 模块写死成那 7 个。显存不够时，9B 有专门的提示，35B 要另写一句：先关占显存的程序，再用「试跑」；不要改成 bf16。
- `novel_train/projects.py` 的 `set_base_model()`：报错句还是「只能是 4B 或 9B」。

开训时的显存检查继续读 `vram_gb`。导出和终端续写要跟着这次运行冻结下来的 `load_in_4bit` / `load_in_16bit`，不能再写死 16bit。`novel_train/infer.py` 已经会读 `train_config.json`，保持这样。

`novel_train/export_ollama.py` 今天无论哪次运行都是 `load_in_4bit=False`、`load_in_16bit=True`，再 `save_pretrained_gguf`。35B 的 bf16 约 70GB，24GB 显存装不上。导出必须按 4bit 加载再转 GGUF。合并过程中会在磁盘上出现一份约 70GB 的 16bit 中间结果，转完 GGUF 后按现在的流程删掉。E 盘空间够，这一步是试跑必须验证的部分。

不要改的：

- 命题反推继续用已经缓存的 Qwen3.5-4B。
- `train_cpt.py` 和 `configs/cpt_4090.json` 仍是 4B bf16。不要把 35B 接进 CPT。
- 后台继续只听 `127.0.0.1`。
- 不提交 `.env`、`outputs/`、`projects/`、`.cache/` 和原文。

界面上，「训练」页的模型卡现在把每张卡都写成「bf16 LoRA」。第三张卡要写成 4bit QLoRA，并写上大约 20GB。文案从接口带出来，不要在 `TrainTab.tsx` 里写死。第一次选 9B 或 35B 时，现有的「先试跑」提示可以沿用。

文档要一起改，不然下次会按旧禁令把这项撤掉：

- `AGENTS.md`：Qwen3.5 仍然禁止 4bit。Qwen3.6-27B 仍然不要加。35B-A3B 只允许 QLoRA。
- `.agents/train-novel/SKILL.md`：删掉「不要加 35B-A3B」，改成上面这条。
- `README.md` 的模型表，以及「更大的模型」那一节。
- `.agents/ux/product.md`、`.agents/ux/components.md`：模型选择从两张卡改成三张卡。

环境已经够用，第一版不要为了这个模型升级依赖：Unsloth `2026.10.9`，transformers `5.17.0`，torch `2.14.1+cu132`，bitsandbytes `0.50.2`。加载失败再考虑升级，升级前先确认 4B 试跑还在。

这个模型是带视觉编码器的 `Qwen3_5MoeForConditionalGeneration`。现有 `speak.py` 已经避开「整段字符串被处理器当成图片」。导出时仍要去掉 GGUF 里的 `tokenizer.chat_template`，让 Ollama 用 Modelfile。Qwen3.6 的模板和现在为 Qwen3.5 写的思考开关可能对不上，试跑挂上之后要看它是在写正文，还是在写思考或工具调用。

## 落地顺序

1. 用国内镜像把 `unsloth/Qwen3.6-35B-A3B` 下到 `.cache/huggingface`（约 70GB）。代码没接上之前不要下。
2. 按上面改模型表、开训锁、训练目标层、导出加载方式和界面文案，并改掉旧的禁止说明。
3. 用一本正式书跑「试跑」（20 步，rank 8），确认显存、LoRA、GGUF 和 Ollama 挂载。这一步没过，不要开「标准」。
4. 试跑的正文正常之后，再按「标准」训。
