---
name: train-novel
description: >-
  Prepares Chinese novel text into premise → chapter samples, runs Qwen3.5-4B
  or 9B bf16 LoRA with Unsloth on the local RTX 4090, and checks the localhost
  training dashboard. Use when working in train-novel, adding novels, running
  prepare, premise generation or SFT, changing model configs, or opening the
  training monitor.
---

# 训练小说模型

仓库根目录是 `E:/train-novel`。Python 走 `pixi run`，依赖和版本在 `pyproject.toml` 与 `pixi.lock`。先读 `AGENTS.md`。

## 项目

每本目标长篇（或同一作者的几本）是一个项目，互不共享数据和产物：

```
projects/<项目>/project.json        名称、预设、基座（base_model：4b / 9b）、Ollama 模型名、当前挂载的运行
projects/<项目>/raw/                原文
projects/<项目>/processed/          sft/train.jsonl、sft/eval.jsonl、passages.jsonl、cpt/books.jsonl、stats.json
projects/<项目>/premises.jsonl      反推的命题，按段落正文的哈希存。重新切样本不会丢
projects/<项目>/premise_status.json 写命题的进度
projects/<项目>/runs/r-时间/         一次运行：run.json、config.json、data/（开训时的样本副本）、
                                    stage.json、metrics.jsonl、train.log、export.log、lora/、checkpoint-*/
projects/.active.json               显卡锁。同一时间只跑一个训练或导出
```

每个项目一次导出挂两个 Ollama 模型：`novel-<项目>` 的模板预先闭合思考，只出正文；`novel-<项目>-think` 保留思考。两个模型的上下文都是 65536（约 5GB 显存），看板前情上限 3 万字，正文最多 8000 字。Modelfile 带训练时的 system 提示和 min_p 采样。挂载前会删掉 GGUF 里的 `tokenizer.chat_template`：Ollama 发现内嵌的 Qwen3.5 模板声明了工具和思考，就会忽略 Modelfile 的 `TEMPLATE`，而那份模板默认开思考。Modelfile 自己的模板在有 `.Tools` 时写出 Qwen3.5 的 XML 工具说明，并设置 `PARSER qwen3-coder`，所以 harness 可以带工具；没有工具时正文模型仍预先闭合思考。从旧目录 `data/raw` 和 `outputs/sft` 迁移来的项目叫 `legacy-sft`，模型名保留 `novel-sft`，思考版是 `novel-sft-think`。

## 数据

目标是「给一个命题，写出一整章」，不是接着前情写两三百字。旧版样本的回答只有 200 到 800 字，模型因此学会了写一小段就收尾（LongWriter 的结论：SFT 回答多长，模型就只会写多长）。不要把回答再改短。

1. 用户在看板里新建项目并导入有权训练的 `.txt`、`.md` 或 `.epub`。命令行等价于把文件放进 `projects/<项目>/raw` 后运行 `python -m novel_train.prepare --project <项目>`。
2. 先保留一本最想模仿的长篇。效果稳定后，只加同一作者的书。多本时每本自动封顶 2000 条。
3. 清洗在 `textutil.clean_document`：去掉网址、广告、作者的话和页码，接回排版断行，去重复行和重复章节，统一省略号、破折号和中文标点。对话每句保持各自一段，不要把短行拼成大段。分隔线（`***`、`———`、`◇◇◇`）不进正文，但位置记在 `report["scene_breaks"]`，切段要用。改清洗或切段后跑 `python -m novel_train.prepare --self-check`。
4. 切段在 `textutil.iter_passages`。很多书不分章、一本几万字，整本就是一个「正文」，所以不能靠章节：长的一段切成接近等长的几份，每个切点在目标长度 ±30% 里挑，分隔线优先，其次「第二天」「与此同时」这类时间跳转，再次句末，不切在两句对话之间。短章合并到 800 字以上。
5. 第一遍按约 6000 字切，短于 4000 字的章节先合并。第一遍少于 300 段时，再按约 4300 字换一组切点切一遍。两遍都在 4000 字以上：模型一次能写多长，取决于训练回答有多长，再切回一两千字会把「写一小段就停」训回去。字数仍有差别，所以它听得懂「字数」这个要求。每本结尾留出约 5% 做验证，两遍都不碰这一段。超过约 6400 字的段落放不进序列，会继续拆开。
6. 每条样本的 user 是「命题、章节、字数、前情」里有的几项（`prompts.build_user_input`），assistant 是整段正文。一半样本带前情、一半只给命题；没有命题又不是章首的段落一定带前情。
7. 命题靠反推：`python -m novel_train.premise --project <项目>`（看板「书和样本」里的「开始写命题」）用缓存的 Qwen3.5-4B 基座读每段、写 60 到 150 字的命题，每段约 15 秒，占显卡锁。写完自动重新切样本。没有命题的样本只学得会接着前情写。
8. 项目里有正式书时，以「示例」开头的文件会被跳过。只有示例书时不要训练。

## 训练

基座在 `novel_train/models.py`，每个项目在看板里选：`unsloth/Qwen3.5-4B`（约 10GB）或 `unsloth/Qwen3.5-9B`（约 22GB，贴着 24GB 上限）。都是 `load_in_16bit: true`、`load_in_4bit: false`。开训前会让 Ollama 释放显存并打印可用显存；爆显存时报成能看懂的话。第一次用 9B 先跑「试跑」。

三个预设在 `novel_train/presets.py`，基础配置是 `configs/sft_4090.json`，序列都是 6144。

| 预设 | 用途 | 关键设置 |
| --- | --- | --- |
| 试跑 | 新项目和换模型后先跑。十来分钟走完整条流程 | 20 步、rank 8、lr 2e-4 |
| 标准 | 试跑成功后自动切到这里 | 2 轮、rank 16、lr 1e-4、早停 |
| 细读 | 文风稳定后想更贴原文 | 3 轮、rank 32、lr 1e-4、早停 |

一条样本是命题 + 前情 + 约 4000 到 6500 字正文。Qwen 分词大约 1.35 字/token，4000 字约 3000 token，加上命题和前情后序列不要低于 6144。超长样本会在训练前被丢掉，不截断。梯度累积是 4。

```powershell
python -m novel_train.jobs --project <项目> --run <运行>            # 训练 → 导出 → 挂载
python -m novel_train.jobs --project <项目> --run <运行> --resume   # 从断点继续
python -m novel_train.jobs --project <项目> --run <运行> --export-only
```

运行目录由看板的「开始训练」创建（`runs.create_run`），它会冻结样本和配置。

- 保持 `enable_thinking` 关闭。样本只学正文。
- 样本按 TRL 的 prompt / completion 对话格式交给 `SFTTrainer`，由 `completion_only_loss` 掩掉命题和前情。不要再手动渲染模板、用字符串找回答位置。
- 回答掩码失败时先停，不要改成训练整段 prompt。
- LoRA 不写 `target_modules`（配置里 `lora_target_modules: null`），由 Unsloth 选语言部分的全部线性层。Qwen3.5 每 4 层里有 3 层是线性注意力（`in_proj_qkv`、`in_proj_z`、`out_proj`），只写 `q_proj` 这类名字会漏掉它们。
- 配置里模型、LoRA、数据之外的字段原样传给 `SFTConfig`，写错名字会直接报错。导入 Unsloth 后 `SFTConfig` 会静默吞掉不认识的参数，所以不要绕过 `sft_overrides` 的检查。transformers 5 没有 `warmup_ratio`，用小于 1 的 `warmup_steps` 表示比例。
- 断点和验证按步数，每次约 1/8 的总步数。标准和细读在验证 loss 连续两次不降时早停，并回到最低的那版。
- 终端写作用 `python -m novel_train.infer --project <项目> --premise "命题" --length 5000`，可加 `--context-file 前情.txt`，默认取当前挂载的运行。一次要 4000 到 6500 字最稳。
- CPT 不是第一步。用户明确要求时再用 `configs/cpt_4090.json`。

## 后台

```powershell
python -m novel_train.dashboard
```

打开 http://127.0.0.1:8765 。服务只绑定 `127.0.0.1`。后端在 `novel_train/dashboard/`（`server.py` 路由、`api.py` 业务、`system.py` 显卡和环境检查），任务在 `novel_train/jobs.py`，作为独立进程运行，重启看板不会打断训练。看板是 `web/` 里的 Vite 应用，用 pnpm 安装依赖，构建到 `web/dist`。改完前端在 `web` 里跑 `pnpm lint`（tsc、@lobehub/lint 的 eslint 和 stylelint、lint-lobe）和 `pnpm build`。

进度在运行目录的 `stage.json`，曲线读 `metrics.jsonl`。改训练循环时保留 `attach_jsonl_callback`。Ollama 测试走本机 `11434`，不要在导出或训练时再把 LoRA 载入显存。合并后的 16bit 权重在写出 GGUF 后删除。

## 不要做的事

- 不要加 Qwen3.6-27B 或 35B-A3B。这张卡装不下推荐的 bf16 LoRA。9B 已经是上限。
- 不要对 Qwen3.5 打开 4bit QLoRA。
- 不要把后台暴露到局域网或公网。
- 不要把用户原文、`.env`、`outputs/`、`projects/` 写进 git。
