---
name: train-novel
description: >-
  Prepares Chinese novel text, runs Qwen3.5-4B bf16 LoRA with Unsloth on the
  local RTX 4090, and checks the localhost training dashboard. Use when
  working in train-novel, adding novels, running prepare or SFT, changing
  model configs, or opening the training monitor.
---

# 训练小说模型

仓库根目录是 `E:/train-novel`。用 `.venv` 里的 Python。先读 `AGENTS.md`。

## 项目

每本目标长篇（或同一作者的几本）是一个项目，互不共享数据和产物：

```
projects/<项目>/project.json        名称、预设、Ollama 模型名、当前挂载的运行
projects/<项目>/raw/                原文
projects/<项目>/processed/          sft/train.jsonl、sft/eval.jsonl、cpt/books.jsonl、stats.json
projects/<项目>/runs/r-时间/         一次运行：run.json、config.json、data/（开训时的样本副本）、
                                    stage.json、metrics.jsonl、train.log、export.log、lora/、checkpoint-*/
projects/.active.json               显卡锁。同一时间只跑一个训练或导出
```

每个项目一次导出挂两个 Ollama 模型：`novel-<项目>` 的模板预先闭合思考，续写只出正文；`novel-<项目>-think` 保留思考。两个模型的上下文都是 65536（约 5GB 显存），看板前情上限 6 万字。挂载前会删掉 GGUF 里的 `tokenizer.chat_template`：Ollama 发现内嵌的 Qwen3.5 模板声明了工具和思考，就会忽略 Modelfile 的 `TEMPLATE`，而那份模板默认开思考。从旧目录 `data/raw` 和 `outputs/sft` 迁移来的项目叫 `legacy-sft`，模型名保留 `novel-sft`，思考版是 `novel-sft-think`。

## 数据

1. 用户在看板里新建项目并导入有权训练的 `.txt`、`.md` 或 `.epub`。命令行等价于把文件放进 `projects/<项目>/raw` 后运行 `python -m novel_train.prepare --project <项目>`。
2. 先保留一本最想模仿的长篇。效果稳定后，只加同一作者的书。多本时每本自动封顶 2000 条。
3. 清洗在 `textutil.clean_document`：去掉网址、广告、作者的话、页码和分隔线，接回排版断行，去重复行和重复章节，统一省略号、破折号和中文标点。对话每句保持各自一段，不要把短行拼成大段。改清洗后跑 `python -m novel_train.prepare --self-check`。
4. 项目里有正式书时，以「示例」开头的文件会被跳过。只有示例书时不要训练。

## 训练

三个预设在 `novel_train/presets.py`，基础配置是 `configs/sft_4090.json`：`unsloth/Qwen3.5-4B`，`load_in_16bit: true`，`load_in_4bit: false`，序列 2048。

| 预设 | 用途 | 关键设置 |
| --- | --- | --- |
| 试跑 | 新项目默认。几分钟走完整条流程 | 30 步、rank 8、lr 2e-4 |
| 标准 | 试跑成功后自动切到这里 | 2 轮、rank 16、lr 1e-4、早停 |
| 细读 | 文风稳定后想更贴原文 | 3 轮、rank 32、lr 1e-4、早停 |

一条前情 + 续写约 1700 token（Qwen 分词约 1.3 字/token），所以序列不要低于 2048。超长样本会在训练前被丢掉，不截断。

```powershell
python -m novel_train.jobs --project <项目> --run <运行>            # 训练 → 导出 → 挂载
python -m novel_train.jobs --project <项目> --run <运行> --resume   # 从断点继续
python -m novel_train.jobs --project <项目> --run <运行> --export-only
```

运行目录由看板的「开始训练」创建（`runs.create_run`），它会冻结样本和配置。

- 保持 `enable_thinking` 关闭。续写样本只学正文。
- 回答掩码失败时先停，不要改成训练整段 prompt。
- 断点和验证按步数，每次约 1/8 的总步数。标准和细读在验证 loss 连续两次不降时早停，并回到最低的那版。
- 续写用 `python -m novel_train.infer --project <项目> --context-file 前情.txt`，默认取当前挂载的运行。
- CPT 不是第一步。用户明确要求时再用 `configs/cpt_4090.json`。

## 后台

```powershell
python -m novel_train.dashboard
```

打开 http://127.0.0.1:8765 。服务只绑定 `127.0.0.1`。后端在 `novel_train/dashboard/`（`server.py` 路由、`api.py` 业务、`system.py` 显卡和环境检查），任务在 `novel_train/jobs.py`，作为独立进程运行，重启看板不会打断训练。看板是 `web/` 里的 Vite 应用，用 pnpm 安装依赖，构建到 `web/dist`。改完前端在 `web` 里跑 `pnpm lint`（tsc、@lobehub/lint 的 eslint 和 stylelint、lint-lobe）和 `pnpm build`。

进度在运行目录的 `stage.json`，曲线读 `metrics.jsonl`。改训练循环时保留 `attach_jsonl_callback`。Ollama 测试走本机 `11434`，不要在导出或训练时再把 LoRA 载入显存。合并后的 16bit 权重在写出 GGUF 后删除。

## 不要做的事

- 不要把模型改成 Qwen3.6-27B 或 35B-A3B。这张卡装不下推荐的 bf16 LoRA。
- 不要对 Qwen3.5 打开 4bit QLoRA。
- 不要把后台暴露到局域网或公网。
- 不要把用户原文、`.env`、`outputs/`、`projects/` 写进 git。
