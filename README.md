# 小说训练台

在一张 RTX 4090（24GB）上，把你自己的长篇自动切成「前情 → 续写」，用 Unsloth 对 `Qwen3.5-4B` 做 bf16 LoRA，学会这本书的文风，再挂到 Ollama 里续写。不需要人工打标。

只放你有权用于训练的文本。代理的约定在 `AGENTS.md` 和 `.agents/train-novel/SKILL.md`。

## 快速开始

需要：Windows、NVIDIA 显卡驱动、Python 3.12。[Ollama](https://ollama.com/download) 用来挂载和续写，没装也能训练。[Node.js](https://nodejs.org) 22+ 只在改了界面、需要重新构建时用到。

**双击根目录的 `start.bat`。**

它会依次：

1. 第一次运行时调用 `setup.ps1` 装好 Python 环境（Unsloth、CUDA 版 PyTorch），之后跳过
2. 界面源码有改动时重新构建页面
3. Ollama 没在运行就把它拉起来
4. 启动训练台，浏览器打开 http://127.0.0.1:8765

训练台已经开着时，再双击只会打开页面。关掉窗口会停掉训练台，但已经开始的训练在后台继续，重新打开后接着显示进度。

装了 pnpm 的话，也可以用 `pnpm start`。

## 第一次训练

训练按项目隔开：一本目标长篇（或同一作者的几本）建一个项目。每个项目有自己的原文、样本、运行记录和 Ollama 模型，互不影响。显卡同一时间只跑一个任务。

新建项目时给它起一个 Ollama 模型名，比如 `yehang`，训完就用 `ollama run yehang` 续写。英文书名会自动填一个，中文书名请自己起英文或拼音名；留空则用 `novel-日期`。模型名只能用小写字母、数字、点、下划线和短横线，不能和别的项目重复。之后可以在「项目设置」里改，已经挂到 Ollama 的模型会跟着改名。

删除项目在项目页右上角的「…」或首页列表每行的「…」里，会删掉原文、样本和所有运行，可以勾选同时从 Ollama 删除这个项目的模型。正在训练的项目不能删。

项目页的「上手」会带你走三步：

1. **新建项目，导入小说**。txt、md、epub 都行，txt 用 UTF-8 或 GB18030。导入后自动清洗，切成续写样本。「书和样本」页能看到每本清掉了什么、样本长什么样。
2. **试跑一次**。新项目默认选「试跑」，30 步，几分钟走完训练、导出 GGUF、挂到 Ollama，确认显存和环境没问题。
3. **正式训练**。试跑成功后，预设自动切到「标准」。训完到「续写」页贴一段前情试写，或在终端运行 `ollama run <模型名>`。

先放一本你最想模仿的长篇。文风学得动之后，再加同一作者的几本，比单本更不容易把情节背死。不同作者、不同题材分开建项目，不要混在一起训。

以「示例」开头的文件只用来确认流程。项目里有别的书时，示例会被自动跳过；只有示例时不能开训。

### 清洗做了什么

- 去掉网址、站点广告、「求月票」「本章完」这类短行、作者的话、页码和分隔线
- 把排版时断开的行接回成段，合并多余空行；对话每句仍各占一段
- 去掉连续重复的行、重复的章节和章节开头重复的标题
- 统一省略号（……）、破折号（——）和夹在中文里的半角标点，去掉中文字之间多余的空格和零宽字符

### 三个预设

| 预设 | 什么时候用 | 设置 |
| --- | --- | --- |
| 试跑 | 新项目第一次 | 最多 30 步，rank 8 |
| 标准 | 先跑通一本长篇 | 2 轮，rank 16，验证 loss 回升会提前停 |
| 细读 | 文风稳定、想更贴原文 | 3 轮，rank 32，验证 loss 回升会提前停 |

模型固定为 `unsloth/Qwen3.5-4B` 的 bf16 LoRA，序列长度 2048。一条前情加续写大约 1700 token；更长的样本会在训练前跳过，不会截掉续写的结尾。标准和细读每隔约 1/8 的总步数验证一次，验证 loss 连续两次不降就停，最后保留最低的那一版。

看曲线时：训练 loss 一直降、验证 loss 开始回升，说明在背原文了。页面会提示，下次少训一些或换回「标准」。

## 常用命令

根目录的 `package.json` 收了常用入口，用 `pnpm <命令>` 运行：

| 命令 | 做什么 |
| --- | --- |
| `pnpm start` | 一键启动，同 `start.bat` |
| `pnpm env:setup` | 安装或修复 Python 环境 |
| `pnpm env:check` | 检查 CUDA、bf16 和 Unsloth（会短暂占用显卡） |
| `pnpm dashboard` | 只启动训练台，不做检查 |
| `pnpm data -- --project <项目>` | 重新清洗、切样本 |
| `pnpm data:check` | 用合成文本自检清洗和切窗 |
| `pnpm write -- --project <项目> --context-file 前情.txt` | 在终端续写 |
| `pnpm web:dev` | 界面开发服务器（接口转发到 8765） |
| `pnpm build` | 构建界面到 `web/dist` |
| `pnpm lint` | tsc、eslint、stylelint、lobe-design 检查和清洗自检 |

训练和导出由训练台的「开始训练」发起。它会新建一个运行目录，冻结当时的样本和配置，再在后台跑：

```powershell
.\.venv\Scripts\python.exe -m novel_train.jobs --project <项目> --run <运行>            # 训练 → 导出 → 挂载
.\.venv\Scripts\python.exe -m novel_train.jobs --project <项目> --run <运行> --resume   # 从断点继续
.\.venv\Scripts\python.exe -m novel_train.jobs --project <项目> --run <运行> --export-only
```

终端续写默认用项目当前挂载的那次运行。不带 `--context-file` 时可以直接输入前情，单独一行打 `///` 回车。一次写一个场景，把角色状态和前情一起贴进去，不要指望一次生成整本书。

## 目录

```
start.bat / start.ps1            一键启动
setup.ps1                        安装 Python 环境
novel_train/                     清洗、训练、导出和训练台后端
web/                             训练台界面（Vite + @lobehub/ui）
configs/                         训练基础配置
projects/<项目>/raw/              原文
projects/<项目>/processed/        清洗和切好的样本
projects/<项目>/runs/r-时间/       一次运行：样本和配置副本、进度、曲线、日志、lora/、断点
```

`projects/`、`outputs/`、`.env` 不进 git。旧版的 `data/raw` 和 `outputs/sft` 会在第一次打开训练台时迁移成项目「第一本书」，Ollama 模型名仍是 `novel-sft`。

模型缓存在 `.cache\huggingface`，下载走 `hf-mirror.com`。`setup.ps1` 装到 CPU 版 PyTorch 时，会改从南京大学或上海交大镜像装 CUDA 版。

## 常见问题

**页面打不开。** 看 `start.bat` 窗口里的报错。8765 被别的程序占了时，先关掉它。训练台只监听本机，局域网里的其他电脑访问不到，这是故意的。

**提示「正在训练」，按钮点不了。** 显卡同一时间只跑一个任务。等它结束，或在那次运行的页面点「停止训练」。停止后断点还在，可以从断点继续。

**训完了但没挂到 Ollama。** 多半是 Ollama 没开。打开 Ollama 后，在那次运行的页面点「挂到 Ollama」，不用重新训练。

**显存不够。** 关掉其他占显卡的程序（游戏、别的模型）。点首页「这台机器」里的「检查环境」，可以看到可用显存。

**续写在背原文。** 验证 loss 回升得早，说明样本太少或训太久。加同一作者的书，或者换回「标准」。

## 改界面

前端在 `web/`，用 Vite、`@lobehub/ui`、`@lobehub/charts` 和 `@lobehub/icons`。代码规范沿用 `@lobehub/lint`。改完跑：

```powershell
pnpm lint
pnpm build
```

`start.bat` 发现界面源码比 `web/dist` 新时也会自动构建。

## 进阶

**领域续训 CPT。** SFT 的文风已经够用时，可以先不做。CPT 用基座模型（不是 Instruct）先浸泡原文，再在上面做 SFT。语料很大时，单卡以天计。

```powershell
.\.venv\Scripts\python.exe -m novel_train.train_cpt --config configs\cpt_4090.json
```

**换模型。** 训练台和 `train_sft` 都锁定 `unsloth/Qwen3.5-4B`。4B 的效果确认之后想试 `Qwen3.5-9B`，要同时放开 `novel_train/runs.py` 的 `locked` 和 `train_sft.py` 里的模型检查。9B 的 bf16 LoRA 大约 22GB，已经贴着这张卡的上限，序列长度保持 2048。Qwen3.6 的 27B 需要大约 56GB，这张卡装不下。
