# 代理须知

改训练、切数据或训练后台之前，先读 `.agents/train-novel/SKILL.md`。

这个仓库在一张 RTX 4090 上，把用户自己的长篇切成「命题 → 一整章正文」（分不分章都行），再用 Unsloth 对 `unsloth/Qwen3.5-4B` 或 `unsloth/Qwen3.5-9B`（每个项目自己选）做 bf16 LoRA。目标是给一个命题就能写出长文，不要把样本回答改回两三百字。

- 命令从仓库根目录运行，Python 一律走 `pixi run`（环境在 `.pixi/envs/default`）。常用入口在根目录 `package.json`（`pnpm start` 一键启动，`pnpm lint`、`pnpm build`），逻辑放在 `start.ps1`、`setup.ps1`、`pyproject.toml` 的 pixi 任务和 Python 模块里，`package.json` 只做转发。
- 依赖写在 `pyproject.toml`：运行时依赖放 `[project].dependencies` 并钉死版本，pixi 自己的工具放 `[tool.pixi.dependencies]`。改完运行 `pixi lock`，看过 `pyproject.toml` 和 `pixi.lock` 的 diff 再跑 `pixi lock --check`。不要手改 `pixi.lock`，不要在环境里直接 `pip install`。改 pixi 配置前读 `.agents/skills/use-pixi/SKILL.md`，换 PyTorch 版本前读 `.agents/skills/install-torch/SKILL.md` 和 `.agents/skills/setup-torch/SKILL.md`。
- 和模板不同的两处：平台只锁 `win-64`，这个项目依赖 PowerShell 和 Windows 进程接口；PyTorch 只装 `torch` 和 `torchvision`，官方 CUDA 源上 Windows 版 `torchaudio` 停在 2.11.1，项目也不用它。
- 这台机器上的 Python 不能在用户 AppData 里建文件。直接调 pixi 时先设 `PIXI_CACHE_DIR=E:/train-novel/.cache/pixi`，`setup.ps1` 和 `start.ps1` 已经设好。
- 每次训练任务属于一个项目，数据和产物都在 `projects/<项目>/` 下，不要跨项目共用。小说只放项目的 `raw/`。以「示例」开头的文件在有正式书时会被跳过。
- 训练集里只有示例书时，不要启动训练。
- 前端改完在 `web` 里跑 `pnpm lint` 和 `pnpm build`。
- Qwen3.5 用 bf16 LoRA，不要改回 4bit QLoRA。9B 约 22GB，已是这张卡的上限；Qwen3.6-27B 需要约 56GB，不要加它。
- 先用一本目标长篇跑通，再加同一作者的几本。不要把不同题材混成一个任务。
- 后台是 `python -m novel_train.dashboard`，地址 http://127.0.0.1:8765 。不要把它改成监听公网。
- 不要提交 `.env`、`outputs/`、`projects/`、`data/processed/` 和用户原文。

<!-- lobe-design:begin · managed by scripts/bootstrap.mjs · 不要手动编辑本块 -->
<!-- lobe-design:mode=compact lang=zh -->
## UI 设计工作指南（lobe-design · 精简）

界面相关工作（新页面、交互、视觉、设计系统、还原设计稿、上线前审查）走 **lobe-design**：`https://github.com/lobehub/lobe-design-skill`（skill 版本 5.8.0）。

```bash
export LOBE_DESIGN="${HOME}/.claude/skills/lobe-design"   # 装在本仓库时：./.agents/skills/lobe-design
```

- **入口**：读 `$LOBE_DESIGN/SKILL.md`；先跑 `node "$LOBE_DESIGN/scripts/route.mjs" "<指令原文>"`，一次只进一个 stage，只读它列出的条目。
- **先读项目层**：`.agents/ux/product.md`（平台与适用性）、`.agents/ux/components.md`（用哪个组件）；缺的那份跑 05-init 补上。
- **平台：web**：原子层取自 `@lobehub/ui`，取值走 `cssVar`。
- **优先级**：本项目 `product.md` 里的平台与适用性（标为「不适用」的规则、按平台限定的规则）优先于 skill 的通用规则；其余冲突以 skill 为准。项目层规则要违背通用原则（可达性、P-xx 原型）时，必须在项目层写明理由。
- **改完代码**：跑 `node "$LOBE_DESIGN/scripts/lint-lobe.mjs" <改动路径>`（按 product.md 的平台自动选规则集），FAIL 必须修。
- **纠正要沉淀**：写进 `.agents/ux/lobe-design-rules.md`；判据、选层与回流见 `$LOBE_DESIGN/learnings/README.md`、`$LOBE_DESIGN/CONTRIBUTING.md`。
- **装完 / 升级后**：跑 05-init 两轮；`node "$LOBE_DESIGN/scripts/project-scan.mjs" --check` 查项目层是否缺失或过期。

> 本块由 `$LOBE_DESIGN/scripts/bootstrap.mjs` 维护，重新运行可更新；加 `--full` 输出完整版（含自迭代与回流细则）。
<!-- lobe-design:end -->
