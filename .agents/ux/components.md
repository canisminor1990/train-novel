<!-- lobe-design-init: version=5.8.0 fingerprint=dd7b4e71d8d8 -->

# 组件盘点

平台是 web。原子层从 `@lobehub/ui` 取，外壳从 `@lobehub/ui/dashboard` 取，曲线从 `@lobehub/charts` 取，模型图标从 `@lobehub/icons` 取。仓库里还没有独立的 `components` 目录，下面四段都写在训练台页面里。

## 薄包装

没有。不要为了换个 class 再包一层 Button、Input、Tag 或 Progress。

## 与 lobe-ui 重复

没有自建按钮、输入框、表格、弹窗或进度条。发现这类自建件时，改回对应的 `@lobehub/ui` 组件，不要继续加功能。

## 业务组件

| 场景 | 注意 |
| --- | --- |
| 批量导入 | 一次接受多本 txt、md、epub。空目录和「只有示例书」要分开说。示例书不能当成可以开训的数据 |
| 预设选择 | `ChoiceCards`（在 `PresetPicker.tsx`）。三张卡：试跑、标准、细读。选中态只有一个。不在这里展开其余超参 |
| 模型选择 | 同一个 `ChoiceCards`，两张卡：Qwen3.5-4B、Qwen3.5-9B，写明显存。训练中不能换 |
| 写命题 | 「书和样本」里的一块：多少段有命题、进度条、开始 / 接着写 / 暂停。占显卡锁，和训练互斥 |
| 阶段进度 | 七段按顺序排：准备样本、加载模型、训练、保存 LoRA、合并权重、导出 GGUF、挂载到 Ollama。进行中、完成、失败、已停止是不同状态。loss 用 `LineChart`，不要再用一块手写画布 |
| Ollama 写作 | 命题 + 字数 + 可选前情。只在项目的正文模型已经挂上、而且没有任务占着显卡时可用。没有挂上时说明下一步，而不是改去加载 LoRA |
