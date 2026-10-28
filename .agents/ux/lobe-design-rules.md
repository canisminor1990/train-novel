# 项目层规则

编号只增不复用。这里只记界面选型，训练约束仍以仓库根的 `AGENTS.md` 为准。

## R-01 原子层不自建

出现自建按钮、输入、标签、进度或弹窗，而 `@lobehub/ui` 已有同职责组件时，改回上游组件。

## R-02 曲线只用图表库

loss 和其他序列用 `@lobehub/charts`。不要用 div 或 canvas 再画一条同样的曲线。

## R-03 配置保持三个预设

训练配置页不增加逐项数值表单。新的训练安排做成第四个预设，或明确替换现有预设的文案。

## R-04 表格用 outlined

页面上的 `Table` 写 `variant="outlined"`。默认的 `borderless` 只留给已经套在带边框容器里的表。

## R-05 外壳撑满视口

`ConsoleShell` 自己是 `block-size: 100%`。`html`、`body`、`#root` 和它的父级都要有确定高度（`100%` 加 `100dvh`），外壳再写 `block-size: 100%`。缺了这一层，外壳会按内容收缩。

## R-06 单块面板用 Surface

工作区里带标题、说明、表格或图的一块面板用 `Surface`。`Section` 只给「标题下面再排一组卡片」，例如三个预设。不要用 `Section variant="card"` 再包一层。
