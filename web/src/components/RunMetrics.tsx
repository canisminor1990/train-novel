import { type MetricItem, MetricStrip } from '@lobehub/ui/dashboard';

import type { RunSummary } from '../api';
import { finishAt, formatCount, formatDuration, formatLoss, isLive, runElapsed } from '../format';

export function RunMetrics({ run }: { run: RunSummary }) {
  const live = isLive(run.state);
  const percent = run.progress != null ? Math.round(run.progress * 100) : null;
  const finish = live ? finishAt(run.eta) : null;
  const items: MetricItem[] = [
    {
      label: '步数',
      value: run.total
        ? `${formatCount(run.step ?? run.last_step)} / ${formatCount(run.total)}`
        : formatCount(run.last_step),
      hint: percent != null ? `${percent}%` : live ? '还没开始计步' : undefined,
    },
    {
      label: '训练 loss',
      value: formatLoss(run.last_loss),
      hint: run.first_loss != null ? `起点 ${formatLoss(run.first_loss)}` : undefined,
    },
    {
      label: '验证 loss 最低',
      value: formatLoss(run.best_eval),
      hint:
        run.best_eval_step != null
          ? `第 ${run.best_eval_step} 步${run.eval_rising ? '，之后在回升' : ''}`
          : '每次存断点时验证',
      valueTone: run.eval_rising ? 'warning' : undefined,
    },
    live
      ? {
          label: '预计还要',
          value: run.stage === 'train' ? formatDuration(run.eta) : '—',
          hint: finish
            ? `约 ${finish} 训完`
            : run.stage && run.stage !== 'train'
              ? '训练后导出不估时'
              : undefined,
        }
      : {
          label: '用时',
          value: formatDuration(runElapsed(run)),
        },
  ];
  return <MetricStrip items={items} />;
}
