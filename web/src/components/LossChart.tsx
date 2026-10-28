import { LineChart } from '@lobehub/charts';
import { useMemo } from 'react';

import type { MetricRow } from '../api';

const TRAIN = '训练 loss';
const EVAL = '验证 loss';

export function LossChart({ metrics, height = 260 }: { metrics: MetricRow[]; height?: number }) {
  const { rows, categories } = useMemo(() => {
    const byStep = new Map<number, Record<string, number>>();
    let hasEval = false;
    for (const row of metrics) {
      const entry = byStep.get(row.step) ?? { step: row.step };
      if (typeof row.loss === 'number') entry[TRAIN] = row.loss;
      if (typeof row.eval_loss === 'number') {
        entry[EVAL] = row.eval_loss;
        hasEval = true;
      }
      byStep.set(row.step, entry);
    }
    return {
      rows: [...byStep.values()].sort((a, b) => a.step - b.step),
      categories: hasEval ? [TRAIN, EVAL] : [TRAIN],
    };
  }, [metrics]);

  return (
    <LineChart
      autoMinValue
      connectNulls
      categories={categories}
      data={rows}
      height={height}
      index="step"
      noDataText={{ title: '还没有 loss', desc: '训练打出第一步之后，曲线会出现在这里。' }}
      showAnimation={false}
      valueFormatter={(value) => (typeof value === 'number' ? value.toFixed(3) : String(value))}
      xAxisLabel="步数"
      yAxisWidth={48}
    />
  );
}
