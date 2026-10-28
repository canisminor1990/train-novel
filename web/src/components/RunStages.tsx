import { Steps } from '@lobehub/ui';

import type { Stage } from '../api';
import { STAGE_STATUS } from '../format';
import { stepTrackClass } from './stepTrack';

function describe(stage: Stage, latestDone: boolean) {
  if (stage.state === 'running' && stage.id === 'train' && stage.step != null && stage.total) {
    return `${stage.step} / ${stage.total} 步`;
  }
  if (stage.state === 'running' || stage.state === 'failed') return stage.detail || undefined;
  if (stage.state === 'stopped') return stage.detail || '已停止';
  if (latestDone) return stage.detail || undefined;
  return undefined;
}

export function RunStages({ stages }: { stages: Stage[] }) {
  const current = stages.findIndex(
    (stage) => stage.state === 'running' || stage.state === 'failed' || stage.state === 'stopped',
  );
  const latestDone = [...stages].reverse().find((stage) => stage.state === 'done')?.id;
  return (
    <Steps
      aria-label="运行阶段"
      className={stepTrackClass}
      current={current >= 0 ? current : stages.filter((stage) => stage.state === 'done').length}
      items={stages.map((stage) => ({
        key: stage.id,
        title: stage.label,
        description: describe(stage, stage.id === latestDone),
        status: STAGE_STATUS[stage.state],
      }))}
    />
  );
}
