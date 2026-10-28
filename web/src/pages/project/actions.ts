import { toast } from '@lobehub/ui';

import { api, type ProjectDetail } from '../../api';
import { go, href } from '../../router';

export function startBlockReason(project: ProjectDetail): string | null {
  if (!project.ready) return project.ready_detail;
  if (project.busy) return project.busy;
  if (!project.presets.selected) return '先选一个预设。';
  return null;
}

export function startTraining(
  project: ProjectDetail,
  onDone: () => void,
  setPending: (value: boolean) => void,
) {
  setPending(true);
  api
    .train(project.id)
    .then((result) => {
      toast.success('训练已经开始。训完会自动导出，并挂到 Ollama。');
      onDone();
      go(href.run(project.id, result.run));
    })
    .catch((exc) => toast.error(exc instanceof Error ? exc.message : '没有开训'))
    .finally(() => setPending(false));
}
