import { ActionIcon, Alert, Button, DropdownMenu, Empty, Flexbox } from '@lobehub/ui';
import { PageHeader, PageLoading, PageTabs } from '@lobehub/ui/dashboard';
import { Activity, MoreHorizontal, Play, Settings2, Trash2 } from 'lucide-react';
import { useState } from 'react';

import { api, ApiError } from '../api';
import { DeleteProjectModal, ProjectSettingsModal } from '../components/ProjectDialogs';
import { formatCount, isLive } from '../format';
import { go, href, type ProjectTab } from '../router';
import { usePoll } from '../usePoll';
import { startBlockReason, startTraining } from './project/actions';
import { DataTab } from './project/DataTab';
import { RunsTab } from './project/RunsTab';
import { TrainTab } from './project/TrainTab';
import { TryTab } from './project/TryTab';

export function ProjectPage({
  projectId,
  tab,
  onProjectsChange,
}: {
  projectId: string;
  tab: ProjectTab;
  onProjectsChange: () => void;
}) {
  const poll = usePoll(() => api.project(projectId), 2000, projectId);
  const [dialog, setDialog] = useState<'settings' | 'delete' | null>(null);
  const [starting, setStarting] = useState(false);
  const project = poll.data;

  const refresh = () => {
    void poll.refresh();
    onProjectsChange();
  };

  if (poll.error && !project) {
    const missing = poll.error instanceof ApiError && poll.error.status === 404;
    return (
      <Empty
        action={<Button onClick={() => go(href.home())}>回到项目列表</Button>}
        description={missing ? '它可能已经被删掉了。' : poll.error.message}
        title={missing ? '没有这个项目' : '项目读不出来'}
        type="page"
      />
    );
  }
  if (!project) return <PageLoading label="正在读取项目" />;

  const liveRun = project.runs.find((run) => isLive(run.state));
  const blocked = startBlockReason(project);

  const description = [
    project.books
      ? `${project.books} 本 · ${formatCount(project.train_samples)} 条样本`
      : '还没有书',
    `Ollama ${project.ollama.model}，带思考 ${project.ollama.think_model}`,
  ].join(' · ');

  const action = (
    <Flexbox horizontal align="center" gap={8}>
      {tab === 'train' ? null : liveRun ? (
        <Button icon={Activity} type="primary" onClick={() => go(href.run(project.id, liveRun.id))}>
          查看进度
        </Button>
      ) : (
        <Button
          disabled={Boolean(blocked)}
          icon={Play}
          loading={starting}
          title={blocked ?? undefined}
          type="primary"
          onClick={() => startTraining(project, refresh, setStarting)}
        >
          开始训练
        </Button>
      )}
      <DropdownMenu
        items={[
          {
            key: 'settings',
            icon: Settings2,
            label: '项目设置',
            onClick: () => setDialog('settings'),
          },
          { type: 'divider' },
          {
            key: 'delete',
            icon: Trash2,
            label: liveRun ? '删除项目（训练中不能删）' : '删除项目',
            danger: true,
            disabled: Boolean(liveRun),
            onClick: () => setDialog('delete'),
          },
        ]}
      >
        <ActionIcon icon={MoreHorizontal} title="更多操作" />
      </DropdownMenu>
    </Flexbox>
  );

  return (
    <>
      <PageHeader action={action} description={description} title={project.name} />
      {poll.error ? (
        <Alert showIcon title={`刷新失败：${poll.error.message}`} type="warning" />
      ) : null}
      <PageTabs<ProjectTab>
        label="项目内容"
        value={tab}
        tabs={[
          {
            value: 'train',
            label: '训练',
            content: (
              <TrainTab
                project={project}
                starting={starting}
                onChange={refresh}
                onStart={() => startTraining(project, refresh, setStarting)}
              />
            ),
          },
          {
            value: 'data',
            label: '书和样本',
            count: project.raw_files.length || undefined,
            content: <DataTab project={project} onChange={refresh} />,
          },
          {
            value: 'runs',
            label: '运行记录',
            count: project.runs.length || undefined,
            content: <RunsTab project={project} onChange={refresh} />,
          },
          { value: 'try', label: '续写', content: <TryTab project={project} /> },
        ]}
        onChange={(next) => go(href.project(project.id, next))}
      />
      {dialog === 'settings' ? (
        <ProjectSettingsModal project={project} onClose={() => setDialog(null)} onDone={refresh} />
      ) : null}
      {dialog === 'delete' ? (
        <DeleteProjectModal
          project={project}
          onClose={() => setDialog(null)}
          onDeleted={() => {
            onProjectsChange();
            go(href.home());
          }}
        />
      ) : null}
    </>
  );
}
