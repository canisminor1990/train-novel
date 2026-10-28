import { Empty, Flexbox, Table, Tag, Text } from '@lobehub/ui';

import type { ProjectDetail, RunSummary } from '../../api';
import { RunStateTag } from '../../components/RunStateTag';
import { formatCount, formatDuration, formatLoss, isLive, runElapsed, runName } from '../../format';
import { go, href } from '../../router';

export function RunsTab({ project }: { project: ProjectDetail; onChange: () => void }) {
  if (!project.runs.length) {
    return (
      <Empty
        description="每次点「开始训练」都会新开一次运行。它会把当时的样本和设置存一份，互不影响。"
        title="还没有训练过"
      />
    );
  }
  return (
    <Table<RunSummary>
      dataSource={project.runs}
      pagination={project.runs.length > 20 ? { pageSize: 20 } : false}
      rowKey="id"
      variant="outlined"
      columns={[
        {
          key: 'name',
          title: '开始时间',
          render: (_, run) => (
            <Flexbox gap={2}>
              <a href={href.run(project.id, run.id)}>{runName(run)}</a>
              <Text code fontSize={12} type="secondary">
                {run.id}
              </Text>
            </Flexbox>
          ),
        },
        { key: 'preset', title: '预设', width: 80, render: (_, run) => run.preset_label },
        {
          key: 'state',
          title: '状态',
          width: 170,
          render: (_, run) => (
            <Flexbox horizontal align="center" gap={6}>
              <RunStateTag state={run.state} />
              {run.mounted ? <Tag color="success">当前模型</Tag> : null}
              {isLive(run.state) && run.progress != null ? (
                <Text type="secondary">{Math.round(run.progress * 100)}%</Text>
              ) : null}
            </Flexbox>
          ),
        },
        {
          key: 'samples',
          title: '样本',
          width: 100,
          align: 'end',
          render: (_, run) => formatCount(run.dataset?.train_samples),
        },
        {
          key: 'steps',
          title: '步数',
          width: 90,
          align: 'end',
          render: (_, run) => formatCount(run.last_step),
        },
        {
          key: 'loss',
          title: '训练 loss',
          width: 100,
          align: 'end',
          render: (_, run) => formatLoss(run.last_loss),
        },
        {
          key: 'eval',
          title: '验证 loss 最低',
          width: 130,
          align: 'end',
          render: (_, run) => (
            <Text type={run.eval_rising ? 'warning' : undefined}>{formatLoss(run.best_eval)}</Text>
          ),
        },
        {
          key: 'time',
          title: '用时',
          width: 110,
          align: 'end',
          render: (_, run) => formatDuration(runElapsed(run)),
        },
      ]}
      onRow={(run) => ({
        onClick: (event) => {
          if ((event.target as HTMLElement).closest('a')) return;
          go(href.run(project.id, run.id));
        },
        style: { cursor: 'pointer' },
      })}
    />
  );
}
