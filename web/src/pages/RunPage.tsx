import {
  Accordion,
  ActionIcon,
  Alert,
  Button,
  confirmModal,
  CopyButton,
  createStaticStyles,
  Descriptions,
  DropdownMenu,
  Empty,
  Flexbox,
  Text,
  toast,
} from '@lobehub/ui';
import { PageHeader, PageLoading, Surface } from '@lobehub/ui/dashboard';
import { MoreHorizontal, RefreshCw, Trash2, UploadCloud } from 'lucide-react';
import { useMemo } from 'react';

import { api, ApiError, type RunDetail } from '../api';
import { LossChart } from '../components/LossChart';
import { RunActions, useRunAction } from '../components/RunActions';
import { RunMetrics } from '../components/RunMetrics';
import { RunStages } from '../components/RunStages';
import { RunStateTag } from '../components/RunStateTag';
import { StageProgress } from '../components/StageProgress';
import { formatBytes, formatCount, formatTime, isLive, runName } from '../format';
import { go, href } from '../router';
import { usePoll } from '../usePoll';

const styles = createStaticStyles(({ css, cssVar }) => ({
  compare: css`
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 16px;

    @media (width <= 800px) {
      grid-template-columns: 1fr;
    }
  `,
  caption: css`
    margin-block-end: 6px;
    font-size: 12px;
    color: ${cssVar.colorTextTertiary};
  `,
  prose: css`
    overflow: auto;

    max-block-size: 320px;
    margin: 0;
    padding: 12px;
    border-radius: ${cssVar.borderRadius};

    font-size: 14px;
    line-height: 1.85;
    color: ${cssVar.colorText};
    white-space: pre-wrap;

    background: ${cssVar.colorFillQuaternary};
  `,
  log: css`
    overflow: auto;

    max-block-size: 360px;
    margin: 0;
    padding: 12px;
    border-radius: ${cssVar.borderRadius};

    font-family: ${cssVar.fontFamilyCode};
    font-size: 12px;
    line-height: 1.6;
    color: ${cssVar.colorTextSecondary};
    white-space: pre-wrap;

    background: ${cssVar.colorFillQuaternary};
  `,
}));

function parsePreview(text: string) {
  const pick = (label: string) => {
    const match = new RegExp(`【${label}】\\n([\\s\\S]*?)(?=\\n\\n【|$)`).exec(text);
    return match ? match[1].trim() : '';
  };
  return { input: pick('输入'), model: pick('模型续写'), original: pick('原文') };
}

const CONFIG_FIELDS: [string, string, (value: unknown) => string][] = [
  ['lora_rank', 'LoRA rank / alpha', (value) => String(value)],
  ['max_seq_length', '序列长度', (value) => String(value)],
  ['learning_rate', '学习率', (value) => String(value)],
  ['num_train_epochs', '轮数', (value) => String(value)],
  ['max_steps', '步数上限', (value) => (value ? String(value) : '不限')],
  ['gradient_accumulation_steps', '梯度累积', (value) => String(value)],
  ['lr_scheduler_type', '学习率日程', (value) => String(value)],
  [
    'early_stopping_patience',
    '提前停止',
    (value) => (value ? `验证 loss 连续 ${value} 次不降` : '关闭'),
  ],
];

function ConfigSection({ run }: { run: RunDetail }) {
  const cfg = run.config;
  if (!cfg) return null;
  const items = CONFIG_FIELDS.filter(([key]) => key in cfg).map(([key, label, render]) => ({
    key,
    label,
    children: key === 'lora_rank' ? `${cfg.lora_rank} / ${cfg.lora_alpha}` : render(cfg[key]),
  }));
  return (
    <Surface
      description={`${String(cfg.model_name)} · bf16 LoRA。开训时冻结，之后改预设不影响这次。`}
      title="这次的设置"
    >
      <Descriptions column={4} items={items} />
    </Surface>
  );
}

function OutputSection({ run }: { run: RunDetail }) {
  const { artifacts } = run;
  return (
    <Surface title="产物">
      <Descriptions
        column={2}
        items={[
          {
            key: 'dir',
            label: '目录',
            span: 2,
            children: (
              <Flexbox horizontal align="center" gap={4}>
                <Text code ellipsis>
                  {artifacts.dir}
                </Text>
                <CopyButton content={artifacts.dir} size="small" title="复制路径" />
              </Flexbox>
            ),
          },
          {
            key: 'lora',
            label: 'LoRA',
            children: run.has_adapter ? formatBytes(artifacts.lora_bytes) : '还没有',
          },
          {
            key: 'gguf',
            label: 'GGUF',
            children: artifacts.gguf
              ? `${artifacts.gguf} · ${formatBytes(artifacts.gguf_bytes)}`
              : '还没有',
          },
          {
            key: 'ollama',
            label: 'Ollama',
            children: run.mounted
              ? `当前挂载为 ${run.ollama_model} 和 ${run.ollama_think_model}`
              : '不是当前挂载的版本',
          },
          { key: 'disk', label: '占用磁盘', children: formatBytes(artifacts.disk_bytes) },
        ]}
      />
    </Surface>
  );
}

export function RunPage({
  projectId,
  runId,
  onChange,
}: {
  projectId: string;
  runId: string;
  onChange: () => void;
}) {
  const poll = usePoll(() => api.run(projectId, runId), 2000, `${projectId}/${runId}`);
  const run = poll.data?.run;
  const preview = useMemo(() => (run?.preview ? parsePreview(run.preview) : null), [run?.preview]);
  const { pending, run: act } = useRunAction(() => {
    void poll.refresh();
    onChange();
  });

  if (poll.error && !run) {
    const missing = poll.error instanceof ApiError && poll.error.status === 404;
    return (
      <Empty
        action={<Button onClick={() => go(href.project(projectId, 'runs'))}>回到运行记录</Button>}
        description={missing ? '它可能已经被删掉了。' : poll.error.message}
        title={missing ? '没有这次运行' : '运行读不出来'}
        type="page"
      />
    );
  }
  if (!run || !poll.data) return <PageLoading label="正在读取运行" />;

  const live = isLive(run.state);
  const refresh = () => {
    void poll.refresh();
    onChange();
  };

  function confirmRemerge() {
    confirmModal({
      title: '重新合并再导出？',
      content: '会丢掉已经合并的 16 位权重，从 LoRA 重新合并再导出。已经完成的训练不会重跑。',
      cancelText: '取消',
      okText: '重新合并',
      okButtonProps: { danger: true },
      onOk: () => act('重新合并', () => api.exportRun(projectId, runId, true), '已从 LoRA 重新开始导出。'),
    });
  }

  function confirmDelete() {
    if (!run) return;
    confirmModal({
      title: '删除这次运行？',
      content: `会删掉这次的 LoRA、断点、GGUF 和日志，共 ${formatBytes(run.artifacts.disk_bytes)}，不能恢复。${run.mounted ? `它是 ${run.ollama_model} 和 ${run.ollama_think_model} 当前挂载的版本，Ollama 里的模型会保留，直到下一次挂载覆盖它们。` : ''}`,
      cancelText: '取消',
      okText: '删除运行',
      okButtonProps: { danger: true },
      onOk: async () => {
        try {
          await api.deleteRun(projectId, run.id);
          toast.success('运行已删除');
          onChange();
          go(href.project(projectId, 'runs'));
        } catch (exc) {
          toast.error(exc instanceof Error ? exc.message : '删除失败');
        }
      },
    });
  }

  const failedStage = run.stages.find((stage) => stage.state === 'failed');
  const description = [
    `${run.preset_label}预设`,
    run.dataset
      ? `${run.dataset.books.length} 本 · ${formatCount(run.dataset.train_samples)} 条训练样本 · ${formatCount(run.dataset.eval_samples)} 条验证`
      : null,
    run.started ? `${formatTime(run.started)} 开始` : null,
  ]
    .filter(Boolean)
    .join(' · ');

  return (
    <>
      <PageHeader
        description={description}
        mark={<RunStateTag state={run.state} />}
        title={`${runName(run)} 的运行`}
        action={
          <Flexbox horizontal align="center" gap={8}>
            <RunActions busy={poll.data.busy} projectId={projectId} run={run} onChange={refresh} />
            <DropdownMenu
              items={[
                ...(run.has_gguf && run.next_action !== 'mount'
                  ? [
                      {
                        key: 'remount',
                        icon: UploadCloud,
                        label: '重新挂载',
                        disabled: live || Boolean(pending),
                        onClick: () =>
                          act(
                            '挂载',
                            () => api.exportRun(projectId, run.id),
                            '开始挂载。已有 GGUF，只会注册到 Ollama。',
                          ),
                      },
                    ]
                  : []),
                ...(run.has_adapter && run.has_merged && !run.has_gguf
                  ? [
                      {
                        key: 'remerge',
                        icon: RefreshCw,
                        label: '重新合并导出',
                        danger: true,
                        disabled: live || Boolean(pending),
                        onClick: confirmRemerge,
                      },
                    ]
                  : []),
                {
                  key: 'delete',
                  icon: Trash2,
                  label: '删除运行',
                  danger: true,
                  disabled: live,
                  onClick: confirmDelete,
                },
              ]}
            >
              <ActionIcon icon={MoreHorizontal} title="更多操作" />
            </DropdownMenu>
          </Flexbox>
        }
      />
      {run.state === 'failed' || run.state === 'interrupted' || run.state === 'trained' ? (
        <Alert
          showIcon
          text={{ detail: '看日志' }}
          type={run.state === 'failed' ? 'error' : 'warning'}
          description={
            failedStage && failedStage.detail !== run.detail ? failedStage.detail : undefined
          }
          extra={
            run.train_log || run.export_log ? (
              <pre className={styles.log}>
                {run.state === 'trained' ? run.export_log : run.train_log}
              </pre>
            ) : undefined
          }
          title={
            run.state === 'interrupted'
              ? `进程已经不在了。${run.has_checkpoint ? '可以从最近的断点继续。' : '没有存下断点，只能重新开训。'}`
              : run.detail
          }
        />
      ) : null}
      {run.eval_rising && !live ? (
        <Alert
          showIcon
          title={`验证 loss 在第 ${run.best_eval_step} 步之后回升。${run.config?.early_stopping_patience ? '保存的是最低的那一版。' : '下次可以少训一些。'}`}
          type="info"
        />
      ) : null}
      <RunMetrics run={run} />
      <Surface description={live ? '每 2 秒刷新一次' : undefined} title="进度">
        <RunStages stages={run.stages} />
      </Surface>
      <StageProgress
        exportLog={run.export_log}
        stages={run.stages}
        trainLog={run.train_log}
        output={{
          gguf: run.artifacts.gguf,
          ggufBytes: run.artifacts.gguf_bytes,
          loraBytes: run.has_adapter ? run.artifacts.lora_bytes : 0,
          model: `${run.ollama_model}、${run.ollama_think_model}`,
        }}
      />
      <Surface
        description="训练 loss 一直降、验证 loss 开始回升，说明在背原文了。"
        title="loss 曲线"
      >
        <LossChart metrics={run.metrics} />
      </Surface>
      {preview && preview.model ? (
        <Surface
          description="训练结束时，拿一段验证集的前情让模型续写，和原文放在一起看。"
          title="续写对照"
        >
          <div className={styles.compare}>
            <div>
              <div className={styles.caption}>模型续写</div>
              <pre className={styles.prose}>{preview.model}</pre>
            </div>
            <div>
              <div className={styles.caption}>原文</div>
              <pre className={styles.prose}>{preview.original}</pre>
            </div>
          </div>
        </Surface>
      ) : null}
      <ConfigSection run={run} />
      <OutputSection run={run} />
      <Surface title="日志">
        <Accordion
          multiple
          gap={8}
          variant="outlined"
          items={[
            {
              key: 'train',
              title: '训练日志',
              children: <pre className={styles.log}>{run.train_log || '还没有输出。'}</pre>,
            },
            ...(run.export_log
              ? [
                  {
                    key: 'export',
                    title: '导出日志',
                    children: <pre className={styles.log}>{run.export_log}</pre>,
                  },
                ]
              : []),
          ]}
        />
      </Surface>
    </>
  );
}
