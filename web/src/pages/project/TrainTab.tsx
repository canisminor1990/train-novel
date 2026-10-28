import { Alert, Button, Flexbox, Steps, Text, toast } from '@lobehub/ui';
import { Section, Surface } from '@lobehub/ui/dashboard';
import { ArrowRight, Play, Upload } from 'lucide-react';
import { useState } from 'react';

import { api, type ProjectDetail, type RunSummary } from '../../api';
import { LossChart } from '../../components/LossChart';
import { PresetPicker } from '../../components/PresetPicker';
import { RunActions } from '../../components/RunActions';
import { RunMetrics } from '../../components/RunMetrics';
import { RunStages } from '../../components/RunStages';
import { RunStateTag } from '../../components/RunStateTag';
import { StageProgress } from '../../components/StageProgress';
import { stepTrackClass } from '../../components/stepTrack';
import { formatLoss, formatTime, isLive, runName } from '../../format';
import { go, href } from '../../router';
import { usePoll } from '../../usePoll';
import { startBlockReason } from './actions';

function LiveRun({
  project,
  run,
  onChange,
}: {
  project: ProjectDetail;
  run: RunSummary;
  onChange: () => void;
}) {
  const detail = usePoll(() => api.run(project.id, run.id), 2000, run.id);
  const full = detail.data?.run;
  return (
    <Section
      title={`${runName(run)} 开始的运行 · ${run.preset_label}`}
      desc={
        full?.stages.find((stage) => stage.state === 'running')?.detail || run.detail || '正在启动'
      }
      extra={
        <Flexbox horizontal gap={8}>
          <RunActions
            busy={project.busy}
            projectId={project.id}
            run={full ?? run}
            onChange={() => {
              onChange();
              void detail.refresh();
            }}
          />
          <Button
            icon={ArrowRight}
            iconPosition="end"
            onClick={() => go(href.run(project.id, run.id))}
          >
            运行详情
          </Button>
        </Flexbox>
      }
    >
      <Flexbox gap={16}>
        <RunMetrics run={full ?? run} />
        {full ? (
          <>
            <Surface title="进度">
              <RunStages stages={full.stages} />
            </Surface>
            <StageProgress
              exportLog={full.export_log}
              stages={full.stages}
              trainLog={full.train_log}
              output={{
                gguf: full.artifacts.gguf,
                ggufBytes: full.artifacts.gguf_bytes,
                loraBytes: full.has_adapter ? full.artifacts.lora_bytes : 0,
                model: `${full.ollama_model}、${full.ollama_think_model}`,
              }}
            />
          </>
        ) : null}
        <Surface title="loss 曲线">
          <LossChart height={220} metrics={full?.metrics ?? []} />
        </Surface>
      </Flexbox>
    </Section>
  );
}

function LastRun({ project, run }: { project: ProjectDetail; run: RunSummary }) {
  const tone =
    run.state === 'failed'
      ? 'error'
      : run.state === 'done'
        ? 'success'
        : run.state === 'trained'
          ? 'info'
          : 'warning';
  const numbers = [
    run.last_loss != null ? `训练 loss ${formatLoss(run.last_loss)}` : null,
    run.best_eval != null ? `验证 loss 最低 ${formatLoss(run.best_eval)}` : null,
  ]
    .filter(Boolean)
    .join(' · ');
  return (
    <Alert
      showIcon
      description={[run.detail?.replace(/。$/, ''), numbers].filter(Boolean).join('。')}
      type={tone}
      variant="outlined"
      action={
        <Flexbox horizontal gap={8}>
          {run.state === 'done' ? (
            <Button size="small" onClick={() => go(href.project(project.id, 'try'))}>
              去续写
            </Button>
          ) : null}
          <Button size="small" onClick={() => go(href.run(project.id, run.id))}>
            查看这次运行
          </Button>
        </Flexbox>
      }
      title={
        <Flexbox horizontal align="center" gap={8}>
          <span>
            上一次运行 · {formatTime(run.created)} · {run.preset_label}
          </span>
          <RunStateTag state={run.state} />
        </Flexbox>
      }
    />
  );
}

function GettingStarted({ ready, tried }: { ready: boolean; tried: boolean }) {
  const current = !ready ? 0 : !tried ? 1 : 2;
  return (
    <Surface description="三步走完，就能在「续写」里拿自己的文风续下一段。" title="上手">
      <Steps
        className={stepTrackClass}
        current={current}
        items={[
          { description: 'txt 或 epub，导入后自动清洗、切样本。', title: '导入小说' },
          { description: '先用「试跑」确认整条流程能走通。', title: '试跑一次' },
          { description: '满意后按「标准」训，再到「续写」里试。', title: '正式训练' },
        ]}
      />
    </Surface>
  );
}

export function TrainTab({
  project,
  starting,
  onStart,
  onChange,
}: {
  project: ProjectDetail;
  starting: boolean;
  onStart: () => void;
  onChange: () => void;
}) {
  const [saving, setSaving] = useState(false);
  const liveRun = project.runs.find((run) => isLive(run.state));
  const lastRun = !liveRun ? project.runs[0] : undefined;
  const blocked = startBlockReason(project);
  const selected = project.presets.items.find((preset) => preset.selected);

  async function select(id: string) {
    setSaving(true);
    try {
      await api.setPreset(project.id, id);
      onChange();
    } catch (exc) {
      toast.error(exc instanceof Error ? exc.message : '预设没有换成');
    } finally {
      setSaving(false);
    }
  }

  const firstTime = !project.runs.some((run) => run.state === 'done');
  const tried = project.runs.length > 0;

  return (
    <Flexbox gap={24}>
      {firstTime && !liveRun ? <GettingStarted ready={project.ready} tried={tried} /> : null}
      {liveRun ? <LiveRun project={project} run={liveRun} onChange={onChange} /> : null}
      {lastRun ? <LastRun project={project} run={lastRun} /> : null}
      {!project.ready ? (
        <Alert
          showIcon
          title={project.ready_detail}
          type="info"
          action={
            <Button icon={Upload} size="small" onClick={() => go(href.project(project.id, 'data'))}>
              导入小说
            </Button>
          }
        />
      ) : null}
      <Section
        desc={`${tried ? '' : '第一次建议先选「试跑」，几分钟走完训练、导出和挂载，确认没问题再按「标准」训。'}模型固定为 Qwen3.5-4B 的 bf16 LoRA。换预设只影响下一次运行。${project.dataset ? ` 当前有 ${project.dataset.train_samples} 条训练样本，${project.dataset.eval_samples} 条验证样本。` : ''}`}
        title="下一次怎么训"
      >
        <Flexbox gap={16}>
          <PresetPicker
            disabled={saving}
            presets={project.presets.items}
            onSelect={(id) => void select(id)}
          />
          <Flexbox horizontal align="center" gap={12} wrap="wrap">
            {liveRun ? null : (
              <Button
                disabled={Boolean(blocked)}
                icon={Play}
                loading={starting}
                type="primary"
                onClick={onStart}
              >
                {selected ? `按「${selected.label}」开始训练` : '开始训练'}
              </Button>
            )}
            <Text type="secondary">
              {liveRun
                ? '这个项目正在跑。等它结束，或先暂停，再开下一次。'
                : !project.ready
                  ? null
                  : (blocked ?? '训完会自动合并、导出 GGUF，并挂到 Ollama。')}
            </Text>
          </Flexbox>
        </Flexbox>
      </Section>
    </Flexbox>
  );
}
