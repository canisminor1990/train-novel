import { Alert, Button, Flexbox, Steps, Text, toast } from '@lobehub/ui';
import { Section, Surface } from '@lobehub/ui/dashboard';
import { ArrowRight, Play, Upload } from 'lucide-react';
import { useState } from 'react';

import { api, type ProjectDetail, type RunSummary } from '../../api';
import { LossChart } from '../../components/LossChart';
import { ChoiceCards, PresetPicker } from '../../components/PresetPicker';
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
      title={`${runName(run)} 开始的运行 · ${run.model_label} · ${run.preset_label}`}
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
              去写一章
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
            上一次运行 · {formatTime(run.created)} · {run.model_label} · {run.preset_label}
          </span>
          <RunStateTag state={run.state} />
        </Flexbox>
      }
    />
  );
}

function GettingStarted({
  ready,
  premised,
  tried,
}: {
  premised: boolean;
  ready: boolean;
  tried: boolean;
}) {
  const current = !ready ? 0 : !premised ? 1 : !tried ? 2 : 3;
  return (
    <Surface description="走完这几步，就能在「写作」里给一个命题，用自己的文风写出一整章。" title="上手">
      <Steps
        className={stepTrackClass}
        current={current}
        items={[
          {
            description: 'txt 或 epub，分不分章都行。导入后自动清洗、切成整章长短的样本。',
            title: '导入小说',
          },
          { description: '在「书和样本」里给每段正文反推一个命题。', title: '写命题' },
          { description: '先用「试跑」确认整条流程能走通。', title: '试跑一次' },
          { description: '满意后按「标准」训，再到「写作」里试。', title: '正式训练' },
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

  async function save(action: () => Promise<unknown>, failure: string) {
    setSaving(true);
    try {
      await action();
      onChange();
    } catch (exc) {
      toast.error(exc instanceof Error ? exc.message : failure);
    } finally {
      setSaving(false);
    }
  }

  const firstTime = !project.runs.some((run) => run.state === 'done');
  const tried = project.runs.length > 0;
  const model = project.models.items.find((item) => item.selected);
  const modelTried = project.runs.some(
    (run) => run.state === 'done' && run.model_label === model?.label,
  );
  const samples = project.dataset?.train_samples ?? 0;
  const { premises } = project;

  return (
    <Flexbox gap={24}>
      {firstTime && !liveRun ? (
        <GettingStarted
          premised={premises.passages > 0 && premises.with_premise >= premises.passages}
          ready={project.ready}
          tried={tried}
        />
      ) : null}
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
        desc={`${tried ? '' : '第一次建议先选「试跑」，十来分钟走完训练、导出和挂载，确认没问题再按「标准」训。'}bf16 LoRA，换模型或预设只影响下一次运行。${project.dataset ? ` 当前有 ${project.dataset.train_samples} 条训练样本，${project.dataset.eval_samples} 条验证样本。` : ''}`}
        title="下一次怎么训"
      >
        <Flexbox gap={16}>
          <ChoiceCards
            disabled={saving || Boolean(liveRun)}
            label="基座模型"
            choices={project.models.items.map((item) => ({
              ...item,
              facts: `bf16 LoRA · 约 ${item.vram_gb}GB 显存`,
            }))}
            onSelect={(id) => void save(() => api.setModel(project.id, id), '模型没有换成')}
          />
          {model && model.id !== '4b' && !modelTried ? (
            <Alert
              showIcon
              title={`第一次用 ${model.label}，先按「试跑」确认显存够用，再按「标准」训。`}
              type="info"
            />
          ) : null}
          <PresetPicker
            disabled={saving}
            presets={project.presets.items}
            onSelect={(id) => void save(() => api.setPreset(project.id, id), '预设没有换成')}
          />
          {project.ready && premises.passages > 0 && premises.with_premise < premises.passages ? (
            <Alert
              showIcon
              title={`${premises.passages} 段正文里有 ${premises.with_premise} 段有命题。没有命题的段落只学得会接着前情写，学不会按命题写。`}
              type="warning"
              action={
                <Button size="small" onClick={() => go(href.project(project.id, 'data'))}>
                  去写命题
                </Button>
              }
            />
          ) : null}
          {project.ready && samples > 0 && samples < 60 ? (
            <Alert
              showIcon
              title={`只有 ${samples} 条训练样本。能学到文风，但容易背原文。可以加同一作者的书；先别用「细读」。`}
              type="info"
            />
          ) : null}
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
