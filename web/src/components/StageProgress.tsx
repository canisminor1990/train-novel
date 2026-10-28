import { createStaticStyles, Flexbox, Progress, Spin, Text } from '@lobehub/ui';
import { Surface } from '@lobehub/ui/dashboard';

import type { Stage } from '../api';
import { formatBytes, formatRemaining } from '../format';

const EXPORT_STAGES = new Set(['merge', 'gguf', 'ollama']);

const styles = createStaticStyles(({ css, cssVar }) => ({
  log: css`
    overflow: auto;

    max-block-size: 220px;
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

export interface StageOutput {
  gguf?: string | null;
  ggufBytes?: number;
  loraBytes?: number;
  model?: string;
}

function currentStage(stages: Stage[]) {
  return (
    stages.find((stage) => stage.state === 'running') ||
    stages.find((stage) => stage.state === 'failed' || stage.state === 'stopped') ||
    [...stages].reverse().find((stage) => stage.state === 'done') ||
    stages[0]
  );
}

function percentOf(stage: Stage) {
  if (stage.state === 'done') return 100;
  if (typeof stage.progress !== 'number') return 0;
  return Math.round(Math.min(1, Math.max(0, stage.progress)) * 100);
}

function statusOf(stage: Stage) {
  if (stage.state === 'done') return 'success' as const;
  if (stage.state === 'failed' || stage.state === 'stopped') return 'exception' as const;
  if (stage.state === 'running') return 'active' as const;
  return 'normal' as const;
}

function estimate(stage: Stage) {
  if (stage.state === 'done') return '已完成';
  if (stage.state === 'failed') return stage.detail || '失败';
  if (stage.state === 'stopped') return stage.detail || '已停止';
  if (stage.state === 'pending') return '还没开始';
  return formatRemaining(stage.eta);
}

function produced(stage: Stage, output: StageOutput) {
  if (stage.id === 'save' && output.loraBytes) return `LoRA ${formatBytes(output.loraBytes)}`;
  if (stage.id === 'gguf' && output.gguf) return `${output.gguf} · ${formatBytes(output.ggufBytes ?? 0)}`;
  if (stage.id === 'ollama' && output.model) return output.model;
  return '';
}

function readable(line: string) {
  const trimmed = line.trim();
  for (const mark of ['[0]', '[1]', '[2]', 'In total', 'Unsloth:']) {
    const at = trimmed.indexOf(mark);
    if (at > 0) return trimmed.slice(at);
  }
  return trimmed;
}

function isDecoration(line: string) {
  return line.startsWith('==((') || line.startsWith('"-____-"') || line.startsWith('\\') || line.startsWith('O^O');
}

function recentLines(raw: string, marker: string) {
  const text = raw.replaceAll('\r\n', '\n').replaceAll('\r', '\n');
  const at = marker ? text.lastIndexOf(marker) : -1;
  const slice = at >= 0 ? text.slice(at) : text;
  const lines = slice
    .split('\n')
    .map((line) => readable(line))
    .filter((line) => line && !line.startsWith('=====') && !isDecoration(line));
  const kept: string[] = [];
  for (const line of lines) {
    if (kept.at(-1) === line) continue;
    kept.push(line.length > 240 ? `${line.slice(0, 240)}…` : line);
  }
  return kept.slice(-8).join('\n');
}

function stageMarker(stage: Stage) {
  if (stage.id === 'load') return 'Loading weights';
  if (stage.id === 'train') return 'loss';
  if (stage.id === 'save') return 'LoRA';
  if (stage.id === 'merge') return 'Merging';
  if (stage.id === 'gguf') return 'Converting to GGUF';
  return 'ollama';
}

export function StageProgress({
  stages,
  trainLog = '',
  exportLog = '',
  output = {},
}: {
  stages: Stage[];
  trainLog?: string;
  exportLog?: string;
  output?: StageOutput;
}) {
  const stage = currentStage(stages);
  if (!stage) return null;
  const percent = percentOf(stage);
  const note = produced(stage, output);
  const log = recentLines(
    EXPORT_STAGES.has(stage.id) ? exportLog : trainLog,
    stageMarker(stage),
  );
  const steps =
    stage.step != null && stage.total ? `${stage.step} / ${stage.total} 步` : undefined;

  return (
    <Surface description={estimate(stage)} title={stage.label}>
      <Flexbox gap={12}>
        {stage.state === 'running' && percent === 0 ? (
          <Flexbox horizontal align="center" gap={8}>
            <Spin size="small" />
            <Text type="secondary">百分比还没出来。</Text>
          </Flexbox>
        ) : (
          <Progress
            aria-label={stage.label}
            format={() => `${percent}%`}
            label={steps}
            percent={percent}
            showInfo={percent > 0 && stage.state !== 'done'}
            size="middle"
            status={statusOf(stage)}
          />
        )}
        {note ? <Text type="secondary">{note}</Text> : null}
        {log ? (
          <pre className={styles.log}>{log}</pre>
        ) : (
          <Text type="secondary">还没有输出。</Text>
        )}
      </Flexbox>
    </Surface>
  );
}
