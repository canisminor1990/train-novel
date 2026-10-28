import { Alert, Button, Flexbox, Progress, Text, toast } from '@lobehub/ui';
import { Surface } from '@lobehub/ui/dashboard';
import { Pause, Sparkles } from 'lucide-react';
import { useState } from 'react';

import { api, type ProjectDetail } from '../../api';
import { formatCount, formatDuration, formatRemaining } from '../../format';

// 4090 上按两三千字段落估过每段约 15 秒。正文加长到四五千字后会更慢一些。
const SECONDS_PER_PASSAGE = 15;

export function PremisePanel({
  project,
  onChange,
}: {
  project: ProjectDetail;
  onChange: () => void;
}) {
  const [pending, setPending] = useState(false);
  const { premises } = project;
  const missing = Math.max(0, premises.passages - premises.with_premise);
  const running = premises.alive || premises.state === 'queued';
  const percent =
    premises.total && premises.done != null ? Math.round((premises.done / premises.total) * 100) : 0;
  const busyElsewhere = project.busy && !premises.alive ? project.busy : null;

  async function act(action: () => Promise<unknown>, done: string, failure: string) {
    setPending(true);
    try {
      await action();
      toast.success(done);
      onChange();
    } catch (exc) {
      toast.error(exc instanceof Error ? exc.message : failure);
    } finally {
      setPending(false);
    }
  }

  if (!premises.passages) return null;

  return (
    <Surface
      description="让 Qwen3.5-4B 读每段正文，反推一个写作命题：谁、在哪、发生了什么、情绪怎么走、停在哪。训练时模型看着命题写出这段正文，之后才会按你给的命题写长文。写好的命题会留着，加书后只补新的段落。"
      title="命题"
    >
      <Flexbox gap={12}>
        <Text>
          {formatCount(premises.passages)} 段正文，{formatCount(premises.with_premise)} 段有命题
          {missing && !running
            ? `。剩下 ${formatCount(missing)} 段约要 ${formatDuration(missing * SECONDS_PER_PASSAGE)}，期间显卡不能训练。`
            : null}
        </Text>
        {running ? (
          <Progress
            aria-label="写命题"
            format={() => `${percent}%`}
            label={`${premises.detail || '正在启动'} · ${formatRemaining(premises.eta)}`}
            percent={percent}
            size="middle"
          />
        ) : null}
        {premises.state === 'failed' ? (
          <Alert showIcon title={`没写完：${premises.detail}`} type="error" />
        ) : null}
        {premises.state === 'interrupted' ? (
          <Alert showIcon title="写命题的进程已经不在。写好的都留着，可以接着写。" type="warning" />
        ) : null}
        {premises.state === 'stopped' || (premises.state === 'done' && !missing) ? (
          <Text type="secondary">{premises.detail}</Text>
        ) : null}
        <Flexbox horizontal align="center" gap={12} wrap="wrap">
          {running ? (
            <Button
              icon={Pause}
              loading={pending}
              onClick={() =>
                void act(() => api.stopPremises(project.id), '已暂停，写好的命题已用进样本', '没能暂停')
              }
            >
              暂停
            </Button>
          ) : missing ? (
            <Button
              disabled={Boolean(busyElsewhere)}
              icon={Sparkles}
              loading={pending}
              type="primary"
              onClick={() =>
                void act(() => api.startPremises(project.id), '开始写命题', '没有开始')
              }
            >
              {premises.with_premise ? `接着写剩下的 ${formatCount(missing)} 段` : '开始写命题'}
            </Button>
          ) : null}
          {busyElsewhere ? <Text type="secondary">{busyElsewhere}</Text> : null}
        </Flexbox>
      </Flexbox>
    </Surface>
  );
}
