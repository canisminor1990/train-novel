import { Button, confirmModal, toast } from '@lobehub/ui';
import { Pause, Play, RotateCcw, UploadCloud } from 'lucide-react';
import { useState } from 'react';

import { api, type RunSummary } from '../api';
import { isLive } from '../format';

export function useRunAction(onDone: () => void) {
  const [pending, setPending] = useState('');
  async function run(label: string, action: () => Promise<unknown>, success: string) {
    setPending(label);
    try {
      await action();
      toast.success(success);
      onDone();
    } catch (exc) {
      toast.error(exc instanceof Error ? exc.message : `${label}没有完成`);
    } finally {
      setPending('');
    }
  }
  return { pending, run };
}

function continueHint(run: RunSummary) {
  if (!run.has_adapter) return '从最近的断点接着训练。';
  if (run.has_merged) return '从已合并的权重接着导出 GGUF，不会重新训练，也不会重新合并。';
  return '从保存的 LoRA 接着导出，不会重新训练。';
}

export function confirmPause(projectId: string, run: RunSummary, onPaused: () => void) {
  const exporting = run.phase === 'export' || ['merge', 'gguf', 'ollama'].includes(run.stage || '');
  confirmModal({
    title: exporting ? '暂停导出？' : '暂停这次训练？',
    content: exporting
      ? '会停掉正在跑的导出。已经保存的 LoRA 还在，合并完的权重也会留着。之后点「继续」从这里接着做，不会重新训练。'
      : run.has_checkpoint
        ? '会停掉当前的训练进程。已经存下的断点还在，之后点「继续」从断点接着训。'
        : '会停掉当前的训练进程。这次还没有存下断点，暂停后只能重试。',
    cancelText: '取消',
    okText: '暂停',
    okButtonProps: { danger: true },
    onOk: async () => {
      try {
        await api.stop(projectId, run.id);
        toast.success('已暂停');
        onPaused();
      } catch (exc) {
        toast.error(exc instanceof Error ? exc.message : '暂停失败');
      }
    },
  });
}

function confirmRetry(projectId: string, run: RunSummary, onDone: () => void) {
  confirmModal({
    title: '重试这次训练？',
    content: '这次还没有可用的断点，会用同一份样本和设置从头训练。',
    cancelText: '取消',
    okText: '重试',
    onOk: async () => {
      try {
        await api.restart(projectId, run.id);
        toast.success('已重新开始训练。');
        onDone();
      } catch (exc) {
        toast.error(exc instanceof Error ? exc.message : '重试没有完成');
      }
    },
  });
}

export function RunActions({
  projectId,
  run,
  busy,
  onChange,
}: {
  projectId: string;
  run: RunSummary;
  busy: string | null;
  onChange: () => void;
}) {
  const { pending, run: act } = useRunAction(onChange);
  const live = isLive(run.state);
  const blocked = Boolean(busy) && !live;
  const disabled = blocked || Boolean(pending);
  const blockedTitle = blocked ? (busy ?? undefined) : undefined;

  if (run.next_action === 'pause') {
    return (
      <Button danger icon={Pause} onClick={() => confirmPause(projectId, run, onChange)}>
        暂停
      </Button>
    );
  }
  if (run.next_action === 'continue') {
    const hint = continueHint(run);
    const training = !run.has_adapter;
    return (
      <Button
        disabled={disabled}
        icon={Play}
        loading={pending === '继续'}
        title={blockedTitle ?? hint}
        type="primary"
        onClick={() =>
          act(
            '继续',
            () => (training ? api.resume(projectId, run.id) : api.exportRun(projectId, run.id)),
            training ? '已从断点继续' : hint,
          )
        }
      >
        继续
      </Button>
    );
  }
  if (run.next_action === 'retry') {
    return (
      <Button
        disabled={disabled}
        icon={RotateCcw}
        title={blockedTitle ?? '这次没有断点，会用同一份样本从头训练。'}
        type="primary"
        onClick={() => confirmRetry(projectId, run, onChange)}
      >
        重试
      </Button>
    );
  }
  if (run.next_action === 'start') {
    return (
      <Button
        disabled={disabled}
        icon={Play}
        loading={pending === '开始'}
        title={blockedTitle ?? '用这次运行保存的样本和设置开始训练。'}
        type="primary"
        onClick={() => act('开始', () => api.restart(projectId, run.id), '训练已经开始。')}
      >
        开始
      </Button>
    );
  }
  if (run.next_action === 'mount') {
    return (
      <Button
        disabled={disabled}
        icon={UploadCloud}
        loading={pending === '挂载'}
        title={blockedTitle ?? 'GGUF 已经写好，只会注册到 Ollama。'}
        type="primary"
        onClick={() =>
          act('挂载', () => api.exportRun(projectId, run.id), '开始挂载。已有 GGUF，只会注册到 Ollama。')
        }
      >
        挂到 Ollama
      </Button>
    );
  }
  return null;
}
