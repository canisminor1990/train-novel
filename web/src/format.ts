import type { RunState, RunSummary, StageState } from './api';

export function formatBytes(bytes: number) {
  if (!bytes) return '0 KB';
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

export function formatCount(value: number | null | undefined) {
  return typeof value === 'number' ? value.toLocaleString('zh-CN') : '—';
}

export function formatChars(value: number) {
  if (value >= 10000) return `${(value / 10000).toFixed(value >= 100000 ? 0 : 1)} 万字`;
  return `${value.toLocaleString('zh-CN')} 字`;
}

export function formatLoss(value: number | null | undefined) {
  return typeof value === 'number' ? value.toFixed(3) : '—';
}

export function formatRemaining(seconds: number | null | undefined) {
  const text = formatDuration(seconds);
  if (text === '—') return '还不好估';
  if (text === '不到 1 分钟') return '还不到 1 分钟';
  return `还要 ${text}`;
}

export function formatDuration(seconds: number | null | undefined) {
  if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds < 0) return '—';
  if (seconds < 60) return '不到 1 分钟';
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} 分钟`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest ? `${hours} 小时 ${rest} 分` : `${hours} 小时`;
}

const pad = (value: number) => String(value).padStart(2, '0');

export function formatTime(seconds: number | null | undefined) {
  if (!seconds) return '—';
  const date = new Date(seconds * 1000);
  const now = new Date();
  const time = `${pad(date.getHours())}:${pad(date.getMinutes())}`;
  if (date.toDateString() === now.toDateString()) return `今天 ${time}`;
  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  if (date.toDateString() === yesterday.toDateString()) return `昨天 ${time}`;
  const day = `${date.getMonth() + 1}月${date.getDate()}日`;
  return date.getFullYear() === now.getFullYear()
    ? `${day} ${time}`
    : `${date.getFullYear()}年${day}`;
}

export function finishAt(eta: number | null | undefined) {
  if (typeof eta !== 'number' || !Number.isFinite(eta)) return null;
  return formatTime(Date.now() / 1000 + eta).replace(/^今天 /, '');
}

export function runName(run: Pick<RunSummary, 'id' | 'created'>) {
  if (run.created) return formatTime(run.created);
  const match = /^r-\d{4}(\d{2})(\d{2})-(\d{2})(\d{2})/.exec(run.id);
  return match ? `${Number(match[1])}月${Number(match[2])}日 ${match[3]}:${match[4]}` : run.id;
}

export function runElapsed(run: Pick<RunSummary, 'started' | 'ended' | 'alive'>) {
  if (!run.started) return null;
  const end = run.ended || (run.alive ? Date.now() / 1000 : null);
  return end ? end - run.started : null;
}

export const RUN_STATE: Record<RunState, { label: string; color?: string }> = {
  queued: { label: '启动中', color: 'processing' },
  running: { label: '进行中', color: 'processing' },
  done: { label: '已挂载', color: 'success' },
  trained: { label: '已训练', color: 'warning' },
  failed: { label: '失败', color: 'error' },
  stopped: { label: '已停止' },
  interrupted: { label: '中断', color: 'warning' },
};

export function isLive(state: RunState | undefined) {
  return state === 'queued' || state === 'running';
}

export const STAGE_STATUS: Record<StageState, 'wait' | 'process' | 'finish' | 'error'> = {
  pending: 'wait',
  running: 'process',
  done: 'finish',
  failed: 'error',
  stopped: 'error',
};
