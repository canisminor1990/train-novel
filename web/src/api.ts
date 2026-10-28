export type StageState = 'pending' | 'running' | 'done' | 'failed' | 'stopped';
export type RunState =
  'queued' | 'running' | 'done' | 'trained' | 'failed' | 'stopped' | 'interrupted';
export type RunAction = 'pause' | 'continue' | 'retry' | 'start' | 'mount';

export interface Stage {
  detail: string;
  eta: number | null;
  id: string;
  label: string;
  progress: number | null;
  started: number | null;
  state: StageState;
  step: number | null;
  total: number | null;
  updated: number | null;
}

export interface RunSummary {
  alive: boolean;
  best_eval: number | null;
  best_eval_step: number | null;
  created: number | null;
  dataset: { books: string[]; train_samples: number; eval_samples: number } | null;
  detail: string;
  ended: number | null;
  eta: number | null;
  eval_rising: boolean;
  first_loss: number | null;
  has_adapter: boolean;
  has_checkpoint: boolean;
  has_gguf: boolean;
  has_merged: boolean;
  id: string;
  last_eval: number | null;
  last_loss: number | null;
  last_step: number;
  mounted: boolean;
  next_action: RunAction | null;
  phase: string | null;
  preset: string | null;
  preset_label: string;
  progress: number | null;
  project: string;
  stage: string | null;
  started: number | null;
  state: RunState;
  step: number | null;
  total: number | null;
}

export interface MetricRow {
  eval_loss?: number;
  learning_rate?: number;
  loss?: number;
  step: number;
  time?: number;
}

export interface RunDetail extends RunSummary {
  artifacts: {
    dir: string;
    lora_bytes: number;
    gguf: string | null;
    gguf_bytes: number;
    disk_bytes: number;
  };
  config: Record<string, unknown> | null;
  export_log: string;
  metrics: MetricRow[];
  ollama_model: string;
  ollama_think_model: string;
  preview: string;
  stages: Stage[];
  train_log: string;
}

export interface OllamaInfo {
  installed: boolean;
  model: string;
  ready: boolean;
  think_model: string;
  think_ready: boolean;
}

export interface ModelCheck {
  error?: string;
  exists?: boolean;
  model?: string;
  ok: boolean;
  think_model?: string;
  valid: boolean;
}

export interface ActiveJob {
  kind: 'train' | 'export';
  pid: number;
  project: string;
  run: string;
  started: number;
}

export interface ProjectSummary {
  book_names: string[];
  books: number;
  created: number;
  id: string;
  latest_run: RunSummary | null;
  mounted_run: string | null;
  name: string;
  ollama: OllamaInfo;
  preset: string | null;
  ready: boolean;
  ready_detail: string;
  run_count: number;
  train_samples: number;
  updated: number;
}

export interface Preset {
  early_stop: boolean;
  epochs: number;
  id: string;
  label: string;
  max_steps: number | null;
  rank: number;
  selected: boolean;
  seq: number;
  steps: number | null;
  summary: string;
}

export interface BookStats {
  book: string;
  chapters: number;
  chars: number;
  duplicate_chapters?: number;
  duplicate_lines?: number;
  eval: number;
  junk_lines: number;
  merged_lines?: number;
  raw_chars?: number;
  train: number;
}

export interface ProjectDetail extends ProjectSummary {
  active: ActiveJob | null;
  busy: string | null;
  dataset: { books: BookStats[]; train_samples: number; eval_samples: number } | null;
  presets: { items: Preset[]; selected: string | null };
  raw_files: { name: string; bytes: number; modified: number }[];
  runs: RunSummary[];
  samples: { book: string; chapter: string; input: string; output: string }[];
}

export interface CheckResult {
  fields: {
    torch?: string;
    cuda?: boolean;
    gpu?: string;
    bf16?: boolean;
    vram?: string;
    unsloth?: string;
  };
  log: string;
  ok: boolean;
  updated: number;
}

export interface AppState {
  active: ActiveJob | null;
  busy: string | null;
  check: CheckResult | null;
  gpu: { name: string; used_mib: number; total_mib: number; util: number; temp_c: number } | null;
  projects: ProjectSummary[];
}

export interface RunPayload {
  busy: string | null;
  project: { id: string; name: string; ollama: OllamaInfo };
  run: RunDetail;
}

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function parse<T>(response: Response): Promise<T> {
  let data: unknown;
  try {
    data = await response.json();
  } catch {
    throw new ApiError(
      response.ok ? '返回内容读不出来' : `后台返回 ${response.status}`,
      response.status,
    );
  }
  const body = data as { ok?: boolean; error?: string };
  if (!response.ok || body.ok === false)
    throw new ApiError(body.error || '没有完成', response.status);
  return data as T;
}

export async function getJson<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path);
  } catch {
    throw new ApiError('连不上后台。确认 python -m novel_train.dashboard 还在运行。', 0);
  }
  return parse<T>(response);
}

export async function postJson<T = { ok: boolean }>(path: string, body?: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body ?? {}),
    });
  } catch {
    throw new ApiError('连不上后台。确认 python -m novel_train.dashboard 还在运行。', 0);
  }
  return parse<T>(response);
}

export async function uploadBooks(projectId: string, files: File[]) {
  const form = new FormData();
  for (const file of files) form.append('file', file, file.name);
  const response = await fetch(`/api/projects/${projectId}/import`, { method: 'POST', body: form });
  return parse<{ ok: boolean; saved: string[]; errors: string[]; log: string }>(response);
}

export const api = {
  state: () => getJson<AppState>('/api/state'),
  project: (id: string) => getJson<ProjectDetail>(`/api/projects/${id}`),
  run: (id: string, run: string) => getJson<RunPayload>(`/api/projects/${id}/runs/${run}`),
  createProject: (name: string, ollamaModel: string) =>
    postJson<{ ok: boolean; project: ProjectSummary; warning: string | null }>('/api/projects', {
      name,
      ollama_model: ollamaModel,
    }),
  checkModel: (ollamaModel: string, projectId?: string) =>
    postJson<ModelCheck>('/api/models/check', { ollama_model: ollamaModel, project: projectId }),
  updateProject: (id: string, fields: { name: string; ollama_model: string }) =>
    postJson<{ ok: boolean; project: ProjectSummary; note: string | null }>(
      `/api/projects/${id}/settings`,
      fields,
    ),
  deleteProject: (id: string, removeModel: boolean) =>
    postJson<{ ok: boolean; note: string | null }>(`/api/projects/${id}/delete`, {
      remove_model: removeModel,
    }),
  setPreset: (id: string, preset: string) => postJson(`/api/projects/${id}/preset`, { preset }),
  removeBook: (id: string, name: string) =>
    postJson<{ ok: boolean; log: string }>(`/api/projects/${id}/files/delete`, { name }),
  train: (id: string) => postJson<{ ok: boolean; run: string }>(`/api/projects/${id}/train`),
  stop: (id: string, run: string) => postJson(`/api/projects/${id}/runs/${run}/stop`),
  resume: (id: string, run: string) => postJson(`/api/projects/${id}/runs/${run}/resume`),
  restart: (id: string, run: string) => postJson(`/api/projects/${id}/runs/${run}/restart`),
  exportRun: (id: string, run: string, force = false) =>
    postJson(`/api/projects/${id}/runs/${run}/export`, force ? { force: true } : {}),
  deleteRun: (id: string, run: string) => postJson(`/api/projects/${id}/runs/${run}/delete`),
  chat: (id: string, text: string) =>
    postJson<{ ok: boolean; text: string; model: string }>(`/api/projects/${id}/chat`, { text }),
  check: () => postJson<CheckResult>('/api/check'),
};
