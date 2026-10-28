import {
  ActionIcon,
  Alert,
  Button,
  createStaticStyles,
  DropdownMenu,
  Empty,
  Flexbox,
  Input,
  Modal,
  Table,
  Tag,
  Text,
  toast,
} from '@lobehub/ui';
import { PageHeader, Surface } from '@lobehub/ui/dashboard';
import { BookPlus, FolderPlus, MoreHorizontal, Settings2, Stethoscope, Trash2 } from 'lucide-react';
import { useState } from 'react';

import { api, type AppState, type ProjectSummary } from '../api';
import { type ModelFieldState, ModelNameField, slugModelName } from '../components/ModelNameField';
import { DeleteProjectModal, ProjectSettingsModal } from '../components/ProjectDialogs';
import { RunStateTag } from '../components/RunStateTag';
import { formatCount, formatTime } from '../format';
import { go, href } from '../router';
import type { Poll } from '../usePoll';

const styles = createStaticStyles(({ css, cssVar }) => ({
  name: css`
    font-weight: 500;
    color: ${cssVar.colorText};
  `,
  check: css`
    margin: 0;
    padding: 12px;
    border-radius: ${cssVar.borderRadius};

    font-family: ${cssVar.fontFamilyCode};
    font-size: 12px;
    color: ${cssVar.colorTextSecondary};
    white-space: pre-wrap;

    background: ${cssVar.colorFillQuaternary};
  `,
}));

function NewProjectModal({
  open,
  onClose,
  onCreated,
}: {
  open: boolean;
  onClose: () => void;
  onCreated: () => void;
}) {
  const [name, setName] = useState('');
  const [model, setModel] = useState('');
  const [modelEdited, setModelEdited] = useState(false);
  const [modelState, setModelState] = useState<ModelFieldState>('empty');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  async function submit() {
    if (!name.trim()) {
      setError('先给项目起个名字，比如书名。');
      return;
    }
    if (modelState === 'error' || modelState === 'checking') return;
    setSaving(true);
    try {
      const result = await api.createProject(name.trim(), model.trim());
      toast.success(`已新建「${result.project.name}」，模型名 ${result.project.ollama.model}`);
      if (result.warning) toast.warning(result.warning);
      setName('');
      setModel('');
      setModelEdited(false);
      onCreated();
      onClose();
      go(href.project(result.project.id, 'data'));
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : '新建失败');
    } finally {
      setSaving(false);
    }
  }

  return (
    <Modal
      cancelText="取消"
      confirmLoading={saving}
      okButtonProps={{ disabled: modelState === 'error' || modelState === 'checking' }}
      okText="新建项目"
      open={open}
      title="新建项目"
      onCancel={onClose}
      onOk={() => void submit()}
    >
      <Flexbox gap={8}>
        <Text type="secondary">
          一个项目放一本书，或同一作者、同一题材的几本。原文、样本、每次训练和 Ollama
          模型都只属于这个项目。
        </Text>
        <label htmlFor="project-name">项目名</label>
        <Input
          autoFocus
          aria-describedby={error ? 'project-name-error' : undefined}
          aria-invalid={error ? true : undefined}
          id="project-name"
          maxLength={40}
          placeholder="例如：夜航"
          value={name}
          onPressEnter={() => void submit()}
          onChange={(event) => {
            setName(event.target.value);
            setError('');
            if (!modelEdited) setModel(slugModelName(event.target.value));
          }}
        />
        {error ? (
          <Text id="project-name-error" type="danger">
            {error}
          </Text>
        ) : null}
        <ModelNameField
          emptyHint="书名是中文时请自己起一个英文或拼音名，比如 yehang。留空就用 novel-日期。"
          id="project-model"
          value={model}
          onState={setModelState}
          onChange={(value) => {
            setModel(value);
            setModelEdited(Boolean(value.trim()));
          }}
        />
      </Flexbox>
    </Modal>
  );
}

function MachineSection({ state }: { state: AppState | null }) {
  const [checking, setChecking] = useState(false);
  const check = state?.check;
  async function runCheck() {
    setChecking(true);
    try {
      const result = await api.check();
      if (result.ok) toast.success('环境正常');
      else toast.error('环境检查没有通过，看下方日志。');
    } catch (exc) {
      toast.error(exc instanceof Error ? exc.message : '检查失败');
    } finally {
      setChecking(false);
    }
  }
  const fields = check?.fields;
  return (
    <Surface
      description="确认 CUDA、bf16 和 Unsloth 都能用。会短暂占用显卡，训练时不能检查。"
      title="这台机器"
      extra={
        <Button
          disabled={Boolean(state?.busy)}
          icon={Stethoscope}
          loading={checking}
          size="small"
          onClick={() => void runCheck()}
        >
          {checking ? '正在检查，最多 3 分钟' : '检查环境'}
        </Button>
      }
    >
      {check ? (
        <Flexbox gap={12}>
          <Flexbox horizontal align="center" gap={8} wrap="wrap">
            <Tag color={check.ok ? 'success' : 'error'}>{check.ok ? '通过' : '没有通过'}</Tag>
            {fields?.gpu ? <Tag>{fields.gpu}</Tag> : null}
            {fields?.torch ? <Tag>torch {fields.torch}</Tag> : null}
            {fields?.unsloth ? <Tag>unsloth {fields.unsloth}</Tag> : null}
            {fields?.bf16 != null ? (
              <Tag color={fields.bf16 ? undefined : 'error'}>
                bf16 {fields.bf16 ? '可用' : '不可用'}
              </Tag>
            ) : null}
            <Text type="secondary">上次检查 {formatTime(check.updated)}</Text>
          </Flexbox>
          {!check.ok ? <pre className={styles.check}>{check.log}</pre> : null}
        </Flexbox>
      ) : (
        <Text type="secondary">还没有检查过。第一次训练前建议先检查一次。</Text>
      )}
    </Surface>
  );
}

function ProjectTable({
  activeProject,
  projects,
  onChange,
}: {
  activeProject: string | undefined;
  projects: ProjectSummary[];
  onChange: () => void;
}) {
  const [dialog, setDialog] = useState<{
    kind: 'settings' | 'delete';
    project: ProjectSummary;
  } | null>(null);
  return (
    <>
      <Table<ProjectSummary>
        dataSource={projects}
        pagination={false}
        rowKey="id"
        variant="outlined"
        columns={[
          {
            key: 'name',
            title: '项目',
            render: (_, project) => (
              <Flexbox gap={2}>
                <a className={styles.name} href={href.project(project.id)}>
                  {project.name}
                </a>
                <Text ellipsis type="secondary">
                  {project.book_names.length ? project.book_names.join('、') : '还没有书'}
                </Text>
              </Flexbox>
            ),
          },
          {
            key: 'samples',
            title: '样本',
            width: 150,
            render: (_, project) =>
              project.books
                ? `${project.books} 本 · ${formatCount(project.train_samples)} 条`
                : '—',
          },
          {
            key: 'run',
            title: '最近一次运行',
            width: 210,
            render: (_, project) =>
              project.latest_run ? (
                <Flexbox horizontal align="center" gap={8}>
                  <RunStateTag state={project.latest_run.state} />
                  <Text type="secondary">{formatTime(project.latest_run.created)}</Text>
                </Flexbox>
              ) : (
                <Text type="secondary">没有训练过</Text>
              ),
          },
          {
            key: 'ollama',
            title: 'Ollama 模型',
            width: 260,
            render: (_, project) => (
              <Flexbox gap={2}>
                <Flexbox horizontal align="center" gap={6}>
                  <Text code ellipsis>
                    {project.ollama.model}
                  </Text>
                  {project.ollama.ready ? <Tag color="success">可写作</Tag> : null}
                </Flexbox>
                <Text code ellipsis type="secondary">
                  {project.ollama.think_model}
                </Text>
              </Flexbox>
            ),
          },
          {
            key: 'actions',
            title: '',
            width: 56,
            render: (_, project) => {
              const busy = project.id === activeProject;
              return (
                <DropdownMenu
                  items={[
                    {
                      key: 'settings',
                      icon: Settings2,
                      label: '项目设置',
                      onClick: () => setDialog({ kind: 'settings', project }),
                    },
                    { type: 'divider' },
                    {
                      key: 'delete',
                      icon: Trash2,
                      label: busy ? '删除项目（训练中不能删）' : '删除项目',
                      danger: true,
                      disabled: busy,
                      onClick: () => setDialog({ kind: 'delete', project }),
                    },
                  ]}
                >
                  <ActionIcon
                    icon={MoreHorizontal}
                    size="small"
                    title={`${project.name} 的更多操作`}
                  />
                </DropdownMenu>
              );
            },
          },
        ]}
        onRow={(project) => ({
          onClick: (event) => {
            const target = event.target as HTMLElement;
            if (!event.currentTarget.contains(target) || target.closest('a, button')) return;
            go(href.project(project.id));
          },
          style: { cursor: 'pointer' },
        })}
      />
      {dialog?.kind === 'settings' ? (
        <ProjectSettingsModal
          project={dialog.project}
          onClose={() => setDialog(null)}
          onDone={onChange}
        />
      ) : null}
      {dialog?.kind === 'delete' ? (
        <DeleteProjectModal
          project={dialog.project}
          onClose={() => setDialog(null)}
          onDeleted={() => onChange()}
        />
      ) : null}
    </>
  );
}

export function HomePage({ state }: { state: Poll<AppState> }) {
  const [creating, setCreating] = useState(false);
  const projects = state.data?.projects ?? [];
  return (
    <>
      <PageHeader
        description="每个项目单独保存原文、样本、训练记录和 Ollama 模型。不同的书分开建项目，不要混在一起训练。"
        title="项目"
        action={
          projects.length ? (
            <Button icon={FolderPlus} type="primary" onClick={() => setCreating(true)}>
              新建项目
            </Button>
          ) : undefined
        }
      />
      {state.error && !state.data ? (
        <Alert showIcon title={state.error.message} type="error" />
      ) : null}
      {state.data?.busy ? <Alert showIcon title={state.data.busy} type="info" /> : null}
      {state.loading ? (
        <Table<ProjectSummary>
          loading
          columns={[{ key: 'name', title: '项目' }]}
          dataSource={[]}
          pagination={false}
          variant="outlined"
        />
      ) : projects.length ? (
        <ProjectTable
          activeProject={state.data?.active?.project}
          projects={projects}
          onChange={() => void state.refresh()}
        />
      ) : (
        <Empty
          description="先用一本最想模仿的长篇建一个项目。跑通之后，再把同一作者的书加进同一个项目。"
          title="还没有项目"
          type="page"
          action={
            <Button icon={BookPlus} type="primary" onClick={() => setCreating(true)}>
              新建第一个项目
            </Button>
          }
        />
      )}
      <MachineSection state={state.data} />
      <NewProjectModal
        open={creating}
        onClose={() => setCreating(false)}
        onCreated={() => void state.refresh()}
      />
    </>
  );
}
