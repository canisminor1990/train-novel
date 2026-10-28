import { Checkbox, Flexbox, Input, Modal, Text, toast } from '@lobehub/ui';
import { useState } from 'react';

import { api, type ProjectSummary } from '../api';
import { type ModelFieldState, ModelNameField } from './ModelNameField';

export function ProjectSettingsModal({
  project,
  onClose,
  onDone,
}: {
  project: ProjectSummary;
  onClose: () => void;
  onDone: () => void;
}) {
  const [name, setName] = useState(project.name);
  const [model, setModel] = useState(project.ollama.model);
  const [modelState, setModelState] = useState<ModelFieldState>('ok');
  const [saving, setSaving] = useState(false);
  const modelChanged = model.trim() !== project.ollama.model;

  async function submit() {
    if (!name.trim()) {
      toast.error('项目名不能为空。');
      return;
    }
    setSaving(true);
    try {
      const result = await api.updateProject(project.id, { name, ollama_model: model.trim() });
      toast.success(result.note ?? '已保存');
      onDone();
      onClose();
    } catch (exc) {
      toast.error(exc instanceof Error ? exc.message : '保存失败');
    } finally {
      setSaving(false);
    }
  }

  return (
    <Modal
      open
      cancelText="取消"
      confirmLoading={saving}
      okButtonProps={{ disabled: modelState !== 'ok' }}
      okText="保存"
      title="项目设置"
      onCancel={onClose}
      onOk={() => void submit()}
    >
      <Flexbox gap={16}>
        <Flexbox gap={6}>
          <label htmlFor="settings-name">项目名</label>
          <Input
            autoFocus
            id="settings-name"
            maxLength={40}
            value={name}
            onChange={(event) => setName(event.target.value)}
            onPressEnter={() => void submit()}
          />
        </Flexbox>
        <ModelNameField
          emptyHint="模型名不能为空。"
          id="settings-model"
          projectId={project.id}
          value={model}
          onChange={setModel}
          onState={setModelState}
        />
        {modelChanged && (project.ollama.ready || project.ollama.think_ready) ? (
          <Text type="warning">
            {`保存时会把 Ollama 里的 ${project.ollama.model}${project.ollama.think_ready ? ` 和 ${project.ollama.think_model}` : ''} 改成新名字，旧名字不再可用。`}
          </Text>
        ) : null}
      </Flexbox>
    </Modal>
  );
}

export function DeleteProjectModal({
  project,
  onClose,
  onDeleted,
}: {
  project: ProjectSummary;
  onClose: () => void;
  onDeleted: (removed: boolean) => void;
}) {
  const [deleting, setDeleting] = useState(false);
  const [removeFiles, setRemoveFiles] = useState(true);
  const [removeModels, setRemoveModels] = useState(true);
  const mounted = project.ollama.ready || project.ollama.think_ready;
  const models = project.ollama.think_ready
    ? `${project.ollama.model} 和 ${project.ollama.think_model}`
    : project.ollama.model;

  async function submit() {
    setDeleting(true);
    try {
      const result = await api.deleteProject(project.id, {
        files: removeFiles,
        models: mounted && removeModels,
      });
      if (result.deleted) toast.success(result.note ? `项目已删除。${result.note}` : '项目已删除');
      else toast.success(result.note ?? '已从 Ollama 删除');
      onDeleted(result.deleted);
      onClose();
    } catch (exc) {
      toast.error(exc instanceof Error ? exc.message : '删除失败');
    } finally {
      setDeleting(false);
    }
  }

  return (
    <Modal
      open
      cancelText="取消"
      confirmLoading={deleting}
      okButtonProps={{ danger: true, disabled: !removeFiles && !(mounted && removeModels) }}
      okText="删除项目"
      title={`删除「${project.name}」？`}
      onCancel={onClose}
      onOk={() => void submit()}
    >
      <Flexbox gap={16}>
        <Checkbox checked={removeFiles} onChange={setRemoveFiles}>
          {`删除项目目录：${project.books} 本原文、切好的样本，以及 ${project.run_count} 次运行的 LoRA、断点和 GGUF`}
        </Checkbox>
        {mounted ? (
          <Checkbox checked={removeModels} onChange={setRemoveModels}>
            {`从 Ollama 删除 ${models}`}
          </Checkbox>
        ) : (
          <Text type="secondary">Ollama 里还没有 {project.ollama.model}，不用另外清理。</Text>
        )}
        <Text type="secondary">勾选的内容不能恢复。</Text>
      </Flexbox>
    </Modal>
  );
}
