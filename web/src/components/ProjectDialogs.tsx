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
  onDeleted: () => void;
}) {
  const [removeModel, setRemoveModel] = useState(project.ollama.ready || project.ollama.think_ready);
  const [deleting, setDeleting] = useState(false);

  async function submit() {
    setDeleting(true);
    try {
      const result = await api.deleteProject(project.id, removeModel);
      toast.success(result.note ? `项目已删除。${result.note}` : '项目已删除');
      onDeleted();
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
      okButtonProps={{ danger: true }}
      okText="删除项目"
      title={`删除「${project.name}」？`}
      onCancel={onClose}
      onOk={() => void submit()}
    >
      <Flexbox gap={16}>
        <Text>
          会删掉这个项目里的 {project.books} 本原文、切好的样本，以及 {project.run_count} 次运行的
          LoRA、断点和 GGUF，不能恢复。
        </Text>
        {project.ollama.ready || project.ollama.think_ready ? (
          <Checkbox checked={removeModel} onChange={setRemoveModel}>
            同时从 Ollama 删除 {project.ollama.model}
            {project.ollama.think_ready ? ` 和 ${project.ollama.think_model}` : ''}
          </Checkbox>
        ) : (
          <Text type="secondary">Ollama 里还没有 {project.ollama.model}，不用另外清理。</Text>
        )}
      </Flexbox>
    </Modal>
  );
}
