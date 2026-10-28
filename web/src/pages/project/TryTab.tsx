import { Ollama } from '@lobehub/icons';
import {
  Alert,
  Button,
  CopyButton,
  createStaticStyles,
  Flexbox,
  Tag,
  Text,
  TextArea,
  toast,
} from '@lobehub/ui';
import { Surface } from '@lobehub/ui/dashboard';
import { PenLine } from 'lucide-react';
import { useState } from 'react';

import { api, type ProjectDetail } from '../../api';
import { runName } from '../../format';
import { go, href } from '../../router';

const MIN = 20;
const MAX = 4000;

const styles = createStaticStyles(({ css, cssVar }) => ({
  prose: css`
    min-block-size: 160px;
    margin: 0;
    padding: 16px;
    border-radius: ${cssVar.borderRadiusLG};

    font-size: 15px;
    line-height: 1.9;
    color: ${cssVar.colorText};
    white-space: pre-wrap;

    background: ${cssVar.colorFillQuaternary};
  `,
}));

export function TryTab({ project }: { project: ProjectDetail }) {
  const [context, setContext] = useState('');
  const [prose, setProse] = useState('');
  const [writing, setWriting] = useState(false);
  const { ollama } = project;
  const mounted = project.runs.find((run) => run.id === project.mounted_run);
  const length = context.trim().length;

  let blocked: string | null = null;
  if (!ollama.installed) blocked = '没有找到 ollama.exe。装好 Ollama 后刷新这一页。';
  else if (!ollama.ready)
    blocked = project.runs.some((run) => run.has_adapter)
      ? `${ollama.model} 还没挂上。到运行详情里点「继续」，导出写完后会挂上。`
      : `${ollama.model} 还没挂上。训练一次，结束后会自动挂载。`;
  else if (project.busy) blocked = `${project.busy}等它结束再续写，避免抢显存。`;

  async function send() {
    setWriting(true);
    try {
      const result = await api.chat(project.id, context);
      setProse(result.text || '模型没有写出正文。');
    } catch (exc) {
      toast.error(exc instanceof Error ? exc.message : '续写失败');
    } finally {
      setWriting(false);
    }
  }

  const hint =
    length === 0
      ? `至少 ${MIN} 字`
      : length < MIN
        ? `还差 ${MIN - length} 字`
        : length > MAX
          ? `超过 ${MAX} 字，先截到最近的一段`
          : `${length} 字`;

  return (
    <Flexbox gap={24}>
      <Flexbox gap={8}>
        <Flexbox horizontal align="center" gap={10} wrap="wrap">
          <Ollama.Combine size={22} />
          <Text code>{ollama.model}</Text>
          <Tag>正文</Tag>
          {ollama.ready ? <Tag color="success">已挂载</Tag> : <Tag>未挂载</Tag>}
          {mounted ? (
            <Text type="secondary">
              来自 <a href={href.run(project.id, mounted.id)}>{runName(mounted)} 的运行</a>
            </Text>
          ) : null}
          {ollama.ready ? (
            <CopyButton content={`ollama run ${ollama.model}`} title="复制正文模型命令" />
          ) : null}
        </Flexbox>
        <Flexbox horizontal align="center" gap={10} wrap="wrap">
          <Text code>{ollama.think_model}</Text>
          <Tag>带思考</Tag>
          {ollama.think_ready ? <Tag color="success">已挂载</Tag> : <Tag>未挂载</Tag>}
          {ollama.think_ready ? (
            <CopyButton content={`ollama run ${ollama.think_model}`} title="复制带思考模型命令" />
          ) : null}
        </Flexbox>
      </Flexbox>
      {blocked ? (
        <Alert
          showIcon
          title={blocked}
          type="info"
          action={
            project.runs[0] ? (
              <Button size="small" onClick={() => go(href.project(project.id, 'runs'))}>
                看运行记录
              </Button>
            ) : undefined
          }
        />
      ) : null}
      <Surface
        description="这里调用本机 Ollama 里的模型，不会再把 LoRA 载入显存。一次写一个场景，把角色状态和前情一起贴进来。"
        title="贴一段前情"
      >
        <Flexbox gap={12}>
          <TextArea
            aria-label="前情"
            autoSize={{ minRows: 6, maxRows: 16 }}
            placeholder={`贴一段至少 ${MIN} 个字的前情`}
            value={context}
            onChange={(event) => setContext(event.target.value)}
          />
          <Flexbox horizontal align="center" gap={12}>
            <Button
              disabled={Boolean(blocked) || length < MIN || length > MAX}
              icon={PenLine}
              loading={writing}
              type="primary"
              onClick={() => void send()}
            >
              {writing ? '正在续写，最长 3 分钟' : '续写'}
            </Button>
            <Text type={length > MAX ? 'danger' : 'secondary'}>{hint}</Text>
          </Flexbox>
        </Flexbox>
      </Surface>
      {prose ? (
        <Surface extra={<CopyButton content={prose} title="复制续写" />} title="续写">
          <pre className={styles.prose}>{prose}</pre>
        </Surface>
      ) : null}
    </Flexbox>
  );
}
