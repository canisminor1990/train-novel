import { Ollama } from '@lobehub/icons';
import {
  Alert,
  Button,
  CopyButton,
  createStaticStyles,
  Flexbox,
  InputNumber,
  Tag,
  Text,
  TextArea,
  toast,
} from '@lobehub/ui';
import { Surface } from '@lobehub/ui/dashboard';
import { PenLine } from 'lucide-react';
import { useState } from 'react';

import { api, type ProjectDetail } from '../../api';
import { formatChars, runName } from '../../format';
import { go, href } from '../../router';

const PREMISE_MIN = 4;
const PREMISE_MAX = 2000;
const CONTEXT_MAX = 30_000;
const LENGTH_MIN = 300;
const LENGTH_MAX = 8000;

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
  field: css`
    font-size: 13px;
    color: ${cssVar.colorTextSecondary};
  `,
}));

export function TryTab({ project }: { project: ProjectDetail }) {
  const [premise, setPremise] = useState('');
  const [context, setContext] = useState('');
  const [length, setLength] = useState(5000);
  const [prose, setProse] = useState('');
  const [writing, setWriting] = useState(false);
  const { ollama } = project;
  const mounted = project.runs.find((run) => run.id === project.mounted_run);
  const premiseLength = premise.trim().length;
  const contextLength = context.trim().length;

  let blocked: string | null = null;
  if (!ollama.installed) blocked = '没有找到 ollama.exe。装好 Ollama 后刷新这一页。';
  else if (!ollama.ready)
    blocked = project.runs.some((run) => run.has_adapter)
      ? `${ollama.model} 还没挂上。到运行详情里点「继续」，导出写完后会挂上。`
      : `${ollama.model} 还没挂上。训练一次，结束后会自动挂载。`;
  else if (project.busy) blocked = `${project.busy}等它结束再写，避免抢显存。`;

  const invalid =
    premiseLength < PREMISE_MIN
      ? `命题至少 ${PREMISE_MIN} 个字`
      : premiseLength > PREMISE_MAX
        ? `命题最多 ${PREMISE_MAX} 字，细节放进前情`
        : contextLength > CONTEXT_MAX
          ? `前情超过 ${CONTEXT_MAX} 字，先截到最近的一段`
          : null;

  async function send() {
    setWriting(true);
    try {
      const result = await api.write(project.id, { premise, length, context });
      setProse(result.text || '模型没有写出正文。');
    } catch (exc) {
      toast.error(exc instanceof Error ? exc.message : '没写出来');
    } finally {
      setWriting(false);
    }
  }

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
              来自 <a href={href.run(project.id, mounted.id)}>{runName(mounted)} 的运行</a> ·{' '}
              {mounted.model_label}
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
        description="调用本机 Ollama 里的模型，不会再把 LoRA 载入显存。命题写清楚这一章谁出场、在哪、发生什么、停在哪；写得越具体，跑题越少。一次写一章，4000 到 6500 字最稳，更长就分章写，把上一章的结尾贴进前情。"
        title="给一个命题"
      >
        <Flexbox gap={12}>
          <TextArea
            aria-label="命题"
            autoSize={{ minRows: 3, maxRows: 10 }}
            placeholder="例如：雨夜，沈砚在旧码头等一个十年没见的人。来的却是对方的女儿，带着一封没寄出的信。两人从戒备到松动，最后停在她说出父亲死因的那一刻。"
            value={premise}
            onChange={(event) => setPremise(event.target.value)}
          />
          <Flexbox horizontal align="center" gap={8}>
            <span className={styles.field}>大约写</span>
            <InputNumber
              aria-label="字数"
              max={LENGTH_MAX}
              min={LENGTH_MIN}
              step={500}
              value={length}
              onChange={(value) => setLength(typeof value === 'number' ? value : 5000)}
            />
            <span className={styles.field}>字</span>
          </Flexbox>
          <TextArea
            aria-label="前情（可选）"
            autoSize={{ minRows: 2, maxRows: 10 }}
            placeholder="前情，可选。接着已有的章节写时，贴上一章最后几百字。"
            value={context}
            onChange={(event) => setContext(event.target.value)}
          />
          <Flexbox horizontal align="center" gap={12}>
            <Button
              disabled={Boolean(blocked) || Boolean(invalid)}
              icon={PenLine}
              loading={writing}
              type="primary"
              onClick={() => void send()}
            >
              {writing ? '正在写，一般 1 到 3 分钟' : '写这一章'}
            </Button>
            <Text type={invalid && premiseLength > 0 ? 'danger' : 'secondary'}>
              {invalid ?? `命题 ${premiseLength} 字${contextLength ? ` · 前情 ${contextLength} 字` : ''}`}
            </Text>
          </Flexbox>
        </Flexbox>
      </Surface>
      {prose ? (
        <Surface
          description={formatChars(prose.replaceAll(/\s/g, '').length)}
          extra={<CopyButton content={prose} title="复制正文" />}
          title="正文"
        >
          <pre className={styles.prose}>{prose}</pre>
        </Surface>
      ) : null}
    </Flexbox>
  );
}
