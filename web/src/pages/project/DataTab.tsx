import {
  ActionIcon,
  Alert,
  confirmModal,
  createStaticStyles,
  Flexbox,
  Table,
  Tag,
  Text,
  toast,
  Upload,
} from '@lobehub/ui';
import { MetricStrip, Surface } from '@lobehub/ui/dashboard';
import { Trash2 } from 'lucide-react';
import { useState } from 'react';

import { api, type BookStats, type ProjectDetail, uploadBooks } from '../../api';
import { formatBytes, formatChars, formatCount, isLive } from '../../format';

const styles = createStaticStyles(({ css, cssVar }) => ({
  sample: css`
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 16px;

    @media (width <= 800px) {
      grid-template-columns: 1fr;
    }
  `,
  prose: css`
    overflow: auto;

    max-block-size: 220px;
    margin: 0;
    padding: 12px;
    border-radius: ${cssVar.borderRadius};

    font-size: 13px;
    line-height: 1.8;
    color: ${cssVar.colorText};
    white-space: pre-wrap;

    background: ${cssVar.colorFillQuaternary};
  `,
  caption: css`
    margin-block-end: 6px;
    font-size: 12px;
    color: ${cssVar.colorTextTertiary};
  `,
}));

interface BookRow {
  bytes: number;
  name: string;
  sample: boolean;
  skipped: boolean;
  stats: BookStats | null;
}

function cleaningSummary(stats: BookStats): string {
  const parts = [
    stats.junk_lines ? `去掉广告和杂行 ${formatCount(stats.junk_lines)}` : null,
    stats.merged_lines ? `接回断行 ${formatCount(stats.merged_lines)}` : null,
    stats.duplicate_lines ? `重复行 ${formatCount(stats.duplicate_lines)}` : null,
    stats.duplicate_chapters ? `重复章节 ${formatCount(stats.duplicate_chapters)}` : null,
  ].filter(Boolean);
  return parts.length ? parts.join(' · ') : '原文已经很干净';
}

const isSampleName = (name: string) =>
  name.startsWith('示例') || name.toLowerCase().startsWith('example');
const stem = (name: string) => name.replace(/\.[^.]+$/, '');

export function DataTab({ project, onChange }: { project: ProjectDetail; onChange: () => void }) {
  const [uploading, setUploading] = useState(false);
  const training = project.runs.some((run) => isLive(run.state));
  const books = project.dataset?.books ?? [];
  const hasReal = project.raw_files.some((file) => !isSampleName(file.name));
  const rows: BookRow[] = project.raw_files.map((file) => {
    const stats = books.find((book) => book.book === stem(file.name)) ?? null;
    const sample = isSampleName(file.name);
    return { name: file.name, bytes: file.bytes, stats, sample, skipped: sample && hasReal };
  });
  const totalChars = books.reduce((sum, book) => sum + book.chars, 0);

  async function onFiles(files: File[]) {
    if (!files.length) return;
    setUploading(true);
    try {
      const result = await uploadBooks(project.id, files);
      toast.success(`已导入 ${result.saved.length} 本，并重新切好样本`);
      if (result.errors.length) toast.warning(result.errors.join('；'));
      onChange();
    } catch (exc) {
      toast.error(exc instanceof Error ? exc.message : '导入失败');
    } finally {
      setUploading(false);
    }
  }

  function confirmRemove(name: string) {
    confirmModal({
      title: `从项目里移除「${name}」？`,
      content:
        '会删掉项目里的这份原文，并按剩下的书重新切样本。已经开始或跑完的运行用的是开训时的样本副本，不受影响。',
      cancelText: '取消',
      okText: '移除这本',
      okButtonProps: { danger: true },
      onOk: async () => {
        try {
          await api.removeBook(project.id, name);
          toast.success('已移除，样本已重新切好');
          onChange();
        } catch (exc) {
          toast.error(exc instanceof Error ? exc.message : '移除失败');
        }
      },
    });
  }

  return (
    <Flexbox gap={24}>
      <Upload
        dragger
        multiple
        accept=".txt,.text,.md,.epub"
        disabled={uploading}
        title={uploading ? '正在导入…' : '把 txt、md 或 epub 拖到这里，或点击选择，一次可以选多本'}
        description={
          uploading
            ? '正在保存并切分，长篇可能要几十秒'
            : 'txt 用 UTF-8 或 GB18030 都可以。同名文件会覆盖。导入后会立刻切成「前情 → 续写」样本。'
        }
        onFiles={(files) => void onFiles(files)}
      />
      {training ? (
        <Alert
          showIcon
          title="这个项目正在训练。现在导入或移除书，只影响下一次运行。"
          type="info"
        />
      ) : null}
      {!project.ready && project.raw_files.length ? (
        <Alert showIcon title={project.ready_detail} type="warning" />
      ) : null}
      {books.length ? (
        <MetricStrip
          items={[
            {
              label: '参与训练的书',
              value: `${books.length} 本`,
              hint: books.length > 1 ? '多本时每本最多取 2000 条，避免一本占满' : undefined,
            },
            { label: '清洗后字数', value: formatChars(totalChars) },
            { label: '训练样本', value: formatCount(project.dataset?.train_samples) },
            {
              label: '验证样本',
              value: formatCount(project.dataset?.eval_samples),
              hint: '每本最后一段，不参与训练',
            },
          ]}
        />
      ) : null}
      <Surface
        bleed
        description="导入时自动清洗：去掉网址、广告、作者的话和页码，接回排版断开的行，合并多余空行和重复章节，统一省略号、破折号和中文标点。对话每句仍各占一段。"
        title="项目里的书"
      >
        <Table<BookRow>
          dataSource={rows}
          locale={{ emptyText: '还没有书。先在上面导入一本最想模仿的长篇。' }}
          pagination={false}
          rowKey="name"
          columns={[
            {
              key: 'name',
              title: '文件',
              render: (_, row) => (
                <Flexbox horizontal align="center" gap={8}>
                  <Text ellipsis>{row.name}</Text>
                  {row.skipped ? (
                    <Tag>示例，不参与训练</Tag>
                  ) : row.sample ? (
                    <Tag color="warning">示例</Tag>
                  ) : null}
                </Flexbox>
              ),
            },
            {
              key: 'size',
              title: '大小',
              width: 100,
              align: 'end',
              render: (_, row) => formatBytes(row.bytes),
            },
            {
              key: 'chars',
              title: '清洗后',
              width: 110,
              align: 'end',
              render: (_, row) => (row.stats ? formatChars(row.stats.chars) : '—'),
            },
            {
              key: 'chapters',
              title: '章节',
              width: 80,
              align: 'end',
              render: (_, row) => (row.stats ? formatCount(row.stats.chapters) : '—'),
            },
            {
              key: 'cleaning',
              title: '清洗',
              render: (_, row) =>
                row.stats ? <Text type="secondary">{cleaningSummary(row.stats)}</Text> : '—',
            },
            {
              key: 'samples',
              title: '训练 / 验证',
              width: 130,
              align: 'end',
              render: (_, row) =>
                row.stats
                  ? `${formatCount(row.stats.train)} / ${formatCount(row.stats.eval)}`
                  : row.skipped
                    ? '跳过'
                    : '没切出样本',
            },
            {
              key: 'actions',
              title: '',
              width: 56,
              render: (_, row) => (
                <ActionIcon
                  icon={Trash2}
                  size="small"
                  title={`移除 ${row.name}`}
                  onClick={() => confirmRemove(row.name)}
                />
              ),
            },
          ]}
        />
      </Surface>
      {project.samples.length ? (
        <Surface
          description="模型看到左边的前情，学着写出右边这一段。只截了开头和结尾的一部分。"
          title="样本长什么样"
        >
          <Flexbox gap={20}>
            {project.samples.map((sample, index) => (
              <Flexbox gap={6} key={index}>
                <Text type="secondary">
                  {sample.book}
                  {sample.chapter && sample.chapter !== '正文' ? ` · ${sample.chapter}` : ''}
                </Text>
                <div className={styles.sample}>
                  <div>
                    <div className={styles.caption}>前情（末尾）</div>
                    <pre className={styles.prose}>…{sample.input}</pre>
                  </div>
                  <div>
                    <div className={styles.caption}>要学会写的续写</div>
                    <pre className={styles.prose}>{sample.output}…</pre>
                  </div>
                </div>
              </Flexbox>
            ))}
          </Flexbox>
        </Surface>
      ) : null}
    </Flexbox>
  );
}
