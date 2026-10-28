import { Qwen } from '@lobehub/icons';
import { createStaticStyles, Flexbox, Icon, Tag, Tooltip } from '@lobehub/ui';
import {
  Breadcrumb,
  type BreadcrumbItem,
  ConsoleBrand,
  ConsoleNav,
  ConsoleShell,
} from '@lobehub/ui/dashboard';
import { BookOpen, LayoutGrid, LoaderCircle } from 'lucide-react';
import { useEffect, useMemo } from 'react';

import { api, type AppState } from './api';
import { ThemeSwitch } from './components/ThemeSwitch';
import { runName } from './format';
import { HomePage } from './pages/HomePage';
import { ProjectPage } from './pages/ProjectPage';
import { RunPage } from './pages/RunPage';
import { href, useRoute } from './router';
import { usePoll } from './usePoll';

const styles = createStaticStyles(({ css }) => ({
  main: css`
    display: flex;
    flex-direction: column;
    gap: 24px;

    inline-size: 100%;
    max-inline-size: 1120px;
    margin-inline: auto;
    padding-block: 8px 48px;
  `,
  shell: css`
    inline-size: 100%;
    block-size: 100%;
  `,
}));

function GpuTag({ state }: { state: AppState | null }) {
  const gpu = state?.gpu;
  if (!state) return <Tag>正在读取显卡</Tag>;
  if (!gpu) return <Tag color="warning">没有读到显卡</Tag>;
  const used = (gpu.used_mib / 1024).toFixed(1);
  const total = (gpu.total_mib / 1024).toFixed(0);
  return (
    <Tooltip title={`${gpu.name} · 利用率 ${gpu.util}% · ${gpu.temp_c}°C`}>
      <Tag>
        显存 {used} / {total} GB
      </Tag>
    </Tooltip>
  );
}

function ActiveTag({ state }: { state: AppState | null }) {
  const active = state?.active;
  if (!active) return null;
  const project = state.projects.find((item) => item.id === active.project);
  const run = project?.latest_run?.id === active.run ? project.latest_run : null;
  const percent =
    run?.progress != null && active.kind === 'train' ? ` ${Math.round(run.progress * 100)}%` : '';
  const premise = active.kind === 'premise';
  const label = premise ? '写命题中' : active.kind === 'train' ? '训练中' : '导出中';
  return (
    <a href={premise ? href.project(active.project, 'data') : href.run(active.project, active.run)}>
      <Tag color="processing" icon={<Icon aria-hidden spin icon={LoaderCircle} size={12} />}>
        {project?.name ?? active.project} · {label}
        {percent}
      </Tag>
    </a>
  );
}

export function App() {
  const route = useRoute();
  const state = usePoll(api.state, 3000, 'state');
  const projects = state.data?.projects ?? [];
  const activeProject = state.data?.active?.project;

  const pageKey =
    route.page === 'home'
      ? 'home'
      : route.page === 'run'
        ? `${route.projectId}/${route.runId}`
        : route.projectId;
  useEffect(() => {
    for (let node = document.getElementById('main'); node; node = node.parentElement)
      node.scrollTop = 0;
    window.scrollTo(0, 0);
  }, [pageKey]);

  const pathname = route.page === 'home' ? href.home() : href.project(route.projectId);
  const currentProject =
    route.page === 'home' ? null : projects.find((item) => item.id === route.projectId);

  const breadcrumb = useMemo<BreadcrumbItem[]>(() => {
    const items: BreadcrumbItem[] = [{ label: '项目', href: href.home() }];
    if (route.page === 'home') return items;
    items.push({
      label: currentProject?.name ?? route.projectId,
      href: href.project(route.projectId),
    });
    if (route.page === 'run')
      items.push({
        label: '运行记录',
        href: href.project(route.projectId, 'runs'),
        optional: true,
      });
    if (route.page === 'run') items.push({ label: runName({ id: route.runId, created: null }) });
    return items;
  }, [route, currentProject]);

  return (
    <ConsoleShell
      breadcrumb={<Breadcrumb items={breadcrumb} label="位置" />}
      className={styles.shell}
      closeNavigationLabel="关闭导航"
      collapseLabel="收起侧栏"
      expandLabel="展开侧栏"
      footer={<ThemeSwitch />}
      openNavigationLabel="打开导航"
      skipToContentLabel="跳到主要内容"
      storageKey="novel.console"
      brand={
        <ConsoleBrand
          href={href.home()}
          label="小说训练台"
          logo={<Qwen.Avatar size={28} />}
          title="小说训练台"
        />
      }
      navigation={
        <ConsoleNav
          items={[{ href: href.home(), icon: LayoutGrid, label: '全部项目', end: true }]}
          label="项目导航"
          pathname={pathname}
          storageKey="novel.nav"
          groups={
            projects.length
              ? [
                  {
                    key: 'projects',
                    label: '项目',
                    items: projects.map((project) => ({
                      href: href.project(project.id),
                      icon:
                        project.id === activeProject ? (
                          <Icon aria-hidden spin icon={LoaderCircle} size={18} />
                        ) : (
                          BookOpen
                        ),
                      label: project.name,
                    })),
                  },
                ]
              : []
          }
        />
      }
      tools={
        <Flexbox horizontal align="center" gap={8}>
          <ActiveTag state={state.data} />
          <GpuTag state={state.data} />
        </Flexbox>
      }
    >
      <main className={styles.main} id="main">
        {route.page === 'home' ? <HomePage state={state} /> : null}
        {route.page === 'project' ? (
          <ProjectPage
            key={route.projectId}
            projectId={route.projectId}
            tab={route.tab}
            onProjectsChange={state.refresh}
          />
        ) : null}
        {route.page === 'run' ? (
          <RunPage
            key={`${route.projectId}/${route.runId}`}
            projectId={route.projectId}
            runId={route.runId}
            onChange={state.refresh}
          />
        ) : null}
      </main>
    </ConsoleShell>
  );
}
