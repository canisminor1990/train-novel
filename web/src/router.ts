import { useEffect, useState } from 'react';

export type ProjectTab = 'train' | 'data' | 'runs' | 'try';

export type Route =
  | { page: 'home' }
  | { page: 'project'; projectId: string; tab: ProjectTab }
  | { page: 'run'; projectId: string; runId: string };

const TABS: ProjectTab[] = ['train', 'data', 'runs', 'try'];

export function parseHash(hash: string): Route {
  const parts = hash.replace(/^#\/?/, '').split('/').filter(Boolean).map(decodeURIComponent);
  if (parts[0] === 'p' && parts[1]) {
    if (parts[2] === 'r' && parts[3]) return { page: 'run', projectId: parts[1], runId: parts[3] };
    const tab = TABS.includes(parts[2] as ProjectTab) ? (parts[2] as ProjectTab) : 'train';
    return { page: 'project', projectId: parts[1], tab };
  }
  return { page: 'home' };
}

export const href = {
  home: () => '#/',
  project: (id: string, tab: ProjectTab = 'train') =>
    tab === 'train' ? `#/p/${id}` : `#/p/${id}/${tab}`,
  run: (id: string, run: string) => `#/p/${id}/r/${run}`,
};

export function go(target: string) {
  window.location.hash = target.replace(/^#/, '');
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseHash(window.location.hash));
  useEffect(() => {
    const onHash = () => setRoute(parseHash(window.location.hash));
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  return route;
}
