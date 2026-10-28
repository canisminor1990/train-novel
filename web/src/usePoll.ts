import { useCallback, useEffect, useRef, useState } from 'react';

export interface Poll<T> {
  data: T | null;
  error: Error | null;
  loading: boolean;
  refresh: () => Promise<void>;
  updated: number | null;
}

/** Polls while the tab is visible. Keeps the last good data when a refresh fails. */
export function usePoll<T>(
  load: (() => Promise<T>) | null,
  interval: number,
  key: string,
): Poll<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [updated, setUpdated] = useState<number | null>(null);
  const loadRef = useRef(load);
  loadRef.current = load;
  const seq = useRef(0);

  const refresh = useCallback(async () => {
    const current = loadRef.current;
    if (!current) return;
    const ticket = ++seq.current;
    try {
      const next = await current();
      if (ticket !== seq.current) return;
      setData(next);
      setError(null);
      setUpdated(Date.now());
    } catch (exc) {
      if (ticket !== seq.current) return;
      setError(exc instanceof Error ? exc : new Error('读取失败'));
    }
  }, []);

  useEffect(() => {
    setData(null);
    setError(null);
    setUpdated(null);
    if (!loadRef.current) return;
    void refresh();
    const timer = window.setInterval(() => {
      if (!document.hidden) void refresh();
    }, interval);
    const onVisible = () => {
      if (!document.hidden) void refresh();
    };
    document.addEventListener('visibilitychange', onVisible);
    return () => {
      seq.current += 1;
      window.clearInterval(timer);
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, [interval, key, refresh]);

  return { data, error, loading: data === null && error === null, updated, refresh };
}
