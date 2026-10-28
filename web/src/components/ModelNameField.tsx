import { Flexbox, Input, Text } from '@lobehub/ui';
import { useEffect, useRef, useState } from 'react';

import { api, type ModelCheck } from '../api';

export type ModelFieldState = 'empty' | 'checking' | 'ok' | 'error';

export function slugModelName(name: string) {
  const slug = name
    .toLowerCase()
    .replaceAll(/[^\d.a-z]+/g, '-')
    .replaceAll(/^-+|-+$/g, '');
  return slug.length >= 2 ? slug.slice(0, 57) : '';
}

export function ModelNameField({
  emptyHint,
  id,
  projectId,
  value,
  onChange,
  onState,
}: {
  emptyHint: string;
  id: string;
  projectId?: string;
  value: string;
  onChange: (value: string) => void;
  onState: (state: ModelFieldState) => void;
}) {
  const [result, setResult] = useState<ModelCheck | null>(null);
  const trimmed = value.trim();
  const stateRef = useRef(onState);
  stateRef.current = onState;

  useEffect(() => {
    const onState = (state: ModelFieldState) => stateRef.current(state);
    if (!trimmed) {
      setResult(null);
      onState('empty');
      return;
    }
    onState('checking');
    setResult(null);
    let cancelled = false;
    const timer = window.setTimeout(() => {
      api
        .checkModel(trimmed, projectId)
        .then((check) => {
          if (cancelled) return;
          setResult(check);
          onState(check.valid ? 'ok' : 'error');
        })
        .catch((exc: unknown) => {
          if (cancelled) return;
          setResult({
            ok: false,
            valid: false,
            error: exc instanceof Error ? exc.message : '检查失败',
          });
          onState('error');
        });
    }, 300);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [trimmed, projectId]);

  const invalid = Boolean(trimmed && result && !result.valid);
  const hintId = `${id}-hint`;
  let hint = emptyHint;
  let tone: 'secondary' | 'danger' | 'warning' = 'secondary';
  if (trimmed && !result) hint = '正在检查…';
  if (trimmed && result) {
    if (!result.valid) {
      hint = result.error ?? '这个名字不能用';
      tone = 'danger';
    } else if (result.exists) {
      hint = `Ollama 里已经有同名模型，训完会覆盖 ${result.model} 和 ${result.think_model ?? `${result.model}-think`}。`;
      tone = 'warning';
    } else {
      const think = result.think_model ?? `${result.model}-think`;
      hint = `训完会挂上 ${result.model}（正文）和 ${think}（带思考）。`;
    }
  }

  return (
    <Flexbox gap={6}>
      <label htmlFor={id}>Ollama 模型名</label>
      <Input
        aria-describedby={hintId}
        aria-invalid={invalid || undefined}
        id={id}
        maxLength={57}
        placeholder="例如 yehang"
        spellCheck={false}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
      <Text id={hintId} type={tone}>
        {hint}
      </Text>
    </Flexbox>
  );
}
