import { createStaticStyles, cx, Text } from '@lobehub/ui';
import { type KeyboardEvent, useRef } from 'react';

import type { Preset } from '../api';
import { formatCount } from '../format';

const styles = createStaticStyles(({ css, cssVar }) => ({
  grid: css`
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;

    @media (width <= 800px) {
      grid-template-columns: 1fr;
    }
  `,
  card: css`
    cursor: pointer;

    display: flex;
    flex-direction: column;
    gap: 6px;
    align-items: flex-start;

    padding: 16px;
    border: 1px solid ${cssVar.colorBorderSecondary};
    border-radius: ${cssVar.borderRadiusLG};

    color: ${cssVar.colorText};
    text-align: start;

    background: ${cssVar.colorBgContainer};

    transition: border-color 0.15s ease;

    &:focus-visible {
      outline: 2px solid ${cssVar.colorPrimaryBorder};
      outline-offset: 2px;
    }

    &:disabled {
      cursor: not-allowed;
      opacity: 0.6;
    }

    &:hover:not(:disabled) {
      border-color: ${cssVar.colorBorder};
    }
  `,
  selected: css`
    border-color: ${cssVar.colorPrimary};
    background: ${cssVar.colorPrimaryBg};

    &:hover:not(:disabled) {
      border-color: ${cssVar.colorPrimary};
    }
  `,
  label: css`
    font-size: 15px;
    font-weight: 600;
  `,
  facts: css`
    margin-block-start: auto;
    padding-block-start: 6px;
    font-size: 12px;
    color: ${cssVar.colorTextTertiary};
  `,
}));

function facts(preset: Preset) {
  const parts = [
    preset.max_steps ? `最多 ${preset.max_steps} 步` : `${preset.epochs} 轮`,
    `序列 ${preset.seq}`,
    `rank ${preset.rank}`,
  ];
  if (preset.steps) parts.push(`约 ${formatCount(preset.steps)} 步`);
  if (preset.early_stop) parts.push('验证 loss 回升会提前停');
  return parts.join(' · ');
}

export function PresetPicker({
  presets,
  disabled,
  onSelect,
}: {
  presets: Preset[];
  disabled?: boolean;
  onSelect: (id: string) => void;
}) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const selectedIndex = Math.max(
    0,
    presets.findIndex((preset) => preset.selected),
  );

  function onKeyDown(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    const delta =
      event.key === 'ArrowRight' || event.key === 'ArrowDown'
        ? 1
        : event.key === 'ArrowLeft' || event.key === 'ArrowUp'
          ? -1
          : 0;
    if (!delta) return;
    event.preventDefault();
    const next = (index + delta + presets.length) % presets.length;
    refs.current[next]?.focus();
    onSelect(presets[next].id);
  }

  return (
    <div
      aria-busy={disabled}
      aria-label="训练预设"
      className={styles.grid}
      role="radiogroup"
      title={disabled ? '正在保存预设' : undefined}
    >
      {presets.map((preset, index) => (
        <button
          aria-checked={preset.selected}
          className={cx(styles.card, preset.selected && styles.selected)}
          disabled={disabled}
          key={preset.id}
          role="radio"
          tabIndex={index === selectedIndex ? 0 : -1}
          type="button"
          ref={(node) => {
            refs.current[index] = node;
          }}
          onClick={() => !preset.selected && onSelect(preset.id)}
          onKeyDown={(event) => onKeyDown(event, index)}
        >
          <span className={styles.label}>{preset.label}</span>
          <Text type="secondary">{preset.summary}</Text>
          <span className={styles.facts}>{facts(preset)}</span>
        </button>
      ))}
    </div>
  );
}
