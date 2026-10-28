import { Icon, Tag } from '@lobehub/ui';
import { LoaderCircle } from 'lucide-react';

import type { RunState } from '../api';
import { isLive, RUN_STATE } from '../format';

export function RunStateTag({ state }: { state: RunState }) {
  const meta = RUN_STATE[state] ?? { label: state };
  return (
    <Tag
      color={meta.color}
      icon={isLive(state) ? <Icon aria-hidden spin icon={LoaderCircle} size={12} /> : undefined}
    >
      {meta.label}
    </Tag>
  );
}
