import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { homedir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const candidates = [
  process.env.LOBE_DESIGN,
  path.resolve(here, '../../.agents/skills/lobe-design'),
  path.join(homedir(), '.cursor/skills/lobe-design'),
  path.join(homedir(), '.claude/skills/lobe-design'),
].filter(Boolean);

const root = candidates.find((dir) => existsSync(path.join(dir, 'scripts/lint-lobe.mjs')));
if (!root) {
  console.error('找不到 lobe-design skill。设置 LOBE_DESIGN 指向它的目录后再跑。');
  process.exit(1);
}

const targets = process.argv.slice(2);
const result = spawnSync(
  process.execPath,
  [path.join(root, 'scripts/lint-lobe.mjs'), ...(targets.length ? targets : ['src'])],
  { stdio: 'inherit' },
);
process.exit(result.status ?? 1);
