import { ActionIcon, Button, Flexbox, setLobeTheme, useThemeMode } from '@lobehub/ui';
import { useConsoleShell } from '@lobehub/ui/dashboard';
import { Moon, Sun } from 'lucide-react';

const COOKIE = 'novel-theme';

function storeTheme(appearance: 'light' | 'dark') {
  document.cookie = `${COOKIE}=${appearance}; Path=/; Max-Age=31536000; SameSite=Lax`;
  document.documentElement.style.colorScheme = appearance;
  setLobeTheme({ appearance });
}

export function ThemeSwitch() {
  const { collapsed } = useConsoleShell();
  const { isDarkMode } = useThemeMode();
  const label = isDarkMode ? '切换为亮色' : '切换为暗色';
  const icon = isDarkMode ? Sun : Moon;
  const toggle = () => storeTheme(isDarkMode ? 'light' : 'dark');

  if (collapsed) {
    return (
      <Flexbox align="center">
        <ActionIcon aria-label={label} icon={icon} size="small" title={label} onClick={toggle} />
      </Flexbox>
    );
  }

  return (
    <Button block icon={icon} type="text" onClick={toggle}>
      {label}
    </Button>
  );
}
