import '@lobehub/ui/theme.css';
import '@lobehub/ui/style.css';
import '@lobehub/ui/global.css';

import {
  ConfigProvider,
  createGlobalStyle,
  createStaticStyles,
  ModalHost,
  ToastHost,
} from '@lobehub/ui';
import { zhCn } from '@lobehub/ui/i18n';
import { motion } from 'motion/react';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import { App } from './App';

const styles = createStaticStyles(({ css }) => ({
  app: css`
    display: flex;
    flex-direction: column;

    height: 100%;
    min-height: 100dvh;
    max-height: 100dvh;

    @media (device-width >= 576px) {
      overflow: hidden;
    }
  `,
}));

const GlobalStyle = createGlobalStyle`
  html,
  body,
  #root {
    height: 100%;
    min-height: 100dvh;
    max-height: 100dvh;

    @media (device-width >= 576px) {
      overflow: hidden;
    }
  }
`;

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <GlobalStyle />
    <ConfigProvider locale="zh-CN" motion={motion} resources={zhCn}>
      <div className={styles.app}>
        <ToastHost />
        <App />
        <ModalHost />
      </div>
    </ConfigProvider>
  </StrictMode>,
);
