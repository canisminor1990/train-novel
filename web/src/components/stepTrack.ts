import { createStaticStyles } from '@lobehub/ui';

const styles = createStaticStyles(({ css, cssVar }) => ({
  root: css`
    && {
      overflow: visible;
      gap: 0;
      align-items: flex-start;
    }

    & > li {
      position: relative;

      overflow: visible;
      flex: 1 1 0;
      flex-direction: column;
      gap: 8px;
      align-items: center;

      min-width: 0;
    }

    & > li > span:first-child {
      z-index: 1;
      width: 28px;
      height: 28px;
    }

    & > li > div {
      align-items: center;
      text-align: center;
    }

    & > li > div > div:first-child {
      text-align: center;
      white-space: normal;
    }

    & > li > span[aria-hidden='true'] {
      position: absolute;
      z-index: 0;
      inset-block-start: 13px;
      inset-inline-start: calc(50% + 14px);

      flex: none;

      width: calc(100% - 26px);
      min-width: 0;
      height: 2px;
    }

    & > li[data-status='finish'] > span:first-child {
      border-color: ${cssVar.colorSuccess};
      color: ${cssVar.colorBgContainer};
      background: ${cssVar.colorSuccess};
    }

    & > li[data-status='finish'] > span[aria-hidden='true'] {
      background: ${cssVar.colorSuccess};
    }

    & > li[data-status='process'] > span:first-child {
      border-color: ${cssVar.colorPrimary};
      color: ${cssVar.colorPrimary};
      background: transparent;
    }

    & > li[data-status='error'] > span:first-child {
      border-color: ${cssVar.colorError};
      color: ${cssVar.colorError};
      background: transparent;
    }

    & > li:has(+ li[data-status='error']) > span[aria-hidden='true'] {
      background: ${cssVar.colorError};
    }
  `,
}));

export const stepTrackClass = styles.root;
