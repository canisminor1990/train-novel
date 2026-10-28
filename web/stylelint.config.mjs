import { stylelint } from '@lobehub/lint';

export default {
  ...stylelint,
  rules: {
    ...stylelint.rules,
    'declaration-property-value-keyword-no-deprecated': null,
    'declaration-property-value-no-unknown': null,
    'selector-class-pattern': null,
    'selector-id-pattern': null,
  },
};
