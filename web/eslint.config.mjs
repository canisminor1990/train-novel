import { fileURLToPath } from 'node:url';

import { eslint } from '@lobehub/lint';
import { restrictedImports } from '@lobehub/ui/eslint';

const tsconfigRootDir = fileURLToPath(new URL('.', import.meta.url));

// lobe-design C-01, straight from @lobehub/ui (no local exceptions).
const [, uiRestrictedImportOptions] = restrictedImports.rules['no-restricted-imports'];

export default eslint(
  {
    ignores: ['node_modules', 'dist'],
    react: 'vite',
  },
  {
    languageOptions: {
      parserOptions: {
        tsconfigRootDir,
      },
    },
  },
  {
    rules: {
      'no-restricted-imports': ['error', uiRestrictedImportOptions],
    },
  },
  // Same global overrides as lobe-admin.
  {
    rules: {
      '@eslint-react/hooks-extra/no-direct-set-state-in-use-effect': 0,
      '@eslint-react/jsx-key-before-spread': 0,
      '@eslint-react/naming-convention/ref-name': 0,
      '@eslint-react/naming-convention/use-state': 0,
      '@eslint-react/no-array-index-key': 0,
      '@typescript-eslint/no-use-before-define': 0,
      'no-extra-boolean-cast': 0,
      'no-restricted-syntax': 0,
      'react-refresh/only-export-components': 0,
      'regexp/match-any': 0,
      'unicorn/better-regex': 0,
      'unicorn/number-literal-case': 0,
      'unicorn/catch-error-name': 0,
      'unicorn/explicit-length-check': 0,
      'unicorn/no-array-for-each': 0,
      'unicorn/no-negated-condition': 0,
      'unicorn/no-null': 0,
      'unicorn/no-typeof-undefined': 0,
      'unicorn/no-useless-undefined': 0,
      'unicorn/prefer-code-point': 0,
      'unicorn/prefer-logical-operator-over-ternary': 0,
      'unicorn/prefer-number-properties': 0,
      'unicorn/prefer-spread': 0,
      'unicorn/prefer-ternary': 0,
      'unicorn/prefer-type-error': 0,
    },
  },
  {
    files: ['**/*.{ts,tsx}'],
    rules: {
      '@typescript-eslint/consistent-type-imports': [2, { fixStyle: 'separate-type-imports' }],
    },
  },
);
