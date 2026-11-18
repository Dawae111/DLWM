import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://dawae111.github.io',
  base: '/DLWM',
  output: 'static',
  trailingSlash: 'always',
  compressHTML: false,
  devToolbar: { enabled: false },
});
