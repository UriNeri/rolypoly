// Custom Mermaid initialization with ELK layout support for MkDocs Material
// This script will be loaded via extra_javascript in mkdocs.yml
// See: https://mermaid.js.org/config/elk.html

import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10.4.0/dist/mermaid.esm.min.mjs';
import ELK from 'https://cdn.jsdelivr.net/npm/@elkjs/elkjs@0.8.2/lib/elk.bundled.js';
import { mermaidAPI } from 'https://cdn.jsdelivr.net/npm/mermaid@10.4.0/dist/mermaid.esm.min.mjs';

// Enable ELK layout
mermaid.initialize({
  startOnLoad: true,
  theme: 'default',
  flowchart: { layout: 'elk' },
  graph: { layout: 'elk' },
  elk: { elkjs: ELK }
});

window.mermaid = mermaid;
window.mermaidAPI = mermaidAPI;
