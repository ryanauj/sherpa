/* Sherpa — lightweight GitHub Pages viewer */

const BASE = '..'; // markdown files are one level up from docs/

const CONTENT_TREE = {
  'Getting Started': {
    type: 'files',
    items: [
      { label: 'README', path: 'README.md' },
      { label: 'Contributing', path: 'CONTRIBUTING.md' },
    ],
  },
  'Routes — iOS Development': {
    type: 'routes',
    items: [
      'xcode-essentials',
      'swift-for-developers',
      'swiftui-fundamentals',
      'uikit-essentials',
      'ios-app-patterns',
      'ios-data-persistence',
      'cloudkit-integration',
      'app-store-publishing',
      'ios-ci-cd-with-github-actions',
    ],
  },
  'Routes — Math & ML': {
    type: 'routes',
    items: [
      'linear-algebra-essentials',
      'linear-algebra-deep-dive',
      'calculus-for-ml',
      'probability-fundamentals',
      'probability-distributions',
      'bayesian-statistics',
      'stats-fundamentals',
      'statistical-inference',
      'stochastic-processes',
      'regression-and-modeling',
      'neural-network-foundations',
      'training-and-backprop',
      'llm-foundations',
    ],
  },
  'Routes — Developer Tools': {
    type: 'routes',
    items: [
      'git-basics',
      'tmux-basics',
      'docker-dev-environments',
      'nix-dev-environments',
      'mise-basics',
      'agent-sandboxing',
    ],
  },
  'Ascents': {
    type: 'ascents',
    items: [
      'my-first-ios-app',
      'neural-net-from-scratch',
    ],
  },
  'Techniques': {
    type: 'files',
    items: [
      { label: 'Overview', path: 'techniques/README.md' },
      { label: 'Map Template', path: 'techniques/map-template-v1.md' },
      { label: 'Sherpa Template', path: 'techniques/sherpa-template-v1.md' },
      { label: 'Guide Template', path: 'techniques/guide-template-v1.md' },
      { label: 'Ascent Template', path: 'techniques/ascent-template-v1.md' },
    ],
  },
};

const ROUTE_FILES = [
  { label: 'Map', file: 'map.md' },
  { label: 'Sherpa', file: 'sherpa.md' },
  { label: 'Guide', file: 'guide.md' },
];

// --- Markdown rendering ---

marked.setOptions({
  highlight: function (code, lang) {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value;
    }
    return hljs.highlightAuto(code).value;
  },
  breaks: false,
  gfm: true,
});

// --- Navigation ---

function buildNav() {
  const tree = document.getElementById('nav-tree');
  let html = '';

  for (const [section, config] of Object.entries(CONTENT_TREE)) {
    html += `<h2>${section}</h2>`;

    if (config.type === 'files') {
      html += '<div class="nav-group">';
      for (const item of config.items) {
        html += `<div class="nav-files open"><a href="#" data-path="${item.path}">${item.label}</a></div>`;
      }
      html += '</div>';
    } else if (config.type === 'routes') {
      for (const route of config.items) {
        const display = route.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
          .replace(/\bMl\b/g, 'ML').replace(/\bIos\b/g, 'iOS')
          .replace(/\bUikit\b/g, 'UIKit').replace(/\bSwiftui\b/g, 'SwiftUI')
          .replace(/\bLlm\b/g, 'LLM').replace(/\bCi Cd\b/g, 'CI/CD')
          .replace(/\bCloudkit\b/g, 'CloudKit').replace(/\bNix\b/g, 'Nix')
          .replace(/\bTmux\b/g, 'tmux').replace(/\bGit\b/g, 'Git');
        html += '<div class="nav-group">';
        html += `<div class="nav-route" data-route="${route}">${display}</div>`;
        html += `<div class="nav-files" data-route-files="${route}">`;
        for (const f of ROUTE_FILES) {
          html += `<a href="#" data-path="routes/${route}/${f.file}">${f.label}</a>`;
        }
        html += '</div></div>';
      }
    } else if (config.type === 'ascents') {
      for (const ascent of config.items) {
        const display = ascent.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
          .replace(/\bIos\b/g, 'iOS');
        html += `<div class="nav-group"><div class="nav-files open">`;
        html += `<a href="#" data-path="ascents/${ascent}/ascent.md">${display}</a>`;
        html += '</div></div>';
      }
    }
  }

  tree.innerHTML = html;

  // Route expand/collapse
  tree.querySelectorAll('.nav-route').forEach(el => {
    el.addEventListener('click', () => {
      const route = el.dataset.route;
      const files = tree.querySelector(`[data-route-files="${route}"]`);
      const wasOpen = files.classList.contains('open');
      // collapse all
      tree.querySelectorAll('.nav-files[data-route-files]').forEach(f => f.classList.remove('open'));
      tree.querySelectorAll('.nav-route').forEach(r => r.classList.remove('open'));
      if (!wasOpen) {
        files.classList.add('open');
        el.classList.add('open');
      }
    });
  });

  // File links
  tree.querySelectorAll('a[data-path]').forEach(el => {
    el.addEventListener('click', e => {
      e.preventDefault();
      loadPage(el.dataset.path);
    });
  });

  // Header link
  document.querySelector('.sidebar-header a').addEventListener('click', e => {
    e.preventDefault();
    loadPage('README.md');
  });
}

// --- Content loading ---

async function loadPage(path) {
  const body = document.getElementById('markdown-body');
  body.innerHTML = '<p class="loading">Loading…</p>';

  // Update active state
  document.querySelectorAll('.nav-files a').forEach(a => a.classList.remove('active'));
  const active = document.querySelector(`a[data-path="${path}"]`);
  if (active) {
    active.classList.add('active');
    // Auto-expand parent route
    const parent = active.closest('.nav-files[data-route-files]');
    if (parent) {
      parent.classList.add('open');
      const route = parent.dataset.routeFiles;
      const routeEl = document.querySelector(`.nav-route[data-route="${route}"]`);
      if (routeEl) routeEl.classList.add('open');
    }
  }

  // Update hash
  window.location.hash = path;

  // Close mobile sidebar
  document.getElementById('sidebar').classList.remove('open');

  try {
    const res = await fetch(`${BASE}/${path}`);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const md = await res.text();
    // Strip YAML frontmatter
    const cleaned = md.replace(/^---[\s\S]*?---\n*/, '');
    body.innerHTML = marked.parse(cleaned);
  } catch (err) {
    body.innerHTML = `<p class="loading">Could not load <code>${path}</code>: ${err.message}</p>`;
  }
}

// --- Init ---

buildNav();

// Sidebar toggle
document.getElementById('sidebar-toggle').addEventListener('click', () => {
  document.getElementById('sidebar').classList.toggle('open');
});

// Load from hash or default
const initial = window.location.hash.slice(1) || 'README.md';
loadPage(initial);

// Handle back/forward
window.addEventListener('hashchange', () => {
  const path = window.location.hash.slice(1);
  if (path) loadPage(path);
});
