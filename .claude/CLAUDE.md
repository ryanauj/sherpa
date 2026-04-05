# Sherpa — Project Instructions

Sherpa is a markdown-based learning system. Learners pick a topic ("route"), and an AI assistant ("sherpa") guides them through it using structured teaching scripts.

## Core Concepts

- **Route**: A complete learning path for a topic, stored as a subdirectory under `routes/`
- **Three files per route**:
  - `map.md` — Syllabus/overview. What the route covers, prerequisites, structure, learning modes.
  - `sherpa.md` — AI teaching script. How an AI should teach the topic: flow, questions, misconceptions, adaptive strategies.
  - `guide.md` — Human learning content. Self-study material with explanations, examples, exercises, and solutions.
- **Three learning modes**: Self-paced (guide only), AI-guided (sherpa drives), Collaborative (human reads guide while AI assists via sherpa)
- **Ascent**: A guided project that spans multiple routes. Learners build a complete application by progressing through route checkpoints, applying each route's skills to a single evolving project. Stored as a subdirectory under `ascents/` with a single markdown file.
- **Atlas**: A domain-level reference that indexes methods, concepts, and techniques across a family of related routes. Stored as a subdirectory under `atlases/` with a single `atlas.md` file. Use an atlas to look up what a method does, when to use it, and which route teaches it.

## Repository Structure

```
sherpa/
├── routes/              # Learning routes, one subdirectory per topic
│   └── <topic>/
│       ├── map.md
│       ├── sherpa.md
│       └── guide.md
├── ascents/             # Cross-route guided projects (ascents)
│   └── <ascent-name>/
│       └── ascent.md
├── atlases/             # Domain-level reference indexes
│   └── <domain>/
│       └── atlas.md
├── tools/               # Reusable scripts, visualizations, quizzes
├── techniques/          # Templates and patterns for creating routes
│   ├── map-template-v1.md
│   ├── sherpa-template-v1.md
│   ├── guide-template-v1.md
│   ├── ascent-template-v1.md
│   └── atlas-template-v1.md
├── CONTRIBUTING.md
└── README.md
```

## Creating a Route

1. Create `routes/<topic>/` directory
2. Copy templates from `techniques/` as starting points
3. Fill in all three files — they must be aligned (same objectives, compatible examples, cross-referenced)
4. Route names should be lowercase-kebab-case describing the topic (e.g., `tmux-basics`, `git-branching`)

## Content Conventions

### map.md
- YAML frontmatter with `title`, `topics`, `related_routes`
- No time estimates or difficulty ratings (these are subjective)
- Clear prerequisites marked as Required or Helpful
- Lists learning modes and references to tools/techniques

### sherpa.md
- YAML frontmatter linking to paired map and guide
- Written as instructions for an AI assistant (imperative: "Ask...", "Explain...")
- Structured teaching flow broken into sections
- Each section has: core concept, how to explain, examples, verification questions, common misconceptions, hints for struggling learners
- Includes assessment strategies (multiple choice + explanation questions)
- Adaptive strategies for different learner levels and styles

### guide.md
- YAML frontmatter linking to paired map and sherpa
- Written for a human reader (second person, conversational)
- Complete, runnable code examples with expected output
- Exercises with progressive hints in collapsible `<details>` blocks
- Self-check checkpoints between sections
- Practice project that integrates all concepts

### ascent.md
- YAML frontmatter with `title` and `routes` (list of route names the ascent spans)
- Describes a complete application the learner builds across multiple routes
- Organized into checkpoints, each tied to a specific route
- Each checkpoint explains what to apply from the route to the project
- Milestones describe the state of the project after each checkpoint

### atlas.md
- YAML frontmatter with `title`, `domain`, `description`, `routes` (list of indexed routes), `last_updated`
- Entries grouped under H2 category headings
- Each entry is an H3 with: 1-3 sentence description, "Use when" guidance, optional "Watch out" for pitfalls, "Routes" with links to specific route sections
- Written for dual audience: human scanning and AI agent lookup
- Keep entries concise — the atlas is a reference index, not a textbook

## Creating an Ascent

1. Create `ascents/<ascent-name>/` directory
2. Copy `techniques/ascent-template-v1.md` to `ascents/<ascent-name>/ascent.md`
3. Fill in the template — checkpoints must reference existing routes
4. Ascent names should be lowercase-kebab-case (e.g., `my-first-ios-app`)

## Creating an Atlas

1. Create `atlases/<domain>/` directory
2. Copy `techniques/atlas-template-v1.md` to `atlases/<domain>/atlas.md`
3. Fill in entries — each entry needs: description, "Use when" guidance, and route links
4. Atlas names should be lowercase-kebab-case matching the domain (e.g., `statistics`, `ios-development`)

## Quality Checks for Routes

All three files in a route must:
- Have consistent learning objectives
- Cover the same concepts
- Use compatible examples
- Cross-reference each other correctly in frontmatter
- Be self-contained enough for their intended audience (AI or human)

## Session Tracking

Learning sessions are logged locally in `.sessions/` (gitignored). When teaching a route via the sherpa script:

1. **Before starting**: Check `.sessions/index.md` and the route's session directory (e.g., `.sessions/tmux-basics/`) for prior session history. Use this to understand what the learner has already covered and where they left off.
2. **After completing a session**: Create a summary file at `.sessions/<route>/<date>.md` covering what was taught, issues encountered, learner confidence level, and any route changes made. Update `.sessions/index.md` with a link to the new summary.

The `.sessions/` directory structure:
```
.sessions/
├── index.md                    # Index of all sessions, links to summaries
└── <route-name>/
    └── YYYY-MM-DD.md           # Session summary
```

## What NOT to Do

- No time estimates in route sections
- No difficulty ratings
- No subjective assessments (keep content factual and practical)
- Don't create tools or techniques directories/files unless actually building reusable resources
