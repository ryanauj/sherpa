# Sherpa — Project Instructions

Sherpa is a markdown-based learning system. Learners pick a topic ("route"), and an AI assistant ("sherpa") guides them through it using structured teaching scripts.

## Core Concepts

- **Route**: A complete learning path for a topic, stored as a subdirectory under `routes/`
- **Three files per route**:
  - `map.md` — Syllabus/overview. What the route covers, prerequisites, structure, learning modes.
  - `sherpa.md` — AI teaching script. How an AI should teach the topic: flow, questions, misconceptions, adaptive strategies.
  - `guide.md` — Human learning content. Self-study material with explanations, examples, exercises, and solutions.
- **Three learning modes**: Self-paced (guide only), AI-guided (sherpa drives), Collaborative (human reads guide while AI assists via sherpa)

## Repository Structure

```
sherpa/
├── routes/              # Learning routes, one subdirectory per topic
│   └── <topic>/
│       ├── map.md
│       ├── sherpa.md
│       └── guide.md
├── tools/               # Reusable scripts, visualizations, quizzes
├── techniques/          # Templates and patterns for creating routes
│   ├── map-template-v1.md
│   ├── sherpa-template-v1.md
│   └── guide-template-v1.md
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

## Quality Checks for Routes

All three files in a route must:
- Have consistent learning objectives
- Cover the same concepts
- Use compatible examples
- Cross-reference each other correctly in frontmatter
- Be self-contained enough for their intended audience (AI or human)

## What NOT to Do

- No time estimates in route sections
- No difficulty ratings
- No subjective assessments (keep content factual and practical)
- Don't create tools or techniques directories/files unless actually building reusable resources
