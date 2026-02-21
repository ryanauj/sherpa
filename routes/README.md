# Routes

This directory contains learning routes - each route is a complete learning path for a specific topic.

## Structure

Each route is organized as a subdirectory containing three key files:

```
routes/
├── git-basics/
│   ├── map.md       # High-level route overview/syllabus
│   ├── sherpa.md    # AI assistant teaching script
│   └── guide.md     # Human-focused learning content
├── python-basics/
└── ...
```

## The Three Files

### map.md - Route Overview
The "map" provides a high-level view of the learning route:
- Learning objectives
- Prerequisites
- Route structure (sections and timing)
- Learning modes available
- Related tools and techniques

Think of this as the syllabus or trail map that both humans and AI can reference.

### sherpa.md - AI Teaching Script
The "sherpa" is for AI assistants acting as guides:
- Structured teaching flow with timing
- Verification questions to assess understanding
- Common misconceptions to address
- Adaptive strategies for different learners
- Exercise guidance and hints

### guide.md - Human Learning Content
The "guide" is for humans learning independently:
- Clear explanations with examples
- Detailed code walkthroughs
- Hands-on exercises with solutions
- Self-check points
- Practice projects

## Learning Modes

Each route supports three modes:

1. **Self-Paced**: Read guide.md independently
2. **AI-Guided**: Learn through conversation with AI using sherpa.md
3. **Collaborative**: Read guide.md while AI assists using sherpa.md

## Creating a New Route

1. Create a new subdirectory: `routes/your-topic/`
2. Use the templates in `/techniques/` as starting points
3. Create all three files: `map.md`, `sherpa.md`, `guide.md`
4. Reference reusable content from `/tools/` and `/techniques/`

See `/CONTRIBUTING.md` for detailed guidelines.
