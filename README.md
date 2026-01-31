# Tides: Collaborative Learning Routes & Guides

Welcome to **Tides** - a repository for hosting static learning routes and guides in markdown format. Think of learning as climbing a mountain: there are many paths (routes) to the summit, and experienced guides (AI assistants) can help you navigate them. These resources are designed to be used collaboratively by both AI assistants and humans together, while remaining accessible to either separately.

## Vision

This repository aims to:
- Document efficient learning paths and processes
- Enable iteration and improvement of routes over time
- Provide clear, structured guides that both humans can understand and AI can effectively use
- Foster a community of learners sharing best practices

## Structure

```
tides/
├── routes/             # Human-readable learning materials (the paths up the mountain)
│   ├── programming/    # Programming routes
│   ├── tools/          # Tool-specific routes
│   └── concepts/       # Conceptual routes
├── guides/             # AI assistant teaching scripts (the sherpa's knowledge)
│   ├── programming/    # Programming teaching guides
│   ├── tools/          # Tool-specific teaching guides
│   └── concepts/       # Conceptual teaching guides
└── templates/          # Templates for creating new content
```

## The Dual Structure

**Routes** and **Guides** are designed to work together as paired content:

- **Routes** are human-readable documents that learners can read and work through at their own pace. They contain explanations, examples, and exercises - the documented path up the mountain.
- **Guides** are AI assistant teaching scripts that provide a structured approach for AI to tutor a human through the same material. They include topics to cover, questions to ask, and ways to verify understanding - the sherpa's knowledge for helping climbers.

Each topic should have both a route (for humans) and a guide (for AI assistants), aligned in structure and content.

## Three Learning Modes

### Mode 1: Self-Paced (Human Only)
The learner reads the route independently without AI assistance.
- Browse the `routes/` directory
- Choose a topic and follow the content
- Complete exercises and practice at your own pace

### Mode 2: AI-Guided (AI + Human Interactive)
The AI assistant uses the guide to teach, without the human reading the route directly.
- AI references the guide in `guides/` directory
- AI covers topics in order, asks verification questions
- Human learns through conversation and exercises
- AI adapts based on human's responses

### Mode 3: Collaborative (AI + Human + Route)
The human reads the route while the AI assists using the guide.
- Human works through the route
- AI follows along using the paired guide
- Human asks questions, AI provides clarification
- AI asks verification questions from the guide
- Both work together through exercises

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Creating paired routes and guides
- Structuring content for self-paced learning (routes)
- Structuring teaching scripts for AI assistants (guides)
- Ensuring routes and guides align with each other
- Testing content with learners

## Creating Content

### Creating a Route (Human-Focused)
Use the template in `templates/route-template.md` to create content that humans can read independently. Key elements:
- **Clear explanations**: Concepts explained in readable prose
- **Examples**: Practical, runnable code with explanations
- **Exercises**: Hands-on practice with solutions
- **Self-contained**: Learner can understand without external help

### Creating a Guide (AI-Focused)
Use the template in `templates/guide-template.md` to create teaching scripts for AI assistants. Key elements:
- **Teaching flow**: Structured progression of topics
- **Questions to ask**: Verification questions at checkpoints
- **Common misconceptions**: What to watch for and clarify
- **Adaptive strategies**: How to help if learner struggles
- **Exercise guidance**: How to present and help with exercises

### Pairing Content
Each route should have a corresponding guide with the same filename in the parallel directory structure. For example:
- `routes/tools/git-basics.md` pairs with `guides/tools/git-basics.md`
- Both cover the same material, but optimized for different modes of learning

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
