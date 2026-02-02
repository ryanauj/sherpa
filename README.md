# Sherpa: Collaborative Learning Routes & Guides

<img width="1408" height="768" alt="IMG_1570" src="https://github.com/user-attachments/assets/dc23d51e-ab72-4b9b-82d1-e9eafbc2a1da" />

**Sherpa** helps you improve at your own pace - with or without AI interaction.

Welcome to **Sherpa** - a repository for hosting static learning routes in markdown format. Think of learning as climbing a mountain: routes are the paths to follow, and sherpas (AI assistants) help guide you along the way.

## Vision

This repository aims to:
- Document efficient learning paths and processes
- Enable iteration and improvement of routes over time
- Provide resources that both humans and AI can effectively use
- Foster a community of learners sharing best practices

## Structure

```
sherpa/
├── routes/             # Learning routes (topic containers)
│   ├── git-basics/
│   │   ├── map.md      # Route overview/syllabus
│   │   ├── sherpa.md   # AI teaching script
│   │   └── guide.md    # Human learning content
│   └── ...
├── tools/              # Reusable scripts, visualizations, quizzes
└── techniques/         # Templates, patterns, best practices
```

## Route Structure

Each route is a complete learning path organized in a subdirectory with three files:

### map.md - Route Overview
The high-level "map" of the learning journey:
- Learning objectives
- Prerequisites
- Route structure with timing
- Available learning modes
- Related resources

### sherpa.md - AI Teaching Script
For AI assistants acting as guides:
- Structured teaching flow
- Verification questions
- Common misconceptions
- Adaptive strategies
- Exercise guidance

### guide.md - Human Learning Content
For humans learning independently:
- Clear explanations
- Detailed examples
- Hands-on exercises
- Self-check points
- Practice projects

## Three Learning Modes

### Mode 1: Self-Paced
Read guide.md independently at your own pace.

### Mode 2: AI-Guided
Learn through conversation with an AI assistant using sherpa.md as their teaching script.

### Mode 3: Collaborative  
Read guide.md while an AI assistant helps using sherpa.md for guidance.

## Tools & Techniques

- **Tools**: Reusable scripts, visualizations, quizzes, and interactive resources
- **Techniques**: Templates, patterns, best practices for creating content, and communication tips for effective AI-human collaboration

These resources can be referenced from any route.

### Communication Tips

Routes, sherpas, and guides can include tips and prompts to help AI assistants and humans communicate more effectively:

- **For humans**: Prompts to ask when stuck, ways to request different explanations, how to pace learning
- **For AI**: Strategies to check understanding, adapt teaching style, provide appropriate hints
- **For both**: Best practices for collaborative learning, effective question formats, feedback mechanisms

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Creating new routes with map, sherpa, and guide files
- Adding tools and techniques
- Testing content with learners
- Maintaining quality and consistency

## Creating a New Route

1. Create a subdirectory: `routes/your-topic/`
2. Use templates from `techniques/`:
   - `map-template-v1.md` → `map.md`
   - `sherpa-template-v1.md` → `sherpa.md`
   - `guide-template-v1.md` → `guide.md`
3. Fill in all three files with aligned content
4. Reference tools and techniques as needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
