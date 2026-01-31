# Routes - Human-Readable Learning Materials

This directory contains **Routes** designed for humans to read and work through at their own pace. Each Route corresponds to a guide in the `/guides` directory that AI assistants can use for teaching.

## Purpose

Routes are self-contained learning materials with:
- Clear explanations in readable prose
- Practical examples with detailed walkthroughs
- Hands-on exercises with hints and solutions
- Self-check points to validate understanding
- Practice projects to apply concepts

## Structure

- **programming/** - Routes for programming languages (Python, JavaScript, etc.)
- **tools/** - Routes for specific tools and frameworks (Git, Docker, etc.)
- **concepts/** - Routes for theoretical concepts (Algorithms, Design Patterns, etc.)

## How to Use Routes

### Self-Paced Learning (No AI)
1. Choose a Route that matches your learning goals
2. Read through each section carefully
3. Try the examples yourself
4. Complete all exercises
5. Build the practice project
6. Use self-check lists to validate understanding

### With AI Assistance
1. Open the Route you want to learn
2. Tell the AI assistant which Route you're working on
3. Read sections of the Route
4. Ask the AI questions about confusing parts
5. Work through exercises with AI support when needed
6. The AI will follow along using the paired guide

### AI-Led Learning (Without Reading)
1. Tell the AI assistant what you want to learn
2. The AI will use the teaching guide to teach you
3. Learn through conversation and guided exercises
4. The AI may reference the Route for additional context
5. You can always read the Route later for review

## Creating a Route

See the template in `/templates/route-template.md` and refer to `/CONTRIBUTING.md` for detailed guidelines.

## Pairing with Guides

Each route should have a corresponding AI teaching guide with the same filename:
- `routes/tools/git-basics.md` ↔ `guides/tools/git-basics.md`
- Both cover the same material, optimized for different learning modes
