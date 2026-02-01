# Contributing to Sherpa

Thank you for your interest in contributing to Sherpa! This document provides guidelines for creating routes with their three-file structure (map, sherpa, guide) that enable flexible AI-human learning.

## Philosophy

Content in Sherpa follows a **route container approach**:
- **Routes**: Complete learning paths organized as subdirectories
- Each route contains three complementary files optimized for different uses

Together, they enable three learning modes:
1. **Self-paced**: Human reads guide.md alone
2. **AI-guided**: AI teaches using sherpa.md
3. **Collaborative**: Human reads guide.md with AI using sherpa.md

## Core Principles

All content should be:
- **Complete**: Each route is a self-contained learning path
- **Triple-file**: Every route has map, sherpa, and guide files
- **Aligned**: All three files cover the same concepts cohesively
- **Practical**: Include real examples and hands-on exercises
- **Iterative**: Designed to be improved over time based on usage

## Route Structure

Each route lives in a subdirectory under `/routes/` with three files:

### map.md - Route Overview
The high-level "map" showing the learning journey:
- Learning objectives
- Prerequisites
- Route structure (sections, no time estimates)
- Learning modes available
- Related routes and resources

**Purpose**: Syllabus that both humans and AI can reference

### sherpa.md - AI Teaching Script
Instructions for AI assistants acting as guides:
- Teaching overview and objectives
- Structured teaching flow by section
- Verification questions (multiple choice and explanation-based)
- Common misconceptions and clarifications
- Adaptive strategies for different learners
- Exercise guidance with progressive hints
- Learner preference configuration options

**Purpose**: Enable AI to teach effectively and adapt to learners

### guide.md - Human Learning Content
Self-contained learning material for humans:
- Clear explanations with examples
- Detailed code walkthroughs
- Hands-on exercises with solutions
- Self-check points
- Practice projects

**Purpose**: Enable self-paced learning without AI assistance

## Creating a New Route

### Step 1: Plan Your Route

Decide what you want to teach:
- What are the learning objectives?
- What prerequisite knowledge is required?
- What related routes exist?
- What tools or techniques will you reference?

### Step 2: Create Route Directory

```bash
mkdir -p routes/your-topic
```

### Step 3: Create the Three Files

Use templates from `/techniques/`:

```bash
cp techniques/map-template.md routes/your-topic/map.md
cp techniques/sherpa-template-v1.md routes/your-topic/sherpa.md
cp techniques/guide-template-v1.md routes/your-topic/guide.md
```

### Step 4: Fill in map.md

1. **Add metadata** (no difficulty or duration):
   ```yaml
   ---
   title: Your Route Title
   topics:
     - Topic 1
     - Topic 2
   related_routes:
     - related-route-1
     - related-route-2
   ---
   ```

2. **Write route structure** (sections without time estimates)
3. **List prerequisites** clearly
4. **Reference tools and techniques** used

### Step 5: Fill in guide.md (Human-Focused)

1. **Add metadata**:
   ```yaml
   ---
   title: Your Route Title
   route_map: /routes/your-topic/map.md
   paired_sherpa: /routes/your-topic/sherpa.md
   prerequisites:
     - Prerequisite 1
   topics:
     - Topic 1
   ---
   ```

2. **Write clear explanations**:
   - Explain concepts in readable prose
   - Use analogies and real-world examples
   - Define technical terms on first use

3. **Include detailed examples**:
   - Provide complete, runnable code
   - Walk through code step-by-step
   - Show expected output

4. **Create exercises**:
   - Clear task descriptions
   - Progressive hints in collapsible sections
   - Complete solutions with explanations

### Step 6: Fill in sherpa.md (AI-Focused)

1. **Add metadata**:
   ```yaml
   ---
   title: Your Route Title
   route_map: /routes/your-topic/map.md
   paired_guide: /routes/your-topic/guide.md
   topics:
     - Topic 1
   ---
   ```

2. **Document learner preferences**:
   - Teaching tone options
   - Assessment format (multiple choice, explanation, mixed)
   - Pacing preferences

3. **Structure the teaching flow**:
   - Break content into clear sections
   - For each section describe:
     - Core concept to teach
     - How to explain it
     - Example to present
     - Verification questions (both types)

4. **Include assessment strategies**:
   - Multiple choice questions for quick checks
   - Explanation questions for deeper understanding
   - Mixed approach recommendations

5. **Add common misconceptions**:
   - What learners often misunderstand
   - How to clarify

6. **Provide adaptive strategies**:
   - How to help struggling learners
   - How to challenge excelling learners

### Step 7: Ensure Alignment

Check that all three files:
- [ ] Have consistent learning objectives
- [ ] Cover the same concepts cohesively
- [ ] Use compatible examples
- [ ] Reference each other correctly
- [ ] Are well-structured and complete

### Step 8: Test Your Route

Before submitting:
1. **Test guide.md independently**: Can someone learn from it alone?
2. **Test with AI using sherpa.md**: Can an AI effectively teach using it?
3. **Test collaborative mode**: Do they work well together?
4. **Get feedback**: Have others review or try your content

## Supporting Resources

### Adding Tools

Tools are reusable scripts, visualizations, quizzes, or executables in `/tools/`.

Create organized subdirectories:
```
tools/
├── git-visualizer/
├── python-practice-quiz/
└── terminal-simulator/
```

Each tool should have clear documentation.

### Adding Techniques

Techniques are templates, patterns, and best practices in `/techniques/`.

- Document new patterns you discover
- Create reusable templates
- Share best practices for teaching/learning
- Include communication tips for AI-human collaboration

## Style Guidelines

### For map.md

- Keep it concise and high-level
- No time estimates (subjective and inaccurate)
- No difficulty ratings (subjective)
- Focus on structure and relationships
- Use clear section headings

### For guide.md (Human-Focused)

**Writing Style:**
- Conversational, encouraging tone
- Second person ("you")
- Clear and concise sentences
- Active voice
- Explain "why" not just "how"

**Code Examples:**
- Complete and runnable
- Well-commented
- Show expected output
- Start simple, build complexity

### For sherpa.md (AI Teaching Scripts)

**Writing Style:**
- Write as instructions for AI
- Imperative tone ("Ask...", "Explain...")
- Be specific about what to say and do
- Include multiple assessment formats

**Content:**
- Break teaching into clear sections (no time estimates)
- Provide specific questions, both multiple choice and explanation
- List multiple ways to explain concepts
- Include configuration options for learner preferences

## Configuration Files

Routes can reference a `.sherpa-config.yml` file (gitignored) where learners specify preferences:

```yaml
teaching:
  tone: objective|encouraging|humorous
  explanation_depth: concise|balanced|detailed
  pacing: learner-led|balanced|structured

assessment:
  quiz_type: multiple_choice|explanation|mixed
  quiz_frequency: after_each_section|after_major_topics|end_of_route
  feedback_style: immediate|summary|detailed
```

Default to objective tone and mixed assessments when no config exists.

## Review Process

### Before Submitting

**Route checklist:**
- [ ] All three files created (map, sherpa, guide)
- [ ] Metadata complete and consistent
- [ ] No difficulty or duration fields
- [ ] No time estimates in sections
- [ ] Prerequisites clearly stated
- [ ] Related routes referenced
- [ ] Working code examples
- [ ] Exercises with solutions
- [ ] Assessment questions (multiple formats)

### Submitting Your Content

1. **Create a pull request** with all three files
2. **In your PR description**, mention:
   - What topic you're covering
   - Who the target audience is
   - How you tested the content
   - Related routes or dependencies

3. **Review process**:
   - Maintainers will review all three files
   - May ask for clarifications or changes
   - Will test with learners if possible
   - Will merge when complete

## Improving Existing Content

When improving existing routes:
- Maintain alignment across all three files
- Update related files together
- Document changes in your PR
- Explain why the change improves learning
- Add communication tips if you discover good patterns

## Questions?

If you need help:
- Check templates in `/techniques/`
- Look at examples in `/routes/`
- Review tools in `/tools/`
- Open an issue to ask questions
- Ask for feedback in PR comments

Thank you for helping make learning more accessible and effective!
