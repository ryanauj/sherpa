# Contributing to Tides

Thank you for your interest in contributing to Tides! This document provides guidelines for creating paired tutorials and guides that enable effective learning in multiple modes.

## Philosophy

Content in Tides follows a **dual structure**:
- **Tutorials**: Human-readable learning materials
- **Guides**: AI assistant teaching scripts

Together, they enable three learning modes:
1. **Self-paced**: Human reads tutorial alone
2. **AI-guided**: AI teaches using guide without tutorial
3. **Collaborative**: Human reads tutorial with AI assistance using guide

## Core Principles

All content should be:
- **Paired**: Every tutorial has a matching guide (and vice versa)
- **Aligned**: Tutorial and guide cover the same concepts in the same order
- **Optimized**: Each piece is optimized for its specific audience (human or AI)
- **Practical**: Include real examples and hands-on exercises
- **Iterative**: Designed to be improved over time based on usage

## Types of Content

### Tutorials (Human-Focused)

Learning materials designed for humans to read independently. Tutorials should:
- Use clear, readable prose that explains concepts thoroughly
- Include detailed code examples with step-by-step walkthroughs
- Provide exercises with expandable hints and solutions
- Have self-check points to validate understanding
- Be self-contained (learner doesn't need external help to understand)
- Build from simple to complex progressively

**Target audience**: Humans learning a topic by reading

**Located in**: `/tutorials/[category]/[topic-name].md`

### Guides (AI-Focused)

Teaching scripts designed for AI assistants to teach interactively. Guides should:
- Provide structured teaching flow with timing estimates
- Include specific verification questions to ask learners
- List common misconceptions and how to address them
- Offer adaptive strategies for different learner levels
- Give guidance on presenting and helping with exercises
- Suggest how to explain concepts multiple ways

**Target audience**: AI assistants teaching the topic

**Located in**: `/guides/[category]/[topic-name].md`

## Creating Paired Content

### Step 1: Choose Your Topic

Decide what you want to teach:
- What are the learning objectives?
- Who is the target audience (beginner/intermediate/advanced)?
- What prerequisite knowledge is required?
- How long will it take to learn?

### Step 2: Create the Tutorial (Human-Focused)

1. **Copy the template:**
   ```bash
   cp templates/tutorial-template.md tutorials/[category]/[topic-name].md
   ```

2. **Fill in the metadata:**
   ```yaml
   ---
   title: Your Tutorial Title
   difficulty: beginner|intermediate|advanced
   duration: X minutes
   paired_guide: /guides/[category]/[topic-name].md
   prerequisites:
     - Prerequisite 1
     - Prerequisite 2
   topics:
     - Topic 1
     - Topic 2
   ---
   ```

3. **Write clear explanations:**
   - Explain concepts in readable prose
   - Use analogies and real-world examples
   - Define technical terms on first use
   - Break down complex ideas into digestible parts

4. **Include detailed examples:**
   - Provide complete, runnable code
   - Walk through the code step-by-step
   - Show expected output
   - Explain what each part does and why

5. **Create exercises:**
   - Clear task descriptions
   - Progressive hints in collapsible sections
   - Complete solutions with explanations
   - Self-check lists after each section

6. **Add a practice project:**
   - Integrates all concepts learned
   - Clear requirements and getting started steps
   - Example solution in collapsible section

### Step 3: Create the Paired Guide (AI-Focused)

1. **Copy the template:**
   ```bash
   cp templates/guide-template.md guides/[category]/[topic-name].md
   ```

2. **Fill in the metadata:**
   ```yaml
   ---
   title: Your Tutorial Title
   difficulty: beginner|intermediate|advanced
   duration: X minutes
   paired_tutorial: /tutorials/[category]/[topic-name].md
   topics:
     - Topic 1
     - Topic 2
   ---
   ```

3. **Structure the teaching flow:**
   - Break content into timed sections
   - For each section, describe:
     - Core concept to teach
     - How to explain it (including analogies)
     - Example to present
     - Verification questions to ask

4. **List common misconceptions:**
   - What learners often misunderstand
   - How to clarify these misconceptions
   - Alternative explanations if first doesn't work

5. **Provide verification questions:**
   - Specific questions to check understanding
   - What good answers sound like
   - What to do if learner struggles

6. **Add adaptive strategies:**
   - How to help struggling learners
   - How to challenge excelling learners
   - Different approaches for different learning styles

7. **Include exercise guidance:**
   - How to present the exercise
   - Progressive hints to provide
   - How to review learner's solutions
   - What to emphasize in the solution

### Step 4: Ensure Alignment

Check that your tutorial and guide:
- [ ] Have identical learning objectives
- [ ] Cover the same concepts in the same order
- [ ] Use the same examples (or complementary ones)
- [ ] Have matching exercises
- [ ] Reference each other in the metadata
- [ ] Have similar section structure (doesn't need to be exact)

### Step 5: Test Your Content

Before submitting:
1. **Test the tutorial independently**: Can someone learn from it alone?
2. **Test the guide**: Can an AI effectively teach using it?
3. **Test together**: Do they work well in collaborative mode?
4. **Get feedback**: Have others review or try your content

## Content Structure Guidelines

### Tutorial Structure (Human-Readable)

```markdown
---
[metadata]
---

# Title

## Overview
Brief description and motivation

## Learning Objectives
Clear, actionable objectives

## Prerequisites
What learners need to know first

## Setup
Installation and configuration

---

## Section 1: [Concept]

### What is [Concept]?
Clear explanation with analogy

### Why [Concept] Matters
Importance and use cases

### Understanding [Concept]
Detailed explanation with examples

### Key Points to Remember
Summary of important takeaways

### Exercise 1.1
Task, hints, solution

### Self-Check
Checklist before moving on

---

[More sections...]

---

## Practice Project
Comprehensive project using all concepts

## Summary
Key takeaways and skills gained

## Next Steps
Where to go from here
```

### Guide Structure (AI Teaching Script)

```markdown
---
[metadata]
---

# Title - AI Teaching Guide

## Teaching Overview
Learning objectives, prerequisites, time estimates

---

## Teaching Flow

### Introduction (X min)
- What to cover
- Opening questions to assess level
- How to adapt based on responses

---

### Section 1: [Concept] (X min)

**Core Concept to Teach:**
Brief description

**How to Explain:**
Step-by-step teaching approach

**Example to Present:**
Code example with walkthrough guidance

**Common Misconceptions:**
What learners misunderstand and corrections

**Verification Questions:**
Questions to check understanding

**If they struggle:**
Strategies to help

**Exercise 1.1:**
How to present and guide through it

---

[More sections...]

---

## Practice Project (X min)
How to guide learners through the project

## Wrap-Up (X min)
Review, assess confidence, suggest next steps

## Adaptive Teaching Strategies
Different approaches for different situations

## Troubleshooting
Common issues and how to address them
```

## Style Guidelines

### For Tutorials (Human-Focused)

**Writing Style:**
- Use conversational, encouraging tone
- Write in second person ("you")
- Keep sentences clear and concise
- Use active voice
- Explain "why" not just "how"

**Formatting:**
- Use `#` for title, `##` for major sections, `###` for subsections
- Use code blocks with language specification: ```python
- Use expandable sections for hints: `<details><summary>`
- Use **bold** for emphasis, `code` for inline code
- Use numbered lists only when order matters

**Code Examples:**
- Complete and runnable
- Well-commented
- Show expected output
- Start simple, build complexity

### For Guides (AI Teaching Scripts)

**Writing Style:**
- Write as teaching instructions for AI
- Use imperative tone ("Ask the learner...", "Explain that...")
- Be specific about what to say and do
- Include timing and pacing guidance

**Content:**
- Break teaching into clear sections with time estimates
- Provide specific questions, not just "check understanding"
- List multiple ways to explain difficult concepts
- Include strategies for different scenarios

**Code Examples:**
- Include same examples as tutorial
- Add guidance on how to walk through them
- Note what to emphasize
- Explain common areas of confusion

## Review Process

### Before Submitting

**Tutorial checklist:**
- [ ] Clear learning objectives
- [ ] Readable explanations without jargon
- [ ] Working code examples (test them!)
- [ ] Exercises with hints and solutions
- [ ] Self-check points throughout
- [ ] Practice project
- [ ] References paired guide in metadata

**Guide checklist:**
- [ ] Structured teaching flow with time estimates
- [ ] Specific verification questions
- [ ] Common misconceptions addressed
- [ ] Adaptive strategies included
- [ ] Exercise guidance (not just answers)
- [ ] References paired tutorial in metadata

**Alignment checklist:**
- [ ] Same learning objectives
- [ ] Same concepts in same order
- [ ] Exercises match
- [ ] Similar examples or complementary ones
- [ ] Cross-referenced in metadata

### Submitting Your Content

1. **Create a pull request** with:
   - Both tutorial and guide files
   - Clear description of what the content teaches
   - Note about target difficulty level
   - Any special requirements or dependencies

2. **In your PR description**, mention:
   - What topic you're covering
   - Who the target audience is
   - How you tested the content
   - Any questions or areas where you want feedback

3. **Review process**:
   - Maintainers will review both files
   - May ask for clarifications or changes
   - Might test content with learners
   - Will merge when both files are ready

## Improving Existing Content

When improving existing content:

**For tutorials:**
- Add clarifications where concepts are confusing
- Improve examples if better ones exist
- Add or improve exercises
- Fix errors or outdated information
- Improve readability

**For guides:**
- Add verification questions if missing
- Improve teaching strategies based on experience
- Add common misconceptions discovered from teaching
- Enhance adaptive strategies
- Update timing estimates

**For both:**
- Maintain alignment between tutorial and guide
- Update both if you change core concepts
- Document what you changed in your PR
- Explain why the change improves learning

## Questions?

If you need help:
- Check the templates in `/templates/`
- Look at existing examples in `/tutorials/` and `/guides/`
- Open an issue to ask questions
- Ask for feedback in pull request comments

Thank you for helping make learning more accessible and effective!
- Objective 1
- Objective 2
- Objective 3
```

3. **Clear Section Structure**:
- Use consistent heading levels
- Number sections if order matters
- Include "Checkpoint" sections for validation

4. **Code Examples**:
- Include complete, runnable examples
- Show expected output
- Explain what the code does

5. **Exercises**:
- Provide clear instructions
- Include hints (use details/summary tags)
- Offer solution hints or approaches

### For Human-Friendly Content

- Use conversational, encouraging tone
- Break down complex concepts
- Provide context and real-world applications
- Include visual aids when helpful (diagrams, flowcharts)
- Add "Common Pitfalls" or "Tips" sections

## Creating a New Tutorial

1. **Use the Template**: Start with `templates/tutorial-template.md`
2. **Choose the Right Directory**:
   - `tutorials/programming/` - Programming language tutorials
   - `tutorials/tools/` - Specific tool or framework tutorials
   - `tutorials/concepts/` - Conceptual or theoretical topics
   - `guides/` - Quick reference guides

3. **Name Your File**: Use descriptive, kebab-case names (e.g., `getting-started-with-python.md`)

4. **Fill in the Template**: Follow the structure provided

5. **Test Your Tutorial**:
   - Work through it yourself
   - Have someone else (or AI) try it
   - Verify all examples work
   - Check that exercises are solvable

6. **Submit a Pull Request**: Include a description of what your tutorial teaches

## Improving Existing Content

When improving existing tutorials:
- Maintain the existing structure unless there's a good reason to change it
- Add clarifications where learners commonly struggle
- Update examples if better ones exist
- Add exercises if they're missing
- Fix errors or outdated information
- Document what you changed in your PR

## Style Guidelines

### Markdown Formatting
- Use `#` for titles, `##` for main sections, `###` for subsections
- Use code blocks with language specification: ```python
- Use bullet points for lists
- Use numbered lists only when order matters
- Use **bold** for emphasis, *italic* for terms, `code` for inline code

### Writing Style
- Use active voice
- Keep sentences clear and concise
- Define technical terms on first use
- Provide context before diving into details
- Use examples liberally

### Code Examples
- Keep examples focused and minimal
- Include comments for complex parts
- Show complete, runnable code
- Indicate expected output
- Test all code before submitting

## Review Process

When submitting a tutorial:
1. Ensure it follows the template structure
2. Test all code examples
3. Have at least one person review (can be AI-assisted)
4. Address feedback constructively
5. Update based on testing with real learners

## Questions?

If you have questions about contributing, feel free to:
- Open an issue for discussion
- Ask in pull request comments
- Reference existing tutorials as examples

Thank you for helping make learning more accessible and effective!
