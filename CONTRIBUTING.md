# Contributing to Tides

Thank you for your interest in contributing to Tides! This document provides guidelines for creating and improving tutorials and guides that work effectively for both AI and human learners.

## Philosophy

Our tutorials and guides should be:
- **Clear and Structured**: Easy to follow for both humans and AI
- **Practical**: Include real examples and hands-on exercises
- **Iterative**: Designed to be improved over time
- **Accessible**: Understandable for the target audience
- **Validatable**: Include checkpoints to confirm understanding

## Types of Content

### Tutorials
Step-by-step learning paths that take a learner from beginner to competent in a specific topic. Tutorials should:
- Have a clear learning progression
- Include practical exercises
- Provide validation checkpoints
- Explain the "why" behind concepts

### Guides
Quick reference materials for specific tasks or concepts. Guides should:
- Be concise and focused
- Provide immediate value
- Include examples
- Reference related tutorials for deeper learning

## Content Structure

### For AI-Friendly Content

To make content usable by AI assistants as tutors, include:

1. **Metadata Section** (at the top):
```markdown
---
title: Tutorial Title
difficulty: beginner|intermediate|advanced
duration: X minutes
prerequisites: [list of required knowledge]
topics: [list of topics covered]
---
```

2. **Learning Objectives**:
```markdown
## Learning Objectives
By the end of this tutorial, you will be able to:
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
