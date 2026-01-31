# Templates

This directory contains templates for creating new tutorials and guides. These templates ensure consistency across the repository and include all the necessary sections for effective learning materials.

## Available Templates

### tutorial-template.md
Use this template when creating comprehensive learning paths. Tutorials should:
- Take learners from beginner to competent in a topic
- Include hands-on exercises
- Provide validation checkpoints
- Explain concepts thoroughly

**When to use:** Creating a full learning path (30+ minutes of content)

### guide-template.md
Use this template when creating quick reference materials. Guides should:
- Focus on a specific task or concept
- Provide immediate, actionable information
- Include troubleshooting tips
- Reference related tutorials

**When to use:** Creating task-specific or problem-solving documentation

## How to Use

1. **Copy the appropriate template:**
   ```bash
   cp templates/tutorial-template.md tutorials/[category]/[your-tutorial-name].md
   ```
   or
   ```bash
   cp templates/guide-template.md guides/[category]/[your-guide-name].md
   ```

2. **Fill in the metadata:** Update the YAML front matter with your content details

3. **Replace placeholder text:** Replace all `[brackets]` with actual content

4. **Follow the structure:** Keep the section headers and organization

5. **Add your content:** Fill in explanations, examples, and exercises

6. **Test it:** Work through your tutorial/guide to ensure it works

7. **Get feedback:** Have someone else review it (can use AI)

## Tips

- Don't remove sections unless they truly don't apply
- Keep the "For AI Tutors" section - it helps AI assistants use your content effectively
- Include working code examples that readers can copy and run
- Add more sections if needed, but keep the core structure
- Check `/CONTRIBUTING.md` for detailed guidelines

## Questions?

If you're unsure which template to use:
- **Tutorial:** Teaching a new skill or concept from scratch
- **Guide:** Helping someone complete a specific task or solve a problem

Still unsure? Open an issue and ask for guidance!
