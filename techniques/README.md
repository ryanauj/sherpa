# Techniques

This directory contains reusable patterns, templates, and best practices that can be applied across different routes and topics.

## Purpose

Techniques are knowledge resources that provide:
- **Templates**: Starting points for creating new content (routes, guides, sherpas)
- **Patterns**: Common approaches and structures that work well
- **Best Practices**: Proven methods for teaching and learning
- **Style Guides**: Conventions for writing and formatting content

## Organization

Techniques are organized by type:
- **templates/**: Templates for creating new routes, guides, and sherpas
- **patterns/**: Common teaching and learning patterns
- **best-practices/**: Documented best practices
- **style-guides/**: Writing and formatting conventions

## Usage

Techniques are referenced throughout the repository. For example:

```bash
# Start a new route using templates
cp techniques/templates/map-template.md routes/new-topic/map.md
cp techniques/templates/sherpa-template.md routes/new-topic/sherpa.md
cp techniques/templates/guide-template.md routes/new-topic/guide.md
```

## Creating Techniques

When you discover a pattern that works well:
1. Document it clearly with examples
2. Add it to the appropriate subdirectory
3. Reference it from routes where it applies

See `/CONTRIBUTING.md` for guidelines on adding techniques.
