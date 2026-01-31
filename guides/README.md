# Guides - AI Teaching Scripts

This directory contains **AI teaching guides** that provide structured approaches for AI assistants to teach various topics. Each guide corresponds to a route in the `/routes` directory.

## Purpose

Guides are teaching scripts designed for AI assistants. They include:
- Structured teaching flow and progression
- Questions to ask learners at each stage
- Common misconceptions to address
- Adaptive strategies for different learning styles
- Exercise guidance and hints to provide

## Structure

- **programming/** - Teaching guides for programming languages
- **tools/** - Teaching guides for specific tools and frameworks
- **concepts/** - Teaching guides for theoretical concepts

## How AI Assistants Use Guides

1. **Select the appropriate guide** based on the topic the learner wants to learn
2. **Follow the teaching flow** section by section
3. **Ask verification questions** at checkpoints to assess understanding
4. **Adapt teaching** based on learner responses (struggling vs. excelling)
5. **Guide through exercises** using the hints and strategies provided
6. **Reference the paired route** if the learner is reading along

## Three Teaching Modes

### Mode 1: Route-Free Teaching
Learner has NOT seen the route. Use the guide to:
- Explain concepts from scratch
- Present examples from the guide
- Check understanding with verification questions
- Work through exercises together

### Mode 2: Route-Assisted Teaching
Learner IS reading the route. Use the guide to:
- Let them read sections, then discuss
- Answer questions about route content
- Ask verification questions from the guide
- Help with exercises in the route

### Mode 3: Review/Support Mode
Learner has COMPLETED the route. Use the guide to:
- Answer specific questions
- Clarify confusing points
- Provide additional examples
- Suggest next steps

## Creating a Guide

See the template in `/templates/guide-template.md` and refer to `/CONTRIBUTING.md` for detailed guidelines.

## Pairing with Routes

Each guide should have a corresponding route with the same filename:
- `guides/tools/git-basics.md` ↔ `routes/tools/git-basics.md`
- Both cover the same material, optimized for different use cases
