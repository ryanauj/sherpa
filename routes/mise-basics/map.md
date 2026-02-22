---
title: mise Basics - Dev Tool Version Management
topics:
  - mise
  - Version Management
  - Environment Variables
  - Task Running
related_routes:
  - nix-dev-environments
  - docker-dev-environments
  - agent-sandboxing
---

# mise Basics - Route Map

## Overview

This route teaches the fundamentals of mise, a polyglot dev tool version manager. mise replaces the patchwork of language-specific version managers (nvm, pyenv, rbenv, etc.) with a single tool. Beyond version management, mise handles project-level environment variables and task running — combining what you'd otherwise piece together from direnv, Makefiles, and npm scripts.

## What You'll Learn

By following this route, you will:
- Understand why mise exists and what it replaces
- Install mise and configure your shell
- Install and manage multiple tool versions (Node, Python, etc.)
- Configure per-project tool versions with `.mise.toml`
- Set project-scoped environment variables
- Define and run project tasks

## Prerequisites

Before starting this route:
- **Required**: Basic command line usage (cd, ls, running programs)
- **Helpful**: Experience with any version manager (nvm, pyenv, rbenv)

## Route Structure

### 1. The Problem mise Solves
- Version manager sprawl (nvm, pyenv, rbenv, etc.)
- How mise unifies them
- Shims vs PATH manipulation
- What mise is NOT (not a package manager, not a sandbox)

### 2. Installation and Shell Setup
- Installing mise
- Activating mise in your shell
- Verifying with `mise doctor`

### 3. Managing Tool Versions
- Installing tools with `mise install`
- Setting project versions with `mise use`
- Listing installed versions with `mise ls`
- Version resolution order (local, parent dirs, global)
- Compatibility with `.tool-versions` files

### 4. Environment Variables
- The `[env]` section in `.mise.toml`
- Loading `.env` files
- PATH manipulation with `_.path`

### 5. Tasks
- Defining tasks in `.mise.toml`
- Running tasks with `mise run`
- Task dependencies
- File-based tasks in `.mise/tasks/`

### 6. Practice Project
- Set up a multi-tool project with Node + Python, environment variables, and build/dev tasks

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- mise installation guides
- Shell configuration basics
- TOML configuration format

## Next Steps

After completing this route:
- **Nix Dev Environments** - Fully reproducible, declarative development environments
- **Docker Dev Environments** - Containerized development workflows
- **Agent Sandboxing** - Isolating AI agent tool execution
