---
title: Agent Sandboxing - Isolating AI Coding Agents
topics:
  - Agent Isolation
  - Git Worktrees
  - Container Sandboxing
  - Orchestration
  - Development Environments
related_routes:
  - git-basics
  - mise-basics
  - nix-dev-environments
  - docker-dev-environments
---

# Agent Sandboxing - Route Map

## Overview

This route covers strategies for isolating AI coding agents (Claude Code, Cursor, Copilot Workspace, and similar tools) so they can work safely and in parallel without interfering with each other or your main development environment. You'll learn a spectrum of isolation techniques — from lightweight git worktrees to full container sandboxing — and when each approach makes sense.

This is a conceptual and practical route. Rather than teaching a single tool in depth, it brings together git worktrees, tool version managers (mise, Nix), and Docker containers into a coherent isolation strategy.

## What You'll Learn

By following this route, you will:
- Understand why agent isolation matters and the risks of uncontained agents
- Use git worktrees to give each agent its own working directory
- Pin tool versions per worktree with mise and Nix for reproducible environments
- Run agent tasks inside Docker containers with restricted network, CPU, memory, and filesystem access
- Orchestrate multiple agents working in parallel on different tasks
- Combine these techniques into layered isolation strategies

## Prerequisites

Before starting this route:
- **Required**: Basic command line usage (cd, ls, pwd, running programs)
- **Required**: git-basics route (branching, committing, merging)
- **Helpful**: mise-basics (for Section 3)
- **Helpful**: nix-dev-environments (for Section 3)
- **Helpful**: docker-dev-environments (for Section 4)

## Route Structure

### 1. Why Agent Isolation Matters
- The blast radius problem
- The isolation spectrum: nothing to cloud sandboxes
- Three reasons for isolation: safety, reproducibility, parallelism

### 2. Code Isolation with Git Worktrees
- What worktrees are and how they work
- Creating and removing worktrees
- The one-agent-per-worktree pattern
- Claude Code's built-in worktree support

### 3. Environment Isolation with Dev Tool Managers
- mise for tool version pinning per worktree
- Nix for full environment reproducibility
- Combining worktrees and tool managers
- Understanding the limits (not security sandboxes)

### 4. Container-Based Sandboxing
- Running agent tasks inside Docker containers
- Restricting network access, CPU, memory, and filesystem
- Volume-mounting worktrees into containers
- Docker-in-Docker considerations

### 5. Orchestration Patterns
- Running multiple agents in parallel
- Task distribution strategies
- Coordination and conflict resolution via git
- Monitoring agent progress

### 6. Practice Project
- Set up an isolated agent workspace end-to-end
- Create a worktree, configure tool versions, run a task in a container, clean up

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- git worktrees (`git worktree add/list/remove`)
- mise (tool version manager)
- Nix flakes (reproducible environments)
- Docker (container runtime)
- Shell scripting (orchestration)

## Next Steps

After completing this route:
- **mise-basics** — Deep dive into tool version management if you haven't done it yet
- **nix-dev-environments** — Full Nix flake setup for development environments
- **docker-dev-environments** — Docker fundamentals for development workflows
- Explore your AI coding tool's built-in isolation features and configure them for your projects
