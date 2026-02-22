---
title: Docker Dev Environments - Container-Based Development
topics:
  - Docker
  - Containers
  - Dev Containers
  - Docker Compose
  - Development Environments
related_routes:
  - mise-basics
  - nix-dev-environments
  - agent-sandboxing
---

# Docker Dev Environments - Route Map

## Overview

This route teaches how to use Docker for development environments — not production deployment. Docker containers provide isolated, reproducible environments where your project's dependencies don't leak into your host system and every team member gets an identical setup. You'll learn Docker fundamentals, how to write Dockerfiles optimized for dev workflows, how to use volumes for live code editing, how to configure dev containers for editor integration, and how to compose Docker with tools like mise and Nix inside containers.

## What You'll Learn

By following this route, you will:
- Understand what containers are and how they differ from VMs
- Run, inspect, and manage Docker containers
- Write Dockerfiles with efficient layer caching for fast rebuilds
- Use bind mounts and volumes for a productive dev workflow
- Set up multi-container environments with Docker Compose
- Configure dev containers for VS Code and other editors
- Combine Docker with mise or Nix for tool version management inside containers

## Prerequisites

Before starting this route:
- **Required**: Basic command line usage (cd, ls, pwd, running programs)
- **Helpful**: mise-basics route (for Section 6)
- **Helpful**: nix-dev-environments route (for Section 6)

## Route Structure

### 1. What Docker Does
- Containers vs VMs
- Images vs containers
- When Docker makes sense for dev environments
- Docker Desktop on macOS

### 2. Docker Fundamentals
- Running containers with `docker run`
- Listing and inspecting containers
- Executing commands in running containers
- Pulling images and running interactive shells
- Stopping and removing containers

### 3. Writing Dockerfiles
- FROM, RUN, COPY, WORKDIR, CMD, ENV, EXPOSE
- Layer caching and ordering for fast rebuilds
- Multi-stage builds for smaller images
- .dockerignore for excluding files

### 4. Volumes and Bind Mounts
- The dev workflow problem: reflecting code changes in containers
- Bind mounts for live code editing
- Named volumes for data persistence
- tmpfs mounts for ephemeral data
- Docker Compose for multi-container dev setups

### 5. Dev Containers
- The devcontainer.json spec
- VS Code and editor integration
- Features for adding tools
- Customizing the dev experience
- When dev containers make sense vs plain Docker

### 6. Composing with mise and Nix
- Using mise inside a Docker container for tool version management
- Using Nix for reproducible container builds
- Multi-stage builds with Nix
- Trade-offs of each approach
- When to compose tools vs use one tool alone

### 7. Practice Project
- Build a dev container for a Node.js + Python project
- Docker Compose with bind mounts for live code editing
- Tool management via mise inside the container

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- Docker Desktop installation
- Docker CLI
- Docker Compose
- Dev Container CLI and VS Code extension
- mise and Nix (for Section 6)

## Next Steps

After completing this route:
- **mise-basics** - Tool version management with mise (if not already completed)
- **nix-dev-environments** - Fully reproducible development environments with Nix
- **agent-sandboxing** - Sandboxing AI agents with containers
