---
title: Nix Dev Environments - Reproducible Development Shells
topics:
  - Nix
  - Reproducible Environments
  - Development Shells
  - Flakes
  - direnv
related_routes:
  - mise-basics
  - docker-dev-environments
  - agent-sandboxing
---

# Nix Dev Environments - Route Map

## Overview

This route teaches you how to use Nix to create fully reproducible development environments. Nix is a functional package manager that guarantees every developer on a project gets the exact same tools, versions, and configuration. You'll go from installation to a complete flake-based dev shell with automatic activation via direnv.

## What You'll Learn

By following this route, you will:
- Understand why reproducible environments matter and what Nix guarantees
- Install Nix and use it for ad-hoc package management
- Read and write basic Nix language expressions
- Create development shells with `shell.nix`
- Use flakes for version-locked, portable dev environments
- Set up direnv integration for automatic shell activation

## Prerequisites

Before starting this route:
- **Required**: Basic command line usage (cd, ls, running programs)
- **Required**: Comfort with new language/configuration syntax
- **Helpful**: Experience with any package manager (brew, apt)
- **Helpful**: mise-basics route (for comparison)

## Route Structure

### 1. Why Nix? The Reproducibility Problem
- "Works on my machine" and why it happens
- Functional package management
- The Nix store and content-addressed paths
- Guarantees Nix provides that other tools can't

### 2. Installing Nix and First Commands
- Installing with the Determinate Systems installer
- Enabling flakes
- Ad-hoc packages with `nix-shell -p`
- Finding packages with `nix search nixpkgs`

### 3. Nix Language Fundamentals
- Attribute sets
- Functions
- Let/in blocks
- `with` and `inherit`
- Strings, multi-line strings, and lists

### 4. Development Shells with shell.nix
- `mkShell` and `buildInputs`
- Shell hooks for setup commands
- Pinning nixpkgs for reproducibility
- Running with `nix-shell`

### 5. Flakes: The Modern Approach
- `flake.nix` structure (inputs/outputs)
- `flake.lock` for version pinning
- `nix develop` command
- Multi-system support
- Why flakes improve on plain `shell.nix`

### 6. direnv Integration
- What direnv does
- nix-direnv for automatic shell activation
- `.envrc` with `use flake`
- The seamless workflow: cd into a project and tools appear

### 7. Practice Project
- Build a reproducible multi-language dev environment with flake.nix + direnv

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- Nix package manager (Determinate Systems installer)
- nixpkgs package repository
- direnv and nix-direnv
- Shell configuration (see techniques/)

## Next Steps

After completing this route:
- **mise-basics** - Compare Nix's approach with mise's simpler version management
- **docker-dev-environments** - Learn containerized environments and how they compare to Nix
- **agent-sandboxing** - Use Nix environments to sandbox AI agent tool access
