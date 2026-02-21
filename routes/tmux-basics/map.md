---
title: tmux Basics - Terminal Multiplexing Fundamentals
topics:
  - Terminal Multiplexing
  - tmux
  - Sessions
  - Windows
  - Panes
related_routes:
  - tmux-advanced
  - command-line-productivity
---

# tmux Basics - Route Map

## Overview

This route teaches the fundamentals of tmux, a terminal multiplexer. tmux lets you run multiple terminal sessions inside a single window, keep them running in the background, and reattach to them later. It's indispensable for remote work, long-running processes, and organizing your terminal workspace.

## What You'll Learn

By following this route, you will:
- Understand what terminal multiplexing is and why it matters
- Create, detach from, and reattach to tmux sessions
- Organize your work with windows and panes
- Navigate efficiently between sessions, windows, and panes
- Customize tmux with a `.tmux.conf` configuration file
- Use copy mode to scroll back through terminal output

## Prerequisites

Before starting this route:
- **Required**: Basic command line usage (cd, ls, pwd, running programs)
- **Required**: tmux installed on your system
- **Helpful**: Familiarity with a terminal emulator

## Route Structure

### 1. Understanding Terminal Multiplexing
- What is tmux?
- The problem tmux solves
- The session/window/pane hierarchy
- The prefix key concept

### 2. Sessions
- Creating and naming sessions
- Detaching and reattaching
- Listing and switching sessions
- Killing sessions

### 3. Windows and Panes
- Creating and naming windows
- Navigating between windows
- Splitting panes horizontally and vertically
- Navigating and resizing panes
- Zooming a pane to full screen

### 4. Customizing tmux
- The `.tmux.conf` file
- Changing the prefix key
- Enabling mouse support
- Setting base index
- Reloading configuration

### 5. Copy Mode
- Entering and exiting copy mode
- Scrolling through terminal output
- Vi vs default key bindings in copy mode

### 6. Practice Project
- Build a development workspace with named sessions, windows, and panes

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- tmux installation guides
- Terminal emulator basics
- Shell customization (see techniques/)

## Next Steps

After completing this route:
- **tmux Advanced** - Scripting, plugins, advanced copy mode, and session automation
- **Command Line Productivity** - Combine tmux with other CLI tools for an efficient workflow
