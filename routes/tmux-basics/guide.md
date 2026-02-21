---
title: tmux Basics - Terminal Multiplexing Fundamentals
route_map: /routes/tmux-basics/map.md
paired_sherpa: /routes/tmux-basics/sherpa.md
prerequisites:
  - Basic command line usage
  - Terminal emulator familiarity
topics:
  - Terminal Multiplexing
  - tmux
  - Sessions
  - Windows
  - Panes
---

# tmux Basics - Guide (Human-Focused Content)

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/tmux-basics/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/tmux-basics/map.md` for the high-level overview.

## Overview

tmux is a terminal multiplexer — it lets you run multiple terminal sessions inside a single window, keep them running in the background, and reattach to them later. This tutorial will teach you the fundamental concepts and commands needed to use tmux effectively. You'll learn how to manage sessions, organize your workspace with windows and panes, and customize tmux to fit your workflow.

## Learning Objectives

By the end of this tutorial, you will be able to:
- Explain what terminal multiplexing is and why it's useful
- Create, detach from, and reattach to tmux sessions
- Organize work with windows and panes
- Navigate efficiently with tmux key bindings
- Write a basic `.tmux.conf` configuration

## Prerequisites

Before starting this tutorial, you should be familiar with:
- Using the command line/terminal
- Running programs from the terminal
- Basic file system navigation (cd, ls, pwd)

## Setup

Install tmux on your system:

**macOS:**
```bash
brew install tmux
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install tmux
```

**Linux (Fedora):**
```bash
sudo dnf install tmux
```

**Verify installation:**
```bash
tmux -V
```

**Expected Output:**
```
tmux 3.x
```

---

## Section 1: Understanding Terminal Multiplexing

### The Problem

Every time you close a terminal window, everything running in it dies. If you're SSH'd into a server and your connection drops, your running processes are gone. If you need to see two things at once, you open two terminal windows and arrange them manually.

tmux solves all of this.

### What tmux Does

tmux creates persistent terminal sessions that:
- **Survive disconnection** — close the terminal, the session keeps running
- **Contain multiple windows** — like tabs in a browser
- **Split into panes** — see multiple terminals side by side in one window

### The Hierarchy

tmux organizes your terminals in three levels:

```
Session (virtual desktop)
├── Window 1 (tab)
│   ├── Pane 1 (split)
│   └── Pane 2 (split)
├── Window 2 (tab)
│   └── Pane 1 (full window)
└── Window 3 (tab)
    ├── Pane 1
    ├── Pane 2
    └── Pane 3
```

- **Sessions** are like virtual desktops — separate workspaces for different projects
- **Windows** are like browser tabs — different views within a workspace
- **Panes** are like split screen — seeing multiple terminals at once within a tab

### The Prefix Key

tmux needs a way to distinguish between keystrokes meant for your terminal and commands meant for tmux itself. It uses a **prefix key**: `Ctrl-b` by default.

The prefix key is **sequential, not simultaneous**:
1. Press `Ctrl-b` (hold Ctrl, tap b)
2. **Release both keys**
3. Press the command key

For example, to detach: press `Ctrl-b`, release, then press `d`.

This is the most common source of confusion for beginners. If a key binding isn't working, make sure you're releasing `Ctrl-b` before pressing the next key.

### Checkpoint 1

Before moving on, make sure you understand:
- [ ] What tmux does and why it's useful
- [ ] The session → window → pane hierarchy
- [ ] How the prefix key works (sequential, not simultaneous)

---

## Section 2: Sessions

Sessions are the foundation of tmux. Everything else lives inside a session.

### Starting tmux

The simplest way to start:
```bash
tmux
```

**What you'll see:**
- Your terminal now has a green status bar at the bottom
- The status bar shows the session name, window list, and system info
- You're now inside a tmux session

### Creating Named Sessions

Named sessions are much easier to manage:
```bash
tmux new -s work
```

**Expected Output:**
A new tmux session opens with the name "work" visible in the status bar.

### Detaching from a Session

To leave a session running in the background:

Press `Ctrl-b`, release, then `d`

**Expected Output:**
```
[detached (from session work)]
```

You're back in your regular terminal. The tmux session is still running.

### Listing Sessions

```bash
tmux ls
```

**Expected Output:**
```
work: 1 windows (created Thu Feb 20 10:00:00 2026)
```

### Reattaching to a Session

```bash
# Attach to the most recent session
tmux attach

# Attach to a specific session by name
tmux attach -t work
```

### The Key Insight: Persistence

Here's where it gets powerful. Try this:

1. Create a session: `tmux new -s demo`
2. Run something visible: `top`
3. Detach: `Ctrl-b d`
4. See the session is still there: `tmux ls`
5. Reattach: `tmux attach -t demo`
6. `top` is still running, exactly where you left it

Detaching is like locking your computer — everything keeps running. `exit` (or `Ctrl-d`) kills the shell and the session. Detach when you want to come back later, exit when you're done.

### Killing Sessions

When you're done with a session:
```bash
tmux kill-session -t sessionname
```

Or from inside tmux, type `exit` in each window until the session is empty.

### Exercise 2.1: Multiple Named Sessions

**Task:** Create two named sessions: "project-a" and "project-b". In "project-a", run `top`. Detach, switch to "project-b", then list all sessions.

<details>
<summary>Hint 1</summary>

Use `tmux new -s name` to create named sessions.
</details>

<details>
<summary>Hint 2</summary>

After detaching from the first session, create the second one the same way.
</details>

<details>
<summary>Solution</summary>

```bash
# Create first session and run top
tmux new -s project-a
# (inside tmux) run: top
# Detach: Ctrl-b d

# Create second session
tmux new -s project-b

# Detach: Ctrl-b d

# List all sessions
tmux ls
```

**Expected output from `tmux ls`:**
```
project-a: 1 windows (created ...)
project-b: 1 windows (created ...)
```
</details>

### Exercise 2.2: Session Persistence

**Task:** Create a named session, run `ping localhost` in it, detach, close your terminal window entirely, open a new terminal, and reattach. Is the ping still running?

<details>
<summary>Hint</summary>

After closing the terminal, open a new one and use `tmux attach -t name` to reconnect.
</details>

<details>
<summary>Solution</summary>

```bash
# Create session
tmux new -s persistent

# Inside tmux, run:
ping localhost

# Detach: Ctrl-b d

# Close terminal window entirely
# Open a new terminal

# Reattach
tmux attach -t persistent
```

The ping is still running. tmux sessions survive terminal closures because the tmux server runs independently.
</details>

### Checkpoint 2

Before moving on, make sure you can:
- [ ] Create named sessions
- [ ] Detach from a session and reattach
- [ ] List running sessions
- [ ] Explain the difference between detaching and exiting

---

## Section 3: Windows and Panes

### Windows

Windows are like tabs within a session. You see one window at a time, and you can switch between them.

**Create a new window:**
`Ctrl-b c`

**Navigate between windows:**
- `Ctrl-b n` — next window
- `Ctrl-b p` — previous window
- `Ctrl-b 0`, `Ctrl-b 1`, `Ctrl-b 2` — jump to window by number

**Rename the current window:**
`Ctrl-b ,` — then type the new name and press Enter

**Close a window:**
Type `exit` or `Ctrl-d` in the last pane of the window.

Look at the status bar at the bottom — it shows all windows in the current session, with the active one highlighted.

### Panes

Panes split a window so you can see multiple terminals at once.

**Split horizontally** (top/bottom):
`Ctrl-b "` (that's Ctrl-b then the double-quote key)

**Split vertically** (left/right):
`Ctrl-b %`

**Navigate between panes:**
`Ctrl-b` then an arrow key (up/down/left/right)

**Zoom a pane** to full screen (toggle):
`Ctrl-b z`

Zooming is useful when you need to focus on one pane temporarily without losing your layout. Press `Ctrl-b z` again to unzoom.

**Close a pane:**
Type `exit` or press `Ctrl-d`

**Show pane numbers** (briefly):
`Ctrl-b q`

### Exercise 3.1: Working with Windows

**Task:** Create a session with 3 named windows: "editor", "server", and "logs". Navigate between them using both next/previous and number keys.

<details>
<summary>Hint 1</summary>

Create a session, then use `Ctrl-b c` to create new windows and `Ctrl-b ,` to name each one.
</details>

<details>
<summary>Hint 2</summary>

The first window is number 0 by default. New windows get the next number.
</details>

<details>
<summary>Solution</summary>

```bash
# Create a session
tmux new -s myworkspace

# Rename the first window
# Ctrl-b , → type "editor" → Enter

# Create second window
# Ctrl-b c
# Ctrl-b , → type "server" → Enter

# Create third window
# Ctrl-b c
# Ctrl-b , → type "logs" → Enter

# Navigate:
# Ctrl-b 0 → jumps to "editor"
# Ctrl-b 1 → jumps to "server"
# Ctrl-b 2 → jumps to "logs"
# Ctrl-b n → next window
# Ctrl-b p → previous window
```

The status bar at the bottom should show: `0:editor  1:server  2:logs`
</details>

### Exercise 3.2: Working with Panes

**Task:** In a single window, create a layout with one large pane on the left and two smaller panes stacked on the right. Navigate between all three panes using arrow keys. Then zoom into one pane and back out.

<details>
<summary>Hint 1</summary>

Start with a full window. Split vertically first (`Ctrl-b %`), then split the right pane horizontally (`Ctrl-b "`).
</details>

<details>
<summary>Hint 2</summary>

Make sure your cursor is in the right pane before splitting it horizontally.
</details>

<details>
<summary>Solution</summary>

```bash
# Start in a full window

# Split vertically (left | right)
# Ctrl-b %

# Cursor is now in the right pane
# Split horizontally (top/bottom on the right side)
# Ctrl-b "

# Now you have:
# ┌──────────┬──────────┐
# │          │  Pane 2  │
# │  Pane 1  ├──────────┤
# │          │  Pane 3  │
# └──────────┴──────────┘

# Navigate with Ctrl-b + arrow keys
# Zoom: Ctrl-b z (toggle)
```
</details>

### Checkpoint 3

Before moving on, make sure you can:
- [ ] Create and name windows
- [ ] Navigate between windows by number and by next/previous
- [ ] Split panes horizontally and vertically
- [ ] Navigate between panes with arrow keys
- [ ] Zoom a pane to full screen and back

---

## Section 4: Customizing tmux

### The Configuration File

tmux reads its configuration from `~/.tmux.conf`. If it doesn't exist, tmux uses defaults.

Create the file:
```bash
touch ~/.tmux.conf
```

### Common Customizations

Add these to your `~/.tmux.conf`:

**Change prefix to Ctrl-a** (easier to reach than Ctrl-b):
```
# Remap prefix from Ctrl-b to Ctrl-a
unbind C-b
set -g prefix C-a
bind C-a send-prefix
```

**Enable mouse support** (click panes, scroll, resize):
```
# Enable mouse support
set -g mouse on
```

**Start numbering at 1** (0 is far from the other keys):
```
# Start window and pane numbering at 1
set -g base-index 1
setw -g pane-base-index 1
```

**Add a config reload shortcut:**
```
# Reload config with prefix-r
bind r source-file ~/.tmux.conf \; display "Config reloaded"
```

### Applying Configuration Changes

Changes to `.tmux.conf` don't take effect automatically. You need to reload the config.

**From the tmux command prompt:**
Press `Ctrl-b :` (or your new prefix + `:`) then type:
```
source-file ~/.tmux.conf
```

**Or if you added the reload shortcut above:**
Press your prefix key, then `r`.

### Exercise 4.1: Set Up Your Configuration

**Task:** Create a `~/.tmux.conf` with the prefix change and mouse support. Reload it inside tmux and verify both work.

<details>
<summary>Hint 1</summary>

Create the file with your text editor, add the config lines, save it.
</details>

<details>
<summary>Hint 2</summary>

To reload: `Ctrl-b :` then type `source-file ~/.tmux.conf`. After reloading, the new prefix is `Ctrl-a`.
</details>

<details>
<summary>Solution</summary>

Add to `~/.tmux.conf`:
```
unbind C-b
set -g prefix C-a
bind C-a send-prefix

set -g mouse on

set -g base-index 1
setw -g pane-base-index 1

bind r source-file ~/.tmux.conf \; display "Config reloaded"
```

Then inside tmux:
1. Reload: `Ctrl-b :` → type `source-file ~/.tmux.conf` → Enter
2. Test new prefix: `Ctrl-a c` should create a new window
3. Test mouse: Click on different panes, try scrolling

**Note:** After reloading, your prefix is now `Ctrl-a`, not `Ctrl-b`.
</details>

### Checkpoint 4

Before moving on, make sure you can:
- [ ] Create and edit `~/.tmux.conf`
- [ ] Reload configuration from inside tmux
- [ ] Verify that configuration changes took effect

---

## Practice Project

### Project Description

Build a development workspace that simulates a real-world setup. You'll create a named session with multiple windows and panes organized for a development workflow.

### Requirements
- Create a named session called "dev"
- Set up 3 windows named: "editor", "server", "monitor"
- In the "monitor" window, split into at least 2 panes
- Run a visible process in at least one pane (`top`, `htop`, `watch ls`, etc.)
- Detach from the session, then reattach to verify everything persists

### Getting Started

```bash
tmux new -s dev
```

Now set up your windows, panes, and processes!

### Validation

After reattaching, verify:
```bash
# Outside tmux, you should see:
tmux ls
# Output: dev: 3 windows (created ...)
```

Inside the session:
- Status bar shows `0:editor  1:server  2:monitor` (or starting from 1 if you configured that)
- The "monitor" window has a split layout with panes
- Your running process survived the detach/reattach

---

## Key Bindings Reference

All bindings use the default prefix `Ctrl-b` (unless you've changed it).

| Action | Key Binding |
|--------|------------|
| **Sessions** | |
| Detach | `Ctrl-b d` |
| List sessions (from shell) | `tmux ls` |
| Session tree (inside tmux) | `Ctrl-b s` |
| **Windows** | |
| New window | `Ctrl-b c` |
| Next window | `Ctrl-b n` |
| Previous window | `Ctrl-b p` |
| Window by number | `Ctrl-b 0-9` |
| Rename window | `Ctrl-b ,` |
| **Panes** | |
| Split horizontal | `Ctrl-b "` |
| Split vertical | `Ctrl-b %` |
| Navigate panes | `Ctrl-b arrow` |
| Zoom toggle | `Ctrl-b z` |
| Show pane numbers | `Ctrl-b q` |
| **Other** | |
| Command prompt | `Ctrl-b :` |
| Window list | `Ctrl-b w` |

## Summary

You've learned the fundamentals of tmux:
- tmux creates persistent terminal sessions that survive disconnection
- Sessions contain windows (tabs), which contain panes (splits)
- The prefix key (`Ctrl-b`) signals tmux commands
- Detaching preserves sessions; exiting kills them
- `~/.tmux.conf` lets you customize tmux to your liking

## Next Steps

Now that you understand tmux basics, explore:
- **Copy Mode**: Search and copy text from terminal output (`Ctrl-b [`)
- **Scripted Sessions**: Automate workspace creation with shell scripts
- **tmux Plugin Manager (tpm)**: Extend tmux with community plugins
- **Advanced Configuration**: Status bar customization, key binding tables, hooks

## Additional Resources

- [tmux man page](https://man.openbsd.org/tmux): Complete reference
- [tmux GitHub](https://github.com/tmux/tmux): Source and wiki
- [The Tao of tmux](https://leanpub.com/the-tao-of-tmux/read): Free online book
