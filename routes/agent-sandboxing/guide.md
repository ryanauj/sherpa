---
title: Agent Sandboxing - Isolating AI Coding Agents
route_map: /routes/agent-sandboxing/map.md
paired_sherpa: /routes/agent-sandboxing/sherpa.md
prerequisites:
  - Basic command line usage
  - git-basics
topics:
  - Agent Isolation
  - Git Worktrees
  - Container Sandboxing
  - Orchestration
  - Development Environments
---

# Agent Sandboxing - Guide (Human-Focused Content)

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/agent-sandboxing/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/agent-sandboxing/map.md` for the high-level overview.

## Overview

AI coding agents — Claude Code, Cursor, Copilot Workspace, and similar tools — can modify files, run commands, install packages, and access the network. That power makes them useful, but it also means a mistake or misconfiguration can affect your entire development environment.

This guide teaches you how to isolate agents so their work can't interfere with yours (or with each other). You'll learn a spectrum of techniques from lightweight (git worktrees) to heavyweight (Docker containers), and how to combine them for running multiple agents in parallel.

This is a conceptual and practical guide. It's about strategies and patterns for working with agents, not a deep dive into any single tool.

## Learning Objectives

By the end of this guide, you will be able to:
- Explain why agent isolation matters and describe the isolation spectrum
- Use git worktrees to give each agent its own working directory
- Configure mise or Nix to pin tool versions per worktree
- Run agent tasks inside Docker containers with resource and network restrictions
- Orchestrate multiple agents working in parallel
- Choose the right isolation level for a given task

## Prerequisites

Before starting this guide, you should be familiar with:
- Basic command line usage (cd, ls, pwd, running programs)
- Git fundamentals (branching, committing, merging — see the git-basics route)

**Helpful but not required:**
- mise-basics route (for Section 3)
- nix-dev-environments route (for Section 3)
- docker-dev-environments route (for Section 4)

## Setup

You'll need a git repository to practice with. If you don't have one handy, create a practice repo:

```bash
mkdir agent-sandbox-practice && cd agent-sandbox-practice
git init
echo "# Agent Sandbox Practice" > README.md
git add README.md
git commit -m "Initial commit"
```

For the full experience, you'll also want:
- **mise** installed (for Section 3) — see the mise-basics route
- **Docker** installed (for Sections 4-5) — see the docker-dev-environments route

If you don't have mise or Docker, you can still work through the concepts and exercises that don't require them.

---

## Section 1: Why Agent Isolation Matters

### The Blast Radius Problem

When you give an AI agent access to your codebase, it can modify any file, run any command, and install any package — all with the same permissions as your user account. If the agent makes a mistake, the damage extends to everything it can reach.

That scope of potential damage is called the **blast radius**.

An agent working in your main git checkout has a blast radius that includes:
- Every file in your repository (including uncommitted changes)
- Your global tool installations
- Your shell environment
- Your network access
- Your filesystem (depending on how the agent is configured)

Consider a concrete scenario: you're working on a feature with uncommitted changes. You ask an agent to refactor a module. It modifies 15 files, breaks the build, and now your uncommitted work is tangled with the agent's broken changes. Untangling that is painful.

### The Isolation Spectrum

Isolation techniques form a spectrum from "nothing" to "completely separate machine":

| Isolation Level | What It Isolates | Overhead | When to Use |
|---|---|---|---|
| Nothing (main checkout) | Nothing | None | Quick, low-risk tasks |
| Git worktree | File changes | Low | Most agent tasks |
| Tool pinning (mise) | Tool versions | Low | When agents need specific tool versions |
| Nix flake | Full environment | Medium | When exact reproducibility matters |
| Docker container | Process, network, filesystem | Medium-High | When you need real containment |
| VM | Entire OS | High | When you need full OS isolation |
| Cloud sandbox | Entire machine | Highest | When you need disposable infrastructure |

You don't always need the heaviest technique. A quick "fix this typo" task might be fine in your main checkout. A "refactor the entire authentication module" task deserves at least a worktree, probably a container.

### Three Reasons for Isolation

**Safety**: Limit what an agent can break. If it's working in a worktree, it can't touch your uncommitted changes. If it's in a container, it can't access your network or install things globally.

**Reproducibility**: Each agent gets the same environment every time. Tool versions are pinned. Dependencies are locked. The agent in your worktree today gets the same Node version as the agent next week.

**Parallelism**: Run multiple agents simultaneously without conflicts. Each agent gets its own worktree, its own branch, its own container. They can't step on each other's changes.

### Checkpoint 1

Before moving on, make sure you understand:
- [ ] What "blast radius" means in the context of agent isolation
- [ ] The isolation spectrum from nothing to cloud sandboxes
- [ ] The three reasons for isolation: safety, reproducibility, parallelism

---

## Section 2: Code Isolation with Git Worktrees

### What Worktrees Are

Normally, a git repository gives you one working directory — the files you see on disk. If you want to work on two branches simultaneously, you either stash and switch, or clone the entire repository.

Git worktrees provide a third option: additional working directories that share the same `.git` repository. Each worktree checks out a different branch, but they all share the same commit history, branches, and remotes.

```
my-project/                  <-- main worktree (your normal checkout)
my-project/.git/             <-- shared git data
../agent-task-1/             <-- worktree on feature/add-auth branch
../agent-task-2/             <-- worktree on feature/fix-tests branch
```

The key insight: changes in one worktree don't affect the others. An agent working in `../agent-task-1` can modify files, commit, even break the build — and none of it touches your main working directory.

### Creating a Worktree

```bash
# Create a worktree at ../agent-task-1, on a new branch feature/add-auth
git worktree add ../agent-task-1 -b feature/add-auth
```

**Expected output:**
```
Preparing worktree (new branch 'feature/add-auth')
HEAD is now at abc1234 Initial commit
```

This creates:
- A new directory at `../agent-task-1`
- A new branch called `feature/add-auth`
- A full checkout of the repository at the current HEAD

### Listing Worktrees

```bash
git worktree list
```

**Expected output:**
```
/path/to/my-project           abc1234 [main]
/path/to/agent-task-1         abc1234 [feature/add-auth]
```

### Working in a Worktree

A worktree is a regular directory. You can cd into it, edit files, run commands, and commit changes:

```bash
# Make a change in the worktree
echo "auth module" > ../agent-task-1/auth.md

# Commit from within the worktree
cd ../agent-task-1
git add auth.md
git commit -m "Add auth module"
cd -
```

Now check your main worktree:

```bash
ls            # No auth.md here
git log --oneline  # No "Add auth module" commit on this branch
```

The change exists only on the `feature/add-auth` branch, in the `../agent-task-1` directory. Your main checkout is untouched.

### One Rule: Branch Exclusivity

Each worktree must be on a different branch. You can't have two worktrees checking out `main` simultaneously. This is actually a feature for agent isolation — it prevents two agents from modifying the same branch.

If you try to create a worktree on a branch that's already checked out:

```bash
git worktree add ../another-task main
```

**Expected output:**
```
fatal: 'main' is already checked out at '/path/to/my-project'
```

### Cleaning Up Worktrees

```bash
# Remove a worktree
git worktree remove ../agent-task-1
```

**Expected output:**
(no output on success)

The directory is deleted, but the branch still exists:

```bash
git branch
```

**Expected output:**
```
  feature/add-auth
* main
```

You can merge or delete the branch separately:

```bash
# Merge the agent's work
git merge feature/add-auth

# Or delete the branch if you don't need it
git branch -d feature/add-auth
```

### Claude Code's Worktree Support

Claude Code has built-in worktree support. When you ask it to work on a task, it can:
- Create a worktree for the task with `git worktree add`
- Work in isolation without affecting your main working directory
- Clean up the worktree when the task completes

This happens transparently — Claude Code manages the worktree lifecycle for you.

### What Worktrees Don't Isolate

Worktrees isolate file changes, not system access. An agent working in a worktree can still:
- Run any command your user can run
- Install packages globally
- Access the network
- Read and write files outside the worktree directory
- Consume unlimited CPU and memory

Worktrees are the first layer of isolation, not the last. For stronger containment, you'll need the techniques in Sections 3 and 4.

### Exercise 2.1: Create and Use Multiple Worktrees

**Task:** Create three worktrees from your practice repository, each on a different feature branch. Make a unique change in each one and commit it. Then go back to your main worktree and verify all three branches exist.

<details>
<summary>Hint 1</summary>

The syntax is `git worktree add <path> -b <branch-name>`. Use different paths and branch names for each one.
</details>

<details>
<summary>Hint 2</summary>

After creating each worktree, `cd` into it, create a file, `git add` it, and `git commit`. Then `cd` back.
</details>

<details>
<summary>Solution</summary>

```bash
# Create three worktrees
git worktree add ../agent-auth -b feature/add-auth
git worktree add ../agent-tests -b feature/fix-tests
git worktree add ../agent-docs -b feature/update-docs

# Make a change in each
echo "auth code" > ../agent-auth/auth.md
cd ../agent-auth && git add auth.md && git commit -m "Add auth module" && cd -

echo "test fixes" > ../agent-tests/tests.md
cd ../agent-tests && git add tests.md && git commit -m "Fix tests" && cd -

echo "docs update" > ../agent-docs/docs.md
cd ../agent-docs && git add docs.md && git commit -m "Update docs" && cd -

# Verify branches exist from the main worktree
git branch
```

**Expected output from `git branch`:**
```
  feature/add-auth
  feature/fix-tests
  feature/update-docs
* main
```

You can also verify worktrees:
```bash
git worktree list
```

**Expected output:**
```
/path/to/my-project       abc1234 [main]
/path/to/agent-auth       def5678 [feature/add-auth]
/path/to/agent-tests      ghi9012 [feature/fix-tests]
/path/to/agent-docs       jkl3456 [feature/update-docs]
```
</details>

### Exercise 2.2: Clean Up Worktrees

**Task:** Remove all three worktrees from Exercise 2.1. Verify they're gone with `git worktree list`. Then delete the feature branches.

<details>
<summary>Hint</summary>

Use `git worktree remove <path>` for each worktree. Then `git branch -d <name>` for each branch.
</details>

<details>
<summary>Solution</summary>

```bash
# Remove worktrees
git worktree remove ../agent-auth
git worktree remove ../agent-tests
git worktree remove ../agent-docs

# Verify
git worktree list
```

**Expected output:**
```
/path/to/my-project  abc1234 [main]
```

```bash
# Delete branches (optional)
git branch -D feature/add-auth
git branch -D feature/fix-tests
git branch -D feature/update-docs
```

Note: Using `-D` (capital D) instead of `-d` because the branches were never merged into main. With `-d`, git would warn you about unmerged changes.
</details>

### Checkpoint 2

Before moving on, make sure you can:
- [ ] Create a worktree on a new feature branch
- [ ] Work in a worktree (edit, commit) without affecting the main checkout
- [ ] List and remove worktrees
- [ ] Explain what worktrees isolate and what they don't

---

## Section 3: Environment Isolation with Dev Tool Managers

### The Problem Worktrees Don't Solve

Two agents in separate worktrees have isolated file trees, but they share the same system tools. If both agents need Node, they both use whatever version is on your system PATH. If one needs Node 18 and the other needs Node 20, you have a conflict.

Tool version managers solve this by letting you pin tool versions per project (or per worktree).

### mise: Tool Version Pinning

mise reads a `.mise.toml` file in your project directory and activates the specified tool versions when you enter that directory. Each worktree can have its own `.mise.toml`.

Create a `.mise.toml` in a worktree:

```bash
cat > ../agent-task-1/.mise.toml << 'EOF'
[tools]
node = "20"
python = "3.12"
EOF
```

When you (or an agent) cd into that directory, mise activates Node 20 and Python 3.12:

```bash
cd ../agent-task-1
mise install
node --version
```

**Expected output:**
```
v20.x.x
```

A different worktree can pin different versions:

```bash
cat > ../agent-task-2/.mise.toml << 'EOF'
[tools]
node = "18"
python = "3.11"
EOF
```

Now each agent gets its own tool versions, activated automatically.

### Nix: Full Environment Reproducibility

Nix goes further than version pinning. Instead of managing individual tool versions, Nix creates a complete, reproducible environment from a `flake.nix` file. Every dependency is pinned to an exact version, and two developers (or two agents) running the same flake get byte-identical environments.

A minimal `flake.nix`:

```nix
{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
  outputs = { nixpkgs, ... }:
    let pkgs = nixpkgs.legacyPackages.x86_64-linux; in {
      devShells.default = pkgs.mkShell {
        packages = [ pkgs.nodejs_20 pkgs.python312 ];
      };
    };
}
```

Running `nix develop` gives you a shell with exactly those tools — nothing more, nothing less. The environment is fully deterministic.

You don't need to learn Nix in depth for this route. The key concept is: Nix provides stronger reproducibility guarantees than mise, at the cost of a steeper learning curve.

### What mise and Nix Are Not

This is important: **mise and Nix are not security sandboxes**. They control which tool versions are active, but they don't restrict what an agent can do. An agent using mise-managed Node can still:
- Access the network
- Read/write files outside the project directory
- Run system commands
- Install packages globally (bypassing mise)

Think of them as **reproducibility tools**, not **containment tools**. They ensure each agent gets the right tools, but they don't limit what the agent can do with those tools.

### Combining Worktrees and Tool Managers

The pattern looks like this:

```
Worktree (code isolation)
  └── .mise.toml or flake.nix (tool version isolation)
       └── Agent works here with isolated code AND isolated tools
```

Each agent gets:
- Its own file tree (worktree)
- Its own tool versions (mise or Nix)
- Its own branch for commits

But the agent still has full system access. For real containment, you need the next layer.

### Exercise 3.1: Per-Worktree Tool Configuration

**Task:** Create two worktrees. In each one, create a `.mise.toml` that pins a different Node version. If you have mise installed, verify that entering each directory activates the correct version.

<details>
<summary>Hint 1</summary>

Create the worktrees with `git worktree add`, then write a `.mise.toml` in each one.
</details>

<details>
<summary>Hint 2</summary>

The `.mise.toml` format for pinning Node:
```toml
[tools]
node = "20"
```
</details>

<details>
<summary>Solution</summary>

```bash
# Create worktrees
git worktree add ../agent-node20 -b feature/node20-task
git worktree add ../agent-node18 -b feature/node18-task

# Configure mise in each
cat > ../agent-node20/.mise.toml << 'EOF'
[tools]
node = "20"
EOF

cat > ../agent-node18/.mise.toml << 'EOF'
[tools]
node = "18"
EOF

# If mise is installed, install and verify
cd ../agent-node20 && mise install && node --version && cd -
cd ../agent-node18 && mise install && node --version && cd -
```

**Expected output:**
```
v20.x.x
v18.x.x
```

If you don't have mise installed, creating the `.mise.toml` files still demonstrates the pattern. An agent entering either directory would pick up the correct tool versions.

```bash
# Clean up
git worktree remove ../agent-node20
git worktree remove ../agent-node18
git branch -D feature/node20-task feature/node18-task
```
</details>

### Checkpoint 3

Before moving on, make sure you understand:
- [ ] Why worktrees alone don't isolate tool versions
- [ ] How mise pins tool versions per directory
- [ ] What Nix adds beyond tool version pinning
- [ ] That mise and Nix are reproducibility tools, not security sandboxes

---

## Section 4: Container-Based Sandboxing

### What Containers Add

Worktrees isolate files. Tool managers isolate tool versions. But an agent can still access the network, consume unlimited CPU and memory, and run commands that affect your system.

Docker containers provide **process-level isolation**: the agent runs inside a sandbox with controlled boundaries for network access, CPU, memory, and filesystem.

This is the first technique in our spectrum that provides actual containment — an agent inside a container genuinely cannot access things outside it (within the constraints you set).

### Key Restrictions

Docker provides four main restrictions for agent sandboxing:

| Restriction | Flag | What It Does |
|---|---|---|
| Network | `--network none` | No network access at all |
| CPU | `--cpus 2` | Limit to 2 CPU cores |
| Memory | `--memory 4g` | Limit to 4GB RAM |
| Filesystem | `--read-only` | Root filesystem is read-only |

Combined with volume mounts, these restrictions create a tight sandbox: the agent can only read and write to the mounted directory.

### Running an Agent Task in a Container

The basic pattern: create a worktree, mount it into a container, run the task, get the results from the worktree after the container exits.

```bash
# Create a worktree for the agent task
git worktree add ../agent-task-1 -b feature/add-auth

# Run the task in a restricted container
docker run --rm \
  --network none \
  --cpus 2 \
  --memory 4g \
  --read-only \
  --tmpfs /tmp \
  -v $(pwd)/../agent-task-1:/workspace:rw \
  -w /workspace \
  node:20 \
  bash -c "echo 'Hello from inside the container' && node --version && ls"
```

**Expected output:**
```
Hello from inside the container
v20.18.0
README.md
```

Breaking down each flag:

- `--rm` — Remove the container when it exits (automatic cleanup)
- `--network none` — No network access; the agent can't download anything or make external requests
- `--cpus 2` — Maximum 2 CPU cores
- `--memory 4g` — Maximum 4GB RAM (killed if exceeded)
- `--read-only` — The container's root filesystem is read-only
- `--tmpfs /tmp` — A writable temporary directory (many tools need this for scratch space)
- `-v .../agent-task-1:/workspace:rw` — Mount the worktree into the container as `/workspace`, read-write
- `-w /workspace` — Set the working directory to `/workspace`
- `node:20` — The Docker image to use
- `bash -c "..."` — The command to run inside the container

### The Network Trade-Off

`--network none` is the safest option: the agent can't download packages, access APIs, or exfiltrate data. But many real tasks require network access.

For example, `npm install` needs to download packages:

```bash
# This will fail — npm can't reach the registry
docker run --rm \
  --network none \
  --read-only \
  --tmpfs /tmp \
  -v $(pwd)/../agent-task-1:/workspace:rw \
  -w /workspace \
  node:20 \
  bash -c "npm install"
```

**Expected output:**
```
npm error code ENOTFOUND
npm error network request to https://registry.npmjs.org/... failed
```

Options for handling this:
- **Pre-install dependencies** before creating the container (run `npm install` in the worktree first)
- **Build a custom Docker image** with dependencies baked in
- **Allow network during setup**, then restrict for execution
- **Use a custom Docker network** that only allows specific destinations

### Filesystem Isolation

With `--read-only` and a single volume mount, the agent can only write to two places:
1. The mounted worktree (`/workspace`)
2. The tmpfs directory (`/tmp`)

Everything else is read-only:

```bash
docker run --rm \
  --read-only \
  --tmpfs /tmp \
  -v $(pwd)/../agent-task-1:/workspace:rw \
  -w /workspace \
  node:20 \
  bash -c "echo 'write to workspace' > /workspace/test.txt && echo 'OK' && echo 'write to root' > /root/test.txt || echo 'BLOCKED (expected)'"
```

**Expected output:**
```
OK
BLOCKED (expected)
```

The write to `/workspace` succeeds because it's a mounted volume. The write to `/root` fails because the root filesystem is read-only.

### Docker-in-Docker Considerations

Sometimes an agent needs to run Docker commands itself (e.g., building a Docker-based project). The simplest approach — mounting the Docker socket — effectively gives the container full Docker access on the host:

```bash
# This works but defeats isolation
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  docker:latest \
  docker ps
```

Mounting the Docker socket means the container can create, start, and stop any container on the host, including privileged ones. For true Docker-in-Docker isolation, look into tools like sysbox or rootless Docker, but those are beyond the scope of this route.

### Exercise 4.1: Run a Restricted Container

**Task:** Create a worktree, then run a command inside a Docker container with `--network none`, `--read-only`, and the worktree mounted as `/workspace`. Have the container create a file in `/workspace`. After the container exits, verify the file exists in the worktree on your host.

<details>
<summary>Hint 1</summary>

Create the worktree first with `git worktree add`, then use `docker run` with the restrictions and a volume mount.
</details>

<details>
<summary>Hint 2</summary>

The volume mount syntax is: `-v $(pwd)/../worktree-path:/workspace:rw`

The container command can be as simple as: `bash -c "echo 'agent output' > /workspace/result.txt"`
</details>

<details>
<summary>Solution</summary>

```bash
# Create worktree
git worktree add ../agent-restricted -b feature/restricted-test

# Run restricted container
docker run --rm \
  --network none \
  --cpus 2 \
  --memory 4g \
  --read-only \
  --tmpfs /tmp \
  -v $(pwd)/../agent-restricted:/workspace:rw \
  -w /workspace \
  node:20 \
  bash -c "echo 'Agent completed task successfully' > /workspace/result.txt && echo 'Done'"
```

**Expected output:**
```
Done
```

Verify the file on the host:
```bash
cat ../agent-restricted/result.txt
```

**Expected output:**
```
Agent completed task successfully
```

Clean up:
```bash
git worktree remove ../agent-restricted
git branch -D feature/restricted-test
```
</details>

### Exercise 4.2: Observe Network Isolation

**Task:** Run two containers: one with `--network none` and one without. In each, try to fetch a URL with `curl`. Compare the results.

<details>
<summary>Hint</summary>

Use the `curlimages/curl` Docker image, which has curl pre-installed. The command is: `curl -s -o /dev/null -w "%{http_code}" https://example.com`
</details>

<details>
<summary>Solution</summary>

```bash
# With network access — should succeed
docker run --rm curlimages/curl \
  curl -s -o /dev/null -w "%{http_code}\n" https://example.com

# Without network access — should fail
docker run --rm --network none curlimages/curl \
  curl -s -o /dev/null -w "%{http_code}\n" https://example.com
```

**Expected output (with network):**
```
200
```

**Expected output (without network):**
```
curl: (6) Could not resolve host: example.com
000
```

The `--network none` container can't resolve DNS or make any network connections.
</details>

### Checkpoint 4

Before moving on, make sure you can:
- [ ] Run a Docker container with network, CPU, memory, and filesystem restrictions
- [ ] Mount a worktree into a container as a writable volume
- [ ] Explain the network isolation trade-off
- [ ] Describe what `--read-only` prevents and what it doesn't (mounted volumes are still writable)

---

## Section 5: Orchestration Patterns

### Running Multiple Agents in Parallel

Once you can isolate a single agent, running multiple agents in parallel is a matter of automation: create a worktree and container for each task, run them simultaneously, and merge the results.

The basic flow:
1. Create a worktree for each task (each on its own feature branch)
2. Start a container for each worktree
3. Wait for all containers to finish
4. Review the results in each worktree
5. Merge branches and resolve any conflicts
6. Clean up worktrees and containers

### Task Distribution Strategies

**One agent per feature branch:** Each agent gets a self-contained feature to implement. Works well when features are independent.

**One agent per module:** Each agent works on a different part of the codebase (frontend, backend, database). Works well when modules have clear boundaries.

**Pipeline:** Agents work in sequence — Agent A creates a foundation, Agent B builds on it, Agent C tests it. Works well for tasks with dependencies.

### Coordination via Git

In this model, agents don't communicate with each other. They coordinate through git:
- Each agent commits to its own branch
- You (or an automated system) merge the branches
- Merge conflicts are resolved the same way as with human developers
- Integration tests run after merging to catch compatibility issues

### An Orchestration Script

Here's a script that creates worktrees, runs containers, waits for results, and cleans up:

```bash
#!/bin/bash
# Run multiple agent tasks in parallel, each in its own worktree + container

TASKS=("add-auth" "fix-tests" "update-docs")

# Create worktrees and start containers
for task in "${TASKS[@]}"; do
  # Create a worktree on a new feature branch
  git worktree add "../agent-${task}" -b "feature/${task}"

  # Run the agent task in a container (background)
  docker run -d \
    --name "agent-${task}" \
    --network none \
    --cpus 1 \
    --memory 2g \
    --read-only \
    --tmpfs /tmp \
    -v "$(pwd)/../agent-${task}":/workspace:rw \
    -w /workspace \
    node:20 \
    bash -c "echo 'Working on ${task}...' && sleep 5 && echo '${task} done!'"
done

echo "All agents started. Waiting for completion..."

# Wait for all containers to finish
for task in "${TASKS[@]}"; do
  docker wait "agent-${task}"
done

echo "All agents completed."

# Check results
for task in "${TASKS[@]}"; do
  echo "=== ${task} ==="
  docker logs "agent-${task}"
  docker rm "agent-${task}"
done

# Cleanup worktrees
for task in "${TASKS[@]}"; do
  git worktree remove "../agent-${task}"
done

echo "Cleanup complete."
```

Walking through the script:
1. It loops through task names, creating a worktree and a detached container (`-d`) for each
2. `docker wait` blocks until each container exits
3. `docker logs` retrieves each container's output
4. `docker rm` removes the stopped containers
5. `git worktree remove` cleans up the worktrees

### Conflict Resolution

When agents work on separate files, merging is straightforward — no conflicts. When they modify the same files, you get merge conflicts, just like with human developers.

Strategies to minimize conflicts:
- Give agents clearly separated areas of the codebase
- Use the "one agent per module" distribution strategy
- Have a final integration step where a human (or another agent) resolves conflicts
- Keep agent tasks small and focused — smaller changes mean fewer conflicts

### Monitoring Agent Progress

While containers are running:

```bash
# See which containers are still running
docker ps

# Follow a specific agent's output in real time
docker logs -f agent-add-auth

# Watch resource usage
docker stats
```

`docker stats` shows a live view of CPU and memory usage per container — useful for spotting agents that are stuck or consuming too many resources.

### Exercise 5.1: Parallel Worktree Tasks

**Task:** Write a shell script that creates two worktrees, simulates an agent task in each (create a file with some content and commit it), and then cleans up. You don't need Docker for this exercise — just demonstrate the parallel worktree pattern.

<details>
<summary>Hint 1</summary>

Use a `for` loop over task names. Inside the loop, create a worktree, make a change, and commit.
</details>

<details>
<summary>Hint 2</summary>

```bash
TASKS=("task-a" "task-b")
for task in "${TASKS[@]}"; do
  # create worktree, make change, commit
done
# then clean up
```
</details>

<details>
<summary>Solution</summary>

```bash
#!/bin/bash

TASKS=("task-a" "task-b")

# Create worktrees and simulate agent work
for task in "${TASKS[@]}"; do
  git worktree add "../agent-${task}" -b "feature/${task}"
  echo "Result of ${task}" > "../agent-${task}/result-${task}.md"
  cd "../agent-${task}"
  git add "result-${task}.md"
  git commit -m "Complete ${task}"
  cd -
done

# Verify branches
echo "Branches:"
git branch

# Verify worktrees
echo "Worktrees:"
git worktree list

# Cleanup
for task in "${TASKS[@]}"; do
  git worktree remove "../agent-${task}"
done

echo "Cleanup complete. Worktrees:"
git worktree list
```

**Expected output:**
```
Preparing worktree (new branch 'feature/task-a')
...
Preparing worktree (new branch 'feature/task-b')
...
Branches:
  feature/task-a
  feature/task-b
* main
Worktrees:
/path/to/my-project       abc1234 [main]
/path/to/agent-task-a     def5678 [feature/task-a]
/path/to/agent-task-b     ghi9012 [feature/task-b]
Cleanup complete. Worktrees:
/path/to/my-project  abc1234 [main]
```
</details>

### Exercise 5.2: Full Orchestration (Requires Docker)

**Task:** Extend the script from Exercise 5.1 to run each task inside a Docker container with `--network none` and resource limits. Use the orchestration script from this section as a starting point.

<details>
<summary>Hint</summary>

Replace the direct file creation and commit steps with a `docker run -d` that does the same work inside a container. Use `docker wait` to wait for completion, and `docker logs` to see the output.
</details>

<details>
<summary>Solution</summary>

```bash
#!/bin/bash

TASKS=("task-a" "task-b")

# Create worktrees and start containers
for task in "${TASKS[@]}"; do
  git worktree add "../agent-${task}" -b "feature/${task}"

  docker run -d \
    --name "agent-${task}" \
    --network none \
    --cpus 1 \
    --memory 2g \
    --read-only \
    --tmpfs /tmp \
    -v "$(pwd)/../agent-${task}":/workspace:rw \
    -w /workspace \
    node:20 \
    bash -c "echo 'Result of ${task}' > result-${task}.md && echo '${task} completed'"
done

echo "Waiting for agents..."

# Wait and collect results
for task in "${TASKS[@]}"; do
  docker wait "agent-${task}"
  echo "=== ${task} ==="
  docker logs "agent-${task}"
  docker rm "agent-${task}"
done

# Verify files were created in worktrees
for task in "${TASKS[@]}"; do
  echo "Contents of ../agent-${task}/result-${task}.md:"
  cat "../agent-${task}/result-${task}.md"
done

# Cleanup
for task in "${TASKS[@]}"; do
  git worktree remove "../agent-${task}"
done

echo "Done."
```

**Expected output:**
```
Preparing worktree (new branch 'feature/task-a')
...
Preparing worktree (new branch 'feature/task-b')
...
Waiting for agents...
=== task-a ===
task-a completed
=== task-b ===
task-b completed
Contents of ../agent-task-a/result-task-a.md:
Result of task-a
Contents of ../agent-task-b/result-task-b.md:
Result of task-b
Done.
```

Note: The container creates files in the worktree via the volume mount, but it doesn't commit them. In a real workflow, you'd commit the changes after reviewing them.
</details>

### Checkpoint 5

Before moving on, make sure you understand:
- [ ] The basic orchestration flow: create, run, wait, review, merge, cleanup
- [ ] Three task distribution strategies (per-branch, per-module, pipeline)
- [ ] How agents coordinate through git without communicating directly
- [ ] How to monitor running containers with `docker ps`, `docker logs`, and `docker stats`

---

## Practice Project

### Project Description

Put everything together: set up a complete isolated agent workspace from scratch. You'll create a worktree, configure tool versions, run a simulated agent task inside a Docker container, verify the results, and clean up.

### Requirements

1. Start from your practice git repository
2. Create a worktree on a new feature branch called `feature/agent-demo`
3. In the worktree, create a `.mise.toml` pinning Node 20
4. Run a simulated agent task inside a Docker container:
   - Mount the worktree as `/workspace`
   - Use `--network none` and `--read-only`
   - The task should create a file in the workspace (e.g., write "Agent task completed" to `result.md`)
5. After the container exits, verify the file exists in the worktree
6. Clean up: remove the worktree

### Getting Started

```bash
# Make sure you're in your practice repo
cd /path/to/agent-sandbox-practice

# Create the worktree
git worktree add ../agent-demo -b feature/agent-demo
```

Now configure the worktree, run the container, verify, and clean up!

### Validation

After completing the project, verify:

```bash
# The file should exist in the worktree
cat ../agent-demo/result.md

# The worktree should be listed
git worktree list

# After cleanup, only the main worktree should remain
git worktree list
```

### If You Don't Have Docker

Skip the Docker container step. Instead, run the simulated task directly in the worktree:

```bash
cd ../agent-demo
echo "Agent task completed" > result.md
cd -
```

You still get the worktree isolation and tool configuration experience. Container isolation is an additional layer you can add later.

---

## The Isolation Spectrum — Summary Table

| Technique | Isolates | Does NOT Isolate | Overhead | Use When |
|---|---|---|---|---|
| **Git worktree** | File changes, branch | System access, network, tools | Low | Any agent task (baseline) |
| **mise** | Tool versions | System access, network, files outside project | Low | Agent needs specific tool versions |
| **Nix flake** | Full dev environment | System access, network | Medium | Exact reproducibility required |
| **Docker container** | Process, network, filesystem, resources | Host kernel (shared) | Medium-High | Agent needs real containment |
| **VM** | Entire OS | Nothing (full isolation) | High | Maximum isolation needed |

In practice, the most common combination is **worktree + Docker container**: the worktree provides the code and branch isolation, the container provides process and network isolation. Add mise or Nix when you need reproducible tool versions across agents.

## Summary

You've learned a spectrum of techniques for isolating AI coding agents:

- **The blast radius problem** motivates isolation — agents with unrestricted access can cause widespread damage from a single mistake
- **Git worktrees** give each agent its own working directory and branch, isolating file changes
- **mise and Nix** pin tool versions per worktree, ensuring reproducible environments (but are not security sandboxes)
- **Docker containers** provide real containment: network, CPU, memory, and filesystem restrictions
- **Orchestration** lets you run multiple isolated agents in parallel, coordinated through git
- **Trade-offs matter** — more isolation means more overhead; match the level to the task

## Next Steps

Now that you understand agent isolation strategies:
- **Start with worktrees** — they're free, fast, and solve the most common problems
- **Add tool pinning** when agents need specific tool versions (take the mise-basics route)
- **Add containers** when you need real containment (take the docker-dev-environments route)
- **Explore your AI tool's built-in isolation** — Claude Code, for example, has native worktree support
- **Experiment with orchestration** — try running parallel agents on a real project

## Additional Resources

- [git worktree documentation](https://git-scm.com/docs/git-worktree): Official git worktree reference
- [mise documentation](https://mise.jdx.dev/): Tool version manager docs
- [Nix flakes](https://nixos.wiki/wiki/Flakes): Nix flake documentation
- [Docker run reference](https://docs.docker.com/engine/reference/run/): Docker container options
- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code): Claude Code's isolation features
