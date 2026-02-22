---
title: Agent Sandboxing - Isolating AI Coding Agents
route_map: /routes/agent-sandboxing/map.md
paired_guide: /routes/agent-sandboxing/guide.md
topics:
  - Agent Isolation
  - Git Worktrees
  - Container Sandboxing
  - Orchestration
  - Development Environments
---

# Agent Sandboxing - Sherpa (AI Teaching Guide)

**Purpose**: This sherpa guide helps AI assistants teach agent isolation strategies effectively through structured interaction. The route is more conceptual than most — it's about combining tools into patterns rather than mastering a single tool.

**Route Map**: See `/routes/agent-sandboxing/map.md` for the high-level overview of this route.
**Paired Guide**: The human-focused content is at `/routes/agent-sandboxing/guide.md`.

---

## Teaching Overview

### Learning Objectives
By the end of this session, the learner should be able to:
- Explain why isolating AI coding agents matters (safety, reproducibility, parallelism)
- Describe the isolation spectrum from nothing to cloud sandboxes
- Use git worktrees to give each agent its own working directory
- Configure mise or Nix to pin tool versions per worktree
- Run agent tasks inside Docker containers with resource and network restrictions
- Orchestrate multiple agents working in parallel with git-based coordination
- Combine techniques into a layered isolation strategy appropriate for their context

### Prerequisites to Verify
Before starting, verify the learner has:
- Basic command line skills (cd, ls, pwd, running programs)
- Working knowledge of git (branching, committing, merging)
- git installed and a repository to practice with

**If prerequisites are missing**: If git knowledge is weak, direct them to the git-basics route first. This route builds heavily on git concepts.

**Optional prerequisites**: mise, Nix, or Docker experience helps for Sections 3-4, but those sections explain enough for a learner to follow along without deep knowledge of any one tool.

### Learner Preferences Configuration

Learners can configure their preferred learning style by creating a `.sherpa-config.yml` file in the repository root (gitignored by default). Configuration options include:

**Teaching Style:**
- `tone`: objective, encouraging, humorous (default: objective and respectful)
- `explanation_depth`: concise, balanced, detailed
- `pacing`: learner-led, balanced, structured

**Assessment Format:**
- `quiz_type`: multiple_choice, explanation, mixed (default: mixed)
- `quiz_frequency`: after_each_section, after_major_topics, end_of_route
- `feedback_style`: immediate, summary, detailed

**Example `.sherpa-config.yml`:**
```yaml
teaching:
  tone: encouraging
  explanation_depth: balanced
  pacing: learner-led

assessment:
  quiz_type: mixed
  quiz_frequency: after_major_topics
  feedback_style: immediate
```

If no configuration file exists, use defaults (objective tone, mixed assessments, balanced pacing).

### Assessment Strategies

Use a combination of assessment types to verify understanding:

**Multiple Choice Questions:**
- Present 3-4 answer options
- Include one correct answer and plausible distractors
- Good for checking factual knowledge about isolation techniques
- Example: "Which isolation technique restricts an agent's network access? A) Git worktrees B) mise C) Docker containers D) Nix flakes"

**Explanation Questions:**
- Ask learner to explain concepts in their own words
- Assess deeper understanding of trade-offs and when to apply each technique
- Example: "When would you choose a Docker container over a plain git worktree for agent isolation?"

**Mixed Approach (Recommended):**
- Use multiple choice for quick checks on specific commands and concepts
- Use explanation questions for trade-off analysis and strategy decisions
- Adapt based on learner responses and confidence level

---

## Teaching Flow

### Introduction

**What to Cover:**
- AI coding agents can modify files, run commands, install packages, and make network requests
- Without isolation, an agent's mistakes affect your entire development environment
- Isolation is about controlling blast radius — how much damage a misbehaving agent can cause
- There's a spectrum of isolation techniques with different trade-offs between safety and overhead

**Opening Questions to Assess Level:**
1. "Have you used AI coding agents (Claude Code, Cursor, Copilot) before? What kinds of tasks do you give them?"
2. "Have you ever had an agent make changes you didn't want — files modified, wrong packages installed, things broken?"
3. "Are you familiar with git worktrees or Docker? Even conceptually?"

**Adapt based on responses:**
- If experienced with agents: Focus on the failure modes they've seen, move quickly to solutions
- If new to agents: Spend more time on the "why" before the "how", use concrete scenarios
- If already using Docker: Draw on their container knowledge, go deeper on orchestration
- If already using worktrees: Acknowledge their experience, focus on combining with other tools

**Good opening framing:**
"When you give an AI agent access to your codebase, you're giving it the ability to modify files, run commands, and potentially break things. The question isn't whether something will go wrong — it's how much damage it can do when it does. That's what isolation controls."

---

### Section 1: Why Agent Isolation Matters

**Core Concept to Teach:**
AI coding agents operate with real filesystem and shell access. Without isolation, a single agent mistake can affect your entire development environment, break other projects, or conflict with work you're doing manually. Isolation is about controlling blast radius through layered techniques.

**How to Explain:**

1. Start with the blast radius problem:
   "An AI agent working in your main checkout has access to everything. If it modifies the wrong file, installs a conflicting package, or runs a destructive command, it affects your entire environment. The 'blast radius' is everything it can reach."

2. Present the isolation spectrum as a table or list:
   ```
   Isolation Level         | What It Isolates        | Overhead
   ─────────────────────────────────────────────────────────────
   Nothing (main checkout) | Nothing                 | None
   Git worktree            | File changes            | Low
   Tool pinning (mise)     | Tool versions           | Low
   Nix flake               | Full environment        | Medium
   Docker container        | Process, network, fs    | Medium-High
   VM                      | Entire OS               | High
   Cloud sandbox           | Entire machine          | Highest
   ```

3. Explain the three motivations for isolation:
   - **Safety**: Limit what an agent can break
   - **Reproducibility**: Each agent gets the same environment every time
   - **Parallelism**: Multiple agents work simultaneously without conflicts

**Discussion Points:**
- "Have you ever had two terminal sessions where you were in the middle of something and accidentally ran a command in the wrong one? Agents have the same problem, but faster."
- "If you're running three agents at once, what happens if they all try to modify `package.json`?"

**Common Misconceptions:**
- Misconception: "Isolation is only needed for untrusted agents" — Clarify: "Even a trusted, well-intentioned agent can make mistakes. Isolation protects against accidents, not just malice. It's like wearing a seatbelt — you don't expect to crash, but you prepare for it."
- Misconception: "More isolation is always better" — Clarify: "Each layer of isolation adds complexity and overhead. A git worktree is free; a cloud VM costs money and setup time. Match the isolation level to the risk."
- Misconception: "I only need isolation if I'm running agents in production" — Clarify: "Development is where agents do the most file modification. Isolation matters most during active development."

**Verification Questions:**
1. "What is 'blast radius' in the context of agent isolation?"
2. "Name the three reasons we isolate agents."
3. Multiple choice: "An agent running in your main git checkout modifies the wrong file. What's the blast radius? A) Just that file B) The entire repository C) The entire filesystem D) It depends on what permissions the agent has"

**Good answer indicators:**
- They understand blast radius as the scope of potential damage
- They can name safety, reproducibility, and parallelism
- They answer D (permissions determine blast radius — the agent can reach anything its user can)

**If they struggle:**
- Use a concrete scenario: "Imagine you're working on a feature in your main branch. You ask an agent to refactor a module. It modifies 15 files, breaks the build, and you have uncommitted changes mixed in. How do you untangle that?"
- The scenario usually clicks — they've been there

---

### Section 2: Code Isolation with Git Worktrees

**Core Concept to Teach:**
Git worktrees let you check out multiple branches simultaneously in separate directories, all sharing a single `.git` repository. This gives each agent its own file tree while keeping full git history and branching available.

**How to Explain:**

1. Start with the problem worktrees solve:
   "Normally, git gives you one working directory. If you want to work on two branches at once, you either stash and switch, or clone the repo again. Worktrees give you additional working directories that share the same repository."

2. Show the mental model:
   ```
   my-project/                  ← main worktree (your normal checkout)
   my-project/.git/             ← shared git data
   ../agent-task-1/             ← worktree on feature/add-auth branch
   ../agent-task-2/             ← worktree on feature/fix-tests branch
   ```

3. Explain why this works for agents:
   "Each agent gets its own directory with its own branch. They can modify files, commit, and even break things — and none of it touches your main working directory. When they're done, you merge their branch like any other."

**Walk Through Together:**

First, make sure they have a git repository to work with. If not, create a practice one:
```bash
mkdir practice-repo && cd practice-repo
git init
echo "# Practice" > README.md
git add README.md
git commit -m "Initial commit"
```

Create a worktree:
```bash
git worktree add ../agent-task-1 -b feature/add-auth
```

Explain: "This creates a new directory `../agent-task-1` checked out to a new branch `feature/add-auth`. It's a full working directory — you can cd into it, edit files, run commands."

List worktrees:
```bash
git worktree list
```

Show it's a real working directory:
```bash
ls ../agent-task-1
```

Make changes in the worktree:
```bash
echo "auth module" > ../agent-task-1/auth.md
cd ../agent-task-1
git add auth.md
git commit -m "Add auth module"
cd -
```

Show that the main worktree is unaffected:
```bash
ls  # No auth.md here
git log --oneline  # No "Add auth module" commit on this branch
```

Clean up:
```bash
git worktree remove ../agent-task-1
```

**Claude Code's Worktree Support:**
"Claude Code already uses this pattern. When you ask it to work on a task, it can create a worktree automatically so it works in isolation from your main checkout. This is configured in its settings and happens transparently."

**Common Misconceptions:**
- Misconception: "A worktree is like a git clone" — Clarify: "A clone copies the entire repository. A worktree shares the same `.git` directory — there's only one repository, just multiple checkouts of it. This means branches created in one worktree are immediately visible in others."
- Misconception: "Two worktrees can be on the same branch" — Clarify: "No — each worktree must be on a different branch. This is actually a feature for agent isolation: it prevents two agents from stepping on each other's changes."
- Misconception: "Worktrees protect against destructive commands" — Clarify: "Worktrees isolate file changes, not system access. An agent in a worktree can still run arbitrary commands, install packages globally, or access the network. Worktrees are the first layer of isolation, not the last."

**Verification Questions:**
1. "What does a git worktree share with the main checkout, and what's separate?"
2. "Why can't two worktrees be on the same branch?"
3. Multiple choice: "You create a worktree with `git worktree add ../task -b feature/x`. An agent working in `../task` commits changes. Where do those commits exist? A) Only in the `../task` directory B) In the shared git repository, on the `feature/x` branch C) In both the main checkout and the worktree D) Nowhere until you push"

**Good answer indicators:**
- They understand shared `.git` but separate working directories
- They know the branch exclusivity rule
- They answer B (commits go to the shared repository on the feature branch)

**If they struggle:**
- Draw the diagram: one `.git` directory, multiple working directories pointing to it
- Compare to having multiple terminal windows open to the same project — except each window sees a different branch

**Exercise 2.1:**
"Create three worktrees from your practice repository, each on a different feature branch. Make a change in each one and commit it. Then go back to your main worktree and verify you can see all three branches in `git branch`."

**How to Guide Them:**
1. "Use `git worktree add` three times with different paths and branch names"
2. If stuck: "Remember the syntax: `git worktree add <path> -b <branch-name>`"
3. After creating: "Now `cd` into each one, make a change, commit it, then come back"
4. Verify: "Run `git branch` in your main worktree — you should see all three feature branches"

**Exercise 2.2:**
"Clean up all three worktrees using `git worktree remove`. Then verify they're gone with `git worktree list` and optionally delete the feature branches with `git branch -d`."

**How to Guide Them:**
1. "Use `git worktree remove <path>` for each one"
2. "Then `git worktree list` should only show your main worktree"
3. "The branches still exist — `git branch` will show them. You can delete them if you want"

---

### Section 3: Environment Isolation with Dev Tool Managers

**Core Concept to Teach:**
Worktrees isolate file changes, but not the development environment. Two agents in different worktrees still share the same system Node, Python, etc. Tool version managers like mise and Nix let you pin tool versions per project (or per worktree), so each agent gets a consistent, independent environment.

**Important caveat to communicate:**
"mise and Nix are convenience tools, not security sandboxes. They control which versions of tools are available — they don't prevent an agent from accessing the rest of your system. Think of them as reproducibility tools, not containment tools."

**How to Explain:**

1. Start with the problem:
   "You have two agents working in two worktrees. One needs Node 18 for a legacy project, the other needs Node 20. If they're sharing the system Node, they'll conflict. Tool version managers solve this."

2. Explain mise:
   "mise reads a `.mise.toml` file in your project directory and activates the right tool versions when you cd into that directory. Each worktree can have its own `.mise.toml`."

3. Explain Nix (briefly):
   "Nix goes further — instead of just managing tool versions, it creates a complete, reproducible environment defined in a `flake.nix` file. Every dependency is pinned. Two developers (or two agents) running the same flake get byte-identical environments."

4. Show how they combine with worktrees:
   "Worktree = isolated files. mise/Nix = isolated tools. Together, each agent gets its own code AND its own development environment."

**Walk Through Together — mise:**

In a worktree, create a `.mise.toml`:
```bash
cd ../agent-task-1
cat > .mise.toml << 'EOF'
[tools]
node = "20"
python = "3.12"
EOF
```

Explain: "When you (or an agent) cd into this directory, mise activates Node 20 and Python 3.12. Other worktrees can pin different versions."

If they have mise installed, have them verify:
```bash
mise install
node --version   # v20.x.x
python --version # Python 3.12.x
```

**Walk Through Together — Nix (conceptual):**

Show a minimal `flake.nix`:
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

Explain: "Running `nix develop` in a directory with this flake gives you a shell with exactly Node 20 and Python 3.12 — nothing more, nothing less. It's fully reproducible."

If the learner doesn't have Nix installed, this section can stay conceptual. The key point is the idea, not the syntax.

**Combining the Layers:**
```
Worktree (code isolation)
  └── .mise.toml or flake.nix (tool version isolation)
       └── Agent works here with isolated code AND isolated tools
```

"Each agent gets its own files (worktree) and its own tool versions (mise or Nix). But the agent can still access the network, write outside the worktree directory, and run any system command. For full containment, you need the next layer."

**Common Misconceptions:**
- Misconception: "mise or Nix sandboxes the agent" — Clarify: "They control which tool versions are active, not what the agent can access. An agent using mise can still run `rm -rf /` if it wants to. These are reproducibility tools, not security boundaries."
- Misconception: "I need both mise AND Nix" — Clarify: "Pick one. mise is simpler and good enough for most cases. Nix is more powerful but has a steeper learning curve. Use Nix when you need exact reproducibility across teams or CI."
- Misconception: "I need to learn Nix in depth for this route" — Clarify: "Understanding what Nix does is enough. You don't need to master Nix syntax to understand its role in agent isolation."

**Verification Questions:**
1. "What's the difference between what worktrees isolate and what mise/Nix isolate?"
2. "Are mise and Nix security sandboxes? Why or why not?"
3. Multiple choice: "Agent A needs Node 18, Agent B needs Node 20. They're in separate worktrees. Without a tool version manager, what happens? A) They each get the version they need automatically B) They both use whatever Node is on the system PATH C) Git resolves the conflict D) The second agent's install overwrites the first"

**Good answer indicators:**
- They understand worktrees = file isolation, mise/Nix = tool isolation
- They know these are NOT security boundaries
- They answer B (without a tool manager, both agents get the system version)

**If they struggle:**
- Focus on mise only — it's simpler and demonstrates the concept
- Use the analogy: "Worktrees give each agent their own desk. mise gives each agent their own toolbox. But they're all still in the same room."

**Exercise 3.1:**
"Create two worktrees. In each one, create a `.mise.toml` that pins a different Node version. Verify that cd-ing into each worktree activates the correct version."

**How to Guide Them:**
1. "Create two worktrees like we did before"
2. "In each one, write a `.mise.toml` with a different Node version"
3. "Run `mise install` in each, then check `node --version`"
4. If they don't have mise: "That's fine — write the `.mise.toml` files anyway so you understand the pattern. We'll move on to Docker next."

---

### Section 4: Container-Based Sandboxing

**Core Concept to Teach:**
Docker containers provide process-level isolation: restricted network access, CPU/memory limits, filesystem boundaries, and separate process namespaces. This is the first technique in our spectrum that provides actual containment — an agent inside a container genuinely cannot access things outside it (within the constraints you set).

**How to Explain:**

1. Start with what containers add:
   "Worktrees isolate files. mise isolates tool versions. But an agent can still access the network, consume unlimited CPU, or run commands that affect your system. Docker containers put the agent in a box with controlled walls."

2. Explain the key restrictions:
   - `--network none`: No network access at all
   - `--cpus 2`: Limit to 2 CPU cores
   - `--memory 4g`: Limit to 4GB of RAM
   - `--read-only`: Root filesystem is read-only (agent can't install globally)
   - `--tmpfs /tmp`: Writable /tmp for scratch space
   - `-v path:/workspace:rw`: Mount a worktree as the only writable directory

3. Show the pattern:
   "You create a worktree, then mount it into a container. The agent works inside the container but the changes appear in the worktree. When the container exits, you still have the worktree with all the agent's changes — ready to review and merge."

**Walk Through Together:**

Run a simple restricted container:
```bash
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

Explain each flag:
- `--rm`: Remove the container when it exits (cleanup)
- `--network none`: No network access — the agent can't download anything or phone home
- `--cpus 2`: Maximum 2 CPU cores
- `--memory 4g`: Maximum 4GB RAM
- `--read-only`: The container's root filesystem is read-only
- `--tmpfs /tmp`: A writable temporary directory (needed for many tools)
- `-v .../agent-task-1:/workspace:rw`: Mount the worktree as the workspace
- `-w /workspace`: Set the working directory
- `node:20`: Use the Node 20 Docker image
- `bash -c "..."`: The command to run

"The agent can read and write files in `/workspace` (the mounted worktree) and `/tmp` (temporary storage). It cannot access the network, cannot write anywhere else, and is limited to 2 CPUs and 4GB RAM."

**A More Realistic Example:**
```bash
docker run --rm \
  --network none \
  --cpus 2 \
  --memory 4g \
  --read-only \
  --tmpfs /tmp \
  --tmpfs /root \
  -v $(pwd)/../agent-task-1:/workspace:rw \
  -w /workspace \
  node:20 \
  bash -c "npm install && npm test"
```

Point out: "Wait — `npm install` needs network access to download packages. With `--network none`, this will fail unless `node_modules` is already populated. This is a real trade-off: full network isolation means you need to pre-install dependencies or use a pre-built image."

**Discuss the network trade-off:**
"In practice, you often need a middle ground. Options include:
- Pre-install dependencies before starting the container
- Build a custom Docker image with dependencies baked in
- Allow network access during setup, then restrict it during execution
- Use `--network` with a custom Docker network that only allows specific destinations"

**Docker-in-Docker Considerations:**
"Sometimes an agent needs to run Docker commands itself (e.g., to build and test a Docker-based project). Running Docker inside Docker is possible but adds complexity. The simplest approach is mounting the Docker socket (`-v /var/run/docker.sock:/var/run/docker.sock`), but this effectively gives the container full Docker access on the host — which defeats much of the isolation. For true Docker-in-Docker isolation, look at tools like sysbox or rootless Docker."

**Common Misconceptions:**
- Misconception: "Containers are always the right answer" — Clarify: "Containers add real overhead: startup time, image management, volume mounting complexity. For a quick 'fix this typo' task, a worktree is faster and good enough."
- Misconception: "`--network none` is always appropriate" — Clarify: "Many agent tasks require network access — installing packages, pulling dependencies, accessing APIs. Fully offline containers work for tasks that only modify code, but many real tasks need some network access."
- Misconception: "Mounting the Docker socket is fine for isolation" — Clarify: "Mounting the Docker socket gives the container full control over Docker on the host. It's convenient but it's a large escape hatch from your isolation."

**Verification Questions:**
1. "What are the four main resource restrictions Docker provides for agent sandboxing?"
2. "Why might `--network none` be too restrictive for some agent tasks?"
3. Multiple choice: "You mount a worktree into a container with `-v ./task:/workspace:rw`. The agent creates a file at `/workspace/result.txt`. Where does that file actually exist? A) Only inside the container B) On the host at `./task/result.txt` C) Both in the container and on the host D) It's deleted when the container exits"

**Good answer indicators:**
- They can name: network, CPU, memory, filesystem restrictions
- They understand the network trade-off
- They answer B/C (the volume mount means the file exists on the host — B and C are both acceptable answers, with C being more precise since it exists in both places while the container is running)

**If they struggle:**
- Focus on the analogy: "A container is like a guest room with controlled access. The guest can use the room (the mounted volume) but can't wander into the rest of the house (the host filesystem), can't use the phone (`--network none`), and has limited power and water (`--cpus`, `--memory`)."
- If Docker is unfamiliar: "Don't worry about memorizing the flags. The key insight is that containers give you real boundaries — a process inside a container genuinely cannot reach things outside it."

**Exercise 4.1:**
"Run a simple command inside a Docker container with `--network none`, `--read-only`, and a worktree mounted as the workspace. Verify that the container can write to the mounted directory but not to the root filesystem."

**How to Guide Them:**
1. "First, create a worktree to use as the workspace"
2. "Run: `docker run --rm --network none --read-only --tmpfs /tmp -v $(pwd)/../task:/workspace:rw -w /workspace node:20 bash -c \"echo hello > /workspace/test.txt && echo 'Write to workspace succeeded' && echo fail > /root/test.txt || echo 'Write to root filesystem failed (expected)'\"`"
3. "Check that `test.txt` exists in the worktree directory on your host"
4. If they don't have Docker: "That's fine — the conceptual understanding is what matters. We'll focus on the patterns rather than running every command."

**Exercise 4.2:**
"Try running `npm install` inside a container with `--network none`. Observe the failure. Then run it without `--network none` and compare."

**How to Guide Them:**
1. "Create a worktree with a `package.json` that has at least one dependency"
2. "Run the container with `--network none` and try `npm install` — it should fail"
3. "Run it again without `--network none` — it should succeed"
4. "This demonstrates the trade-off: network isolation prevents dependency installation"

---

### Section 5: Orchestration Patterns

**Core Concept to Teach:**
Once you can isolate a single agent, the next step is running multiple agents in parallel on different tasks. Git provides the coordination mechanism: each agent works on its own branch in its own worktree, and you merge the results.

**How to Explain:**

1. Start with the motivation:
   "If one isolated agent can work on one task, why not run three agents on three tasks simultaneously? Each gets its own worktree and container. They all commit to their own branches. When they're done, you review and merge."

2. Explain task distribution strategies:
   - **One agent per feature branch**: Each agent gets a self-contained feature
   - **One agent per module**: Each agent works on a different part of the codebase
   - **Pipeline**: Agents work in sequence, each building on the previous agent's output

3. Explain coordination via git:
   "Agents don't talk to each other. They coordinate through git. Each one commits to its own branch. You (or an automated system) merge the branches, resolve conflicts, and run integration tests."

**Walk Through Together — A Simple Orchestration Script:**

```bash
#!/bin/bash
# Run multiple agent tasks in parallel, each in its own worktree + container

TASKS=("add-auth" "fix-tests" "update-docs")

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
```

Walk through the script:
- "First, it creates a worktree and a container for each task"
- "Each container runs in the background (`-d` flag)"
- "Then it waits for all containers to finish"
- "Finally, it checks the logs and cleans up"

**Conflict Resolution:**
"When agents work on separate files, merging is straightforward. When they touch the same files, you'll get merge conflicts — just like with human developers. Strategies:
- Give agents clearly separated areas of the codebase
- Have a final integration step where a human (or another agent) resolves conflicts
- Use feature flags so independent features don't interfere at the code level"

**Monitoring Agent Progress:**
"With `docker logs -f agent-name` you can follow an agent's output in real time. For a dashboard view, use `docker ps` to see which containers are still running and their resource usage with `docker stats`."

**Common Misconceptions:**
- Misconception: "Agents can coordinate with each other during tasks" — Clarify: "In this model, agents are isolated from each other. They don't share state or communicate. Coordination happens through git after they finish."
- Misconception: "More agents always means faster results" — Clarify: "Agent tasks need to be independent enough to parallelize. If every task depends on the output of another task, running them in parallel doesn't help."
- Misconception: "Merge conflicts from agents are different from human merge conflicts" — Clarify: "They're exactly the same. Git doesn't know or care whether a human or an agent made the changes."

**Verification Questions:**
1. "In the orchestration model, how do agents coordinate with each other?"
2. "What are the three task distribution strategies we discussed?"
3. Multiple choice: "Two agents are working in parallel. Agent A modifies `auth.js`, Agent B modifies `auth.js`. What happens when you try to merge both branches? A) Git automatically picks the better change B) You get a merge conflict C) The second merge overwrites the first D) Git refuses to merge"

**Good answer indicators:**
- They understand agents don't communicate — they coordinate through git
- They can name per-branch, per-module, and pipeline strategies
- They answer B (merge conflict, just like with human developers)

**If they struggle:**
- Relate it to human team workflows: "It's the same as two developers working on separate branches and merging. The tools are identical."
- If the scripting feels overwhelming: "The script is just automating what you'd do manually: create worktrees, start containers, wait, check results, clean up. Each step is something you already know."

**Exercise 5.1:**
"Write a shell script that creates two worktrees, runs a simulated task in each (just a shell command that creates a file), waits for both to complete, and cleans up the worktrees."

**How to Guide Them:**
1. "Start with the loop: `for task in task-a task-b; do ...`"
2. "Inside the loop: `git worktree add`, then make a change, then commit"
3. "After the loop: clean up with `git worktree remove`"
4. "You can skip Docker for this exercise — just focus on the worktree orchestration"

---

## Practice Project

**Project Introduction:**
"Let's put it all together. You'll set up a complete isolated agent workspace: create a worktree, configure tool versions, run a simulated agent task in a Docker container, and clean up when done."

**Requirements:**
Present one at a time:
1. "Start from any git repository (your practice repo is fine)"
2. "Create a worktree for an 'agent task' on a new feature branch"
3. "In the worktree, create a `.mise.toml` pinning at least one tool version"
4. "Run a simulated agent task inside a Docker container with the worktree mounted — the task should create or modify a file in the workspace"
5. "Use `--network none` and `--read-only` restrictions on the container"
6. "Verify the agent's changes exist in the worktree after the container exits"
7. "Clean up: remove the container (if not `--rm`) and the worktree"

**Scaffolding Strategy:**
- Let them work independently first
- Check in after worktree creation: "Got the worktree set up on a feature branch?"
- Check in after mise config: "What tools did you pin?"
- After container run: "Did the container's changes show up in the worktree?"
- Final check: "Everything cleaned up? `git worktree list` should only show your main worktree"

**Checkpoints During Project:**
- After worktree creation: Verify with `git worktree list`
- After mise config: Check `.mise.toml` exists in the worktree
- After container run: Verify the expected file exists in the worktree
- After cleanup: Verify worktree is removed, container is gone

**If They Get Stuck:**
- "Which step are you on? Worktree, mise, Docker, or cleanup?"
- "What error are you seeing?"
- If Docker is the blocker: "Skip Docker and just run the simulated task directly in the worktree. The isolation concepts still apply."

**If They Don't Have Docker:**
- "Run the simulated task directly in the worktree instead. The practice project works without Docker — you just lose the container isolation layer. The worktree + mise layers still demonstrate the pattern."

**Extension Ideas if They Finish Early:**
- "Write an orchestration script that runs three agent tasks in parallel, each in its own worktree + container"
- "Add resource monitoring: use `docker stats` to watch CPU/memory usage of your containers"
- "Try building a custom Docker image with pre-installed dependencies to avoid the `--network none` limitation"
- "Explore your AI coding tool's built-in isolation features — does it support worktrees? Custom containers?"

---

## Wrap-Up and Next Steps

**Review Key Takeaways:**
"Let's review the isolation spectrum we covered:"
- **Worktrees** isolate file changes — each agent gets its own working directory and branch
- **mise/Nix** isolate tool versions — each agent gets consistent, pinned tools
- **Docker containers** isolate processes — network, CPU, memory, and filesystem boundaries
- **Orchestration** lets you run multiple isolated agents in parallel, coordinated through git
- More isolation means more safety but also more overhead — match the level to the task

**Ask them to explain one concept:**
"If a colleague asked you 'how should I set up my environment for running AI coding agents safely?', what would you tell them?"
(This reinforces the practical application and tests whether they can synthesize the techniques)

**Assess Confidence:**
"How confident do you feel about setting up isolated agent workspaces for your own projects?"

**Respond based on answer:**
- Low confidence: "Start with just worktrees — they're free and solve the most common problems. Add layers as you need them."
- Medium confidence: "You've got the concepts. Try it on a real project — set up a worktree for your next agent task and see how it feels."
- High confidence: "Great. Try orchestrating parallel agents on a real project. The orchestration script from Section 5 is a good starting point."

**Suggest Next Steps:**
Based on their progress and interests:
- "For deeper tool management: Take the mise-basics or nix-dev-environments routes"
- "For Docker mastery: Take the docker-dev-environments route"
- "For immediate practice: Set up worktrees for your next AI coding session"
- "For team workflows: Write an orchestration script tailored to your project"

**Encourage Questions:**
"Do you have any questions about anything we covered?"
"What isolation level makes sense for your current projects?"
"Is there a specific tool or pattern you want to explore further?"

---

## Adaptive Teaching Strategies

### If Learner is Struggling

**Signs:**
- Confused about what each isolation layer does vs. the others
- Overwhelmed by the number of tools (git, mise, Nix, Docker)
- Can't see why they'd need isolation for their workflow

**Strategies:**
- Focus on the spectrum — don't try to teach every tool in depth
- Start with worktrees only and make sure they're solid before moving on
- Use concrete failure scenarios: "Your agent just deleted your uncommitted changes. How would worktrees have prevented that?"
- Skip Nix entirely if they don't have it installed — mise is simpler and demonstrates the same concept
- If Docker is unfamiliar, keep it conceptual — the pattern matters more than the flags
- Reduce the practice project to just worktree creation and cleanup

### If Learner is Excelling

**Signs:**
- Completes exercises quickly
- Asks about production use cases or CI integration
- Already using some of these techniques

**Strategies:**
- Move faster through basics, focus on orchestration and trade-offs
- Discuss advanced Docker features: custom networks, seccomp profiles, user namespaces
- Explore Nix in more depth if they're interested
- Challenge: "Design an isolation strategy for your actual project — what layers would you use and why?"
- Discuss how CI/CD pipelines already implement many of these patterns
- Introduce the idea of ephemeral development environments (Gitpod, Codespaces) as the cloud extreme of the spectrum

### If Learner Seems Disengaged

**Signs:**
- Short responses
- Not asking questions
- Seems to find it too theoretical

**Strategies:**
- Get hands-on immediately — skip theory, go straight to creating worktrees
- Connect to their real work: "What AI coding tool do you use? Let's set up isolation for it"
- Show a concrete disaster scenario: "Here's what happens when two agents modify the same file without isolation"
- Focus on the practical — less about Docker flags, more about the workflow pattern
- Ask what they'd like to spend more time on

### Different Learning Styles

**Visual learners:**
- Draw the isolation spectrum as concentric circles or stacked layers
- Show the worktree directory structure as a tree diagram
- Diagram the orchestration flow: create → run → wait → merge → cleanup

**Hands-on learners:**
- Skip ahead to exercises, learn by doing
- "Create a worktree, break something in it, and show me that the main checkout is fine"
- Let them build the orchestration script incrementally

**Conceptual learners:**
- Discuss the theory behind isolation: process namespaces, cgroups, filesystem layers
- Compare to other isolation models: VMs, cloud functions, WebAssembly
- Talk about the security model and its limitations

---

## Troubleshooting Common Issues

### Git Worktree Issues

**"fatal: 'branch' is already checked out" Error:**
- Another worktree is already on that branch
- List worktrees: `git worktree list` to find which one
- Use a different branch name, or remove the conflicting worktree first

**Worktree directory already exists:**
- The path you specified already has files in it
- Choose a different path or remove the existing directory first

**Can't remove a worktree:**
- Make sure you're not cd'd into it
- Use `git worktree remove --force` if there are uncommitted changes (be careful — this discards changes)

### mise Issues

**mise not activating in a worktree:**
- Make sure mise is initialized in your shell (check `.bashrc` or `.zshrc`)
- Run `mise install` in the worktree to install the pinned versions
- Verify the `.mise.toml` file exists and has correct syntax

### Docker Issues

**Permission denied mounting a volume:**
- On Linux, check file ownership — the container user may not have access
- Try adding `--user $(id -u):$(id -g)` to match your host user
- On macOS with Docker Desktop, make sure the path is in Docker's file sharing settings

**`--network none` causing failures:**
- Many tools need network access for dependency installation
- Options: pre-install dependencies, use a custom image, or allow network during setup
- For agent tasks that only modify code (no installs), `--network none` works well

**Container can't write to mounted volume:**
- Check the volume mount flag: should be `:rw` not `:ro`
- Check that the directory exists on the host
- With `--read-only`, only mounted volumes and tmpfs directories are writable

**Docker not installed:**
- macOS: Install Docker Desktop
- Linux: Install via your distribution's package manager or Docker's official install script
- If Docker isn't available, the worktree + mise concepts still apply — container isolation is an additional layer, not a prerequisite

### Orchestration Issues

**Containers started but nothing happened:**
- Check container logs: `docker logs agent-name`
- Make sure the command in the container is correct
- Verify the volume mount path is right

**Worktree cleanup fails after container use:**
- Make sure the container has exited first: `docker ps` to check
- Remove the container before removing the worktree
- Check for leftover files created by the container with different ownership

---

## Teaching Notes

**Key Emphasis Points:**
- The isolation spectrum is the central mental model — everything hangs on it
- Worktrees are the foundation — make sure this section is solid
- Trade-offs matter more than tools — knowing when NOT to use a heavy technique is as important as knowing how
- This route is about patterns, not mastering any single tool

**Pacing Guidance:**
- Don't rush Section 1 (the "why") — if they don't understand the problem, the solutions feel unmotivated
- Section 2 (worktrees) is the core skill — give it the most time
- Section 3 (mise/Nix) can be lighter if they lack the prerequisites — the concept is what matters
- Section 4 (Docker) can stay conceptual if they don't have Docker installed
- Section 5 (orchestration) is advanced — okay to skim if time is short

**Success Indicators:**
You'll know they've got it when they:
- Can explain why a bare agent in a main checkout is risky
- Choose the right isolation level for a given scenario (not always the heaviest)
- Can set up a worktree and articulate what it isolates (and what it doesn't)
- Understand the trade-offs between convenience and safety
- Start thinking about how to apply this to their own agent workflows

**Most Common Confusion Points:**
1. **What each layer isolates**: Worktrees isolate files, mise isolates tools, Docker isolates processes
2. **Security vs convenience**: mise and Nix are not security tools
3. **Network isolation trade-offs**: `--network none` breaks dependency installation
4. **Worktrees vs clones**: Worktrees share the `.git` directory, clones don't

**Teaching Philosophy:**
- This route is more about judgment than skill — knowing when to use each technique
- The isolation spectrum is a thinking tool, not a prescription
- Start simple (worktrees), add complexity only when the learner sees the need
- Real-world trade-offs are more important than perfect isolation
- Not everyone needs Docker containers — sometimes a worktree is enough
