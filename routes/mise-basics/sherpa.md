---
title: mise Basics - Dev Tool Version Management
route_map: /routes/mise-basics/map.md
paired_guide: /routes/mise-basics/guide.md
topics:
  - mise
  - Version Management
  - Environment Variables
  - Task Running
---

# mise Basics - Sherpa (AI Teaching Guide)

**Purpose**: This sherpa guide helps AI assistants teach mise fundamentals effectively through structured interaction.

**Route Map**: See `/routes/mise-basics/map.md` for the high-level overview of this route.
**Paired Guide**: The human-focused content is at `/routes/mise-basics/guide.md`.

---

## Teaching Overview

### Learning Objectives
By the end of this session, the learner should be able to:
- Explain what mise replaces and why a unified tool manager matters
- Install mise and activate it in their shell
- Install and manage tool versions for any language
- Configure per-project tool versions with `.mise.toml`
- Set project-scoped environment variables
- Define and run tasks

### Prerequisites to Verify
Before starting, verify the learner has:
- Basic command line skills (cd, ls, running programs)
- A shell they're comfortable with (zsh, bash, fish)

**If prerequisites are missing**: Provide a quick primer on terminal basics before continuing.

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
- Good for checking factual knowledge quickly
- Example: "What command sets a tool version AND writes it to .mise.toml? A) mise install B) mise use C) mise set D) mise activate"

**Explanation Questions:**
- Ask learner to explain concepts in their own words
- Assess deeper understanding and ability to apply knowledge
- Example: "What's the difference between `mise install node@20` and `mise use node@20`?"

**Mixed Approach (Recommended):**
- Use multiple choice for quick checks after introducing new commands
- Use explanation questions for concepts like version resolution order
- Adapt based on learner responses and confidence level

---

## Teaching Flow

### Introduction

**What to Cover:**
- mise is a polyglot dev tool version manager — it replaces nvm, pyenv, rbenv, and similar tools
- It also manages environment variables (like direnv) and runs tasks (like Make or npm scripts)
- Configuration lives in `.mise.toml` — one file per project, checked into version control
- mise uses PATH manipulation, not shims (this makes it faster than asdf)

**Opening Questions to Assess Level:**
1. "Have you used any version managers before — nvm, pyenv, rbenv, asdf?"
2. "How do you currently handle different projects needing different Node or Python versions?"
3. "Do you use any tools for project-specific environment variables, like direnv or .env files?"

**Adapt based on responses:**
- If experienced with nvm/pyenv: Draw direct comparisons, emphasize unification and `.mise.toml`
- If experienced with asdf: Focus on differences — TOML config instead of `.tool-versions`, PATH manipulation instead of shims, built-in env vars and tasks
- If complete beginner to version management: Start with the problem — why you can't just have one global version of everything
- If they use direnv: Show how `[env]` in `.mise.toml` replaces `.envrc`

**Good opening analogy:**
"Imagine if every programming language had its own package manager, its own config file, its own shell hooks, and its own way of switching versions. That's actually where we are — nvm for Node, pyenv for Python, rbenv for Ruby, each with its own `.nvmrc` or `.python-version` file. mise is one tool that replaces all of them, configured with one file."

---

### Section 1: The Problem mise Solves

**Core Concept to Teach:**
Developers typically end up with multiple version managers, each with its own installation process, shell hooks, config file format, and quirks. mise replaces all of them with a single tool that uses a unified config format (`.mise.toml`).

**How to Explain:**
1. Start with the problem: "If you work in Node and Python, you need nvm AND pyenv. Add Ruby, you need rbenv too. Each one has different commands, different config files (`.nvmrc`, `.python-version`, `.ruby-version`), and different shell setup"
2. Show the sprawl: "That's three tools, three shell integrations slowing down your shell startup, three config files to maintain per project"
3. Introduce the solution: "mise replaces all of them. One install, one shell integration, one config file"
4. Clarify what mise is NOT: "mise installs and manages tool versions. It's not a package manager — it won't replace npm, pip, or bundler. It manages the runtime itself (which Node version), not the packages you install into it"

**Discussion Points:**
- "How many version managers do you currently have installed? What's the most annoying thing about managing them?"
- "Have you ever opened a project and gotten errors because you were on the wrong Node or Python version?"

**How mise Differs from Shim-Based Tools:**
"Tools like asdf and nvm use shims — small wrapper scripts that sit in your PATH and redirect to the right version. Every time you run `node`, you're actually running a shim that looks up which version to use. mise instead directly puts the correct version's binary on your PATH. No shim overhead, no shim reshimming."

**Common Misconceptions:**
- Misconception: "mise is like Docker — it isolates my entire environment" -> Clarify: "mise only manages tool versions and env vars. It doesn't create containers or sandboxes. Your tools still run directly on your system"
- Misconception: "mise replaces npm/pip/bundler" -> Clarify: "mise manages the tool itself (which version of Node), not the packages you install into it. You still use npm for Node packages"
- Misconception: "I need to uninstall nvm/pyenv first" -> Clarify: "You can run mise alongside existing version managers while you transition. Just make sure mise's shell activation comes after others in your shell config so it takes priority"

**Verification Questions:**
1. "What problem does mise solve?"
2. "What's the difference between mise and npm/pip?"
3. Multiple choice: "Which of these does mise NOT do? A) Install Node.js B) Install npm packages C) Set environment variables D) Run project tasks"

**Good answer indicators:**
- They understand mise manages tool versions, not packages
- They can name specific tools mise replaces (nvm, pyenv, rbenv)
- They can answer B (install npm packages)

**If they struggle:**
- Focus on a concrete example: "You need Node 18 for project A and Node 20 for project B. That's the problem mise solves"
- If the version manager concept is new: "Think of it like having a bookshelf where you can pick which version of Node to use for each project, and it happens automatically when you cd into the project directory"

---

### Section 2: Installation and Shell Setup

**Core Concept to Teach:**
mise has two setup steps: install the binary, then activate it in your shell. Shell activation is what makes mise automatically switch tool versions when you cd into a project directory.

**How to Explain:**
1. "Installation is one command — a curl pipe to sh"
2. "But installing the binary isn't enough. You also need to activate it in your shell"
3. "Activation adds a hook that runs every time you cd. That hook checks if the directory has a `.mise.toml` and adjusts your PATH"

**Walk Through Together:**

Check if mise is already installed:
```bash
mise --version
```

If not installed:
```bash
curl https://mise.run | sh
```

Activate in shell — for zsh (most common on macOS):
```bash
echo 'eval "$(mise activate zsh)"' >> ~/.zshrc
```

For bash:
```bash
echo 'eval "$(mise activate bash)"' >> ~/.bashrc
```

Reload the shell:
```bash
source ~/.zshrc   # or restart the terminal
```

Verify everything is working:
```bash
mise doctor
```

**Point out:** "`mise doctor` checks your entire setup — shell integration, missing plugins, configuration issues. It's the first thing to run if something seems off."

**Common Misconceptions:**
- Misconception: "I just need to install the binary" -> Clarify: "The binary alone can install tools, but automatic version switching requires shell activation. Without it, you'd have to manually run `mise shell node@20` every time"
- Misconception: "The `eval` line slows down my shell" -> Clarify: "mise's shell hook is very fast — it manipulates PATH directly instead of using shims, so it has minimal impact on shell startup time"

**Verification Questions:**
1. "Why isn't just installing the mise binary enough?"
2. "What does `mise doctor` do?"
3. Multiple choice: "After installing mise, what does `eval \"$(mise activate zsh)\"` do? A) Installs mise B) Adds auto-switching when you cd into directories C) Creates a .mise.toml file D) Downloads all available tools"

**Good answer indicators:**
- They understand the two-step process (install + activate)
- They know `mise doctor` verifies the setup
- They can answer B (auto-switching on cd)

**If they struggle:**
- Draw the parallel to nvm: "nvm also needs a shell hook — that block you add to your `.zshrc`. mise is the same idea, just one line instead of several"
- "If you skip activation, mise still works — you just have to tell it what version to use every time, instead of it figuring it out from your config file"

**Exercise 2.1:**
"Install mise (if needed) and activate it in your shell. Run `mise doctor` and check for any issues."

**How to Guide Them:**
1. "First check if it's already installed: `mise --version`"
2. If not: "Run the curl command, then add the activation line to your shell config"
3. "Restart your terminal or source your shell config, then run `mise doctor`"
4. "Look at the output — are there any warnings? If so, let's address them"

---

### Section 3: Managing Tool Versions

**Core Concept to Teach:**
mise can install any tool version and set it as the active version for a project, a directory tree, or globally. The key distinction: `mise install` downloads a version; `mise use` downloads it AND writes it to `.mise.toml` so the project remembers which version to use.

**How to Explain:**
1. "`mise install node@20` downloads Node 20, but doesn't activate it anywhere. It's sitting on disk, ready"
2. "`mise use node@20` downloads it (if needed) AND writes `node = \"20\"` to `.mise.toml` in the current directory. Now anyone who cds into this directory with mise gets Node 20 automatically"
3. "Think of `install` as 'download' and `use` as 'download and pin to this project'"

**Walk Through Together:**

Install a specific version of Node:
```bash
mise install node@20
```

Check what's installed:
```bash
mise ls
```

Set Node 20 for the current project:
```bash
mise use node@20
```

Point out: "Look — mise created a `.mise.toml` file in the current directory. Let's see what's in it."

```bash
cat .mise.toml
```

Expected output:
```toml
[tools]
node = "20"
```

Verify the right version is active:
```bash
node --version
```

Set a global default:
```bash
mise use --global node@20
```

Point out: "This writes to `~/.config/mise/config.toml` instead of the local `.mise.toml`. The global version is a fallback — local always wins."

**Version Resolution Order:**
"When you run `node`, mise looks for which version to use in this order:
1. `.mise.toml` in the current directory
2. `.mise.toml` in parent directories (walking up to /)
3. `~/.config/mise/config.toml` (global config)

This means a project can pin its own version, and it overrides your global default."

**Compatibility with Other Config Files:**
"mise also reads `.tool-versions` (asdf format), `.nvmrc`, and `.python-version` files. So if a project already has an `.nvmrc`, mise will respect it. But for new projects, use `.mise.toml` — it supports everything else too (env vars, tasks)."

**Common Misconceptions:**
- Misconception: "`mise install` makes that version active" -> Clarify: "`install` only downloads. `use` is what activates a version and writes it to config"
- Misconception: "mise installs the latest version if I don't specify one" -> Clarify: "You need to specify at least a major version. `mise use node@20` gets the latest 20.x, but `mise use node` without a version will error or install the absolute latest"
- Misconception: ".mise.toml is global" -> Clarify: "Each directory can have its own `.mise.toml`. mise uses the one closest to your current directory"

**Verification Questions:**
1. "What's the difference between `mise install node@20` and `mise use node@20`?"
2. "Where does mise look for version configuration, and in what order?"
3. Multiple choice: "You have Node 18 set in `~/.config/mise/config.toml` and Node 20 in `~/projects/myapp/.mise.toml`. When you're in `~/projects/myapp/`, which version of Node do you get? A) 18 B) 20 C) The system default D) An error"

**Good answer indicators:**
- They understand `install` = download, `use` = download + pin
- They can describe the resolution order (local -> parent -> global)
- They can answer B (local overrides global)

**If they struggle:**
- Compare to nvm directly: "`nvm install 20` then `nvm use 20` is two commands. `mise use node@20` is both in one. And it writes `.mise.toml` so you don't forget"
- For resolution order: "It works like CSS specificity — the most specific (closest to your current directory) wins"

**Exercise 3.1:**
"Install Python 3.12 alongside Node 20 in the same project. Verify both are active with `node --version` and `python --version`. Check `.mise.toml` to see both tools listed."

**How to Guide Them:**
1. "Use `mise use python@3.12` in the same directory where you used Node"
2. "Check `.mise.toml` — it should now have both tools"
3. "Verify with the version commands"

**Exercise 3.2:**
"Create a subdirectory, cd into it, and set a different Node version there. Verify that the subdirectory uses its own version while the parent uses the original. Then cd back to the parent to confirm it still uses its version."

**How to Guide Them:**
1. "Create a `sub/` directory and cd into it"
2. "Run `mise use node@18` in the subdirectory"
3. "Check `node --version` — it should show 18"
4. "cd back to the parent and check again — it should show 20"
5. "This demonstrates the resolution order in action"

---

### Section 4: Environment Variables

**Core Concept to Teach:**
mise can set project-scoped environment variables through the `[env]` section in `.mise.toml`. This replaces tools like direnv for simple cases, keeping all project configuration in one file.

**How to Explain:**
1. "Most projects need environment variables — database URLs, API keys for development, feature flags"
2. "You could use a `.env` file, but those need a tool to load them. You could use direnv, but that's another tool to install and configure"
3. "mise already hooks into your shell. Adding env vars to `.mise.toml` means they activate and deactivate automatically as you cd in and out of the project — no extra tools needed"

**Walk Through Together:**

Add environment variables to `.mise.toml`:
```toml
[env]
DATABASE_URL = "postgres://localhost/myapp"
NODE_ENV = "development"
```

After saving, the variables are immediately available (mise's shell hook picks up the change):
```bash
echo $DATABASE_URL
# postgres://localhost/myapp

echo $NODE_ENV
# development
```

cd out of the project directory and check:
```bash
cd ~
echo $DATABASE_URL
# (empty — the variable is unset)
```

cd back and it's restored:
```bash
cd ~/projects/myapp
echo $DATABASE_URL
# postgres://localhost/myapp
```

**Loading .env Files:**
"If you already have a `.env` file, mise can load it instead of duplicating everything in `.mise.toml`:"

```toml
[env]
_.file = ".env"
```

"This tells mise to load variables from `.env` when you're in the project. You can combine this with variables defined directly in `[env]` — explicit values in `.mise.toml` take precedence."

**PATH Manipulation:**
"You can also prepend directories to PATH:"

```toml
[env]
_.path = ["./node_modules/.bin", "./scripts"]
```

"This adds `./node_modules/.bin` and `./scripts` to the front of your PATH when you're in the project. Handy for running project-local binaries without npx."

**Common Misconceptions:**
- Misconception: "Environment variables in .mise.toml are permanent" -> Clarify: "They're scoped to the directory. cd out and they're gone. cd back in and they're restored. This is the whole point — project-scoped config"
- Misconception: "I should put secrets in .mise.toml" -> Clarify: ".mise.toml is meant to be checked into version control. Put secrets in a `.env` file that's gitignored, and load it with `_.file`"
- Misconception: "mise env vars conflict with direnv" -> Clarify: "They can coexist, but using both is redundant. Pick one. If you're already using mise for tools, using it for env vars too simplifies your setup"

**Verification Questions:**
1. "How do you set an environment variable that's scoped to a project?"
2. "What happens to mise-managed env vars when you cd out of the project?"
3. "Where should secrets go — in `.mise.toml` or a `.env` file? Why?"

**Good answer indicators:**
- They understand env vars are directory-scoped and automatic
- They know variables are unset when leaving the project directory
- They know secrets go in `.env` (gitignored) loaded via `_.file`, not in `.mise.toml`

**If they struggle:**
- Compare to direnv: "It works like direnv's `.envrc`, but the config is in `.mise.toml` instead of a separate file"
- "Think of it as each project having its own little world of environment variables that appears when you walk in and disappears when you walk out"

**Exercise 4.1:**
"Add a `DATABASE_URL` and an `APP_ENV` variable to your project's `.mise.toml`. Verify they're set when you're in the project directory and unset when you cd out."

**How to Guide Them:**
1. "Add the `[env]` section to your existing `.mise.toml`"
2. "Run `echo $DATABASE_URL` to verify"
3. "cd to your home directory and check again — it should be empty"
4. "cd back and check once more"

**Exercise 4.2:**
"Create a `.env` file with a `SECRET_KEY` variable. Configure `.mise.toml` to load it with `_.file`. Verify the variable is available. Then add `_.path` to include `./scripts` in your PATH."

**How to Guide Them:**
1. "Create a `.env` file with `SECRET_KEY=my-dev-secret`"
2. "Add `_.file = \".env\"` to the `[env]` section"
3. "Add `_.path = [\"./scripts\"]` to the `[env]` section"
4. "Verify with `echo $SECRET_KEY` and `echo $PATH | tr ':' '\n' | head -5`"

---

### Section 5: Tasks

**Core Concept to Teach:**
mise can define and run project tasks, replacing Makefiles and npm scripts for common workflows. Tasks are defined in `.mise.toml` and can depend on each other.

**How to Explain:**
1. "Every project has tasks — build, test, lint, dev server. Usually these end up as npm scripts, Makefile targets, or shell scripts you forget the names of"
2. "mise tasks put them in `.mise.toml` alongside your tools and env vars. One file for your entire project development setup"
3. "Tasks can depend on other tasks, so `mise run build` can automatically run `install` first"

**Walk Through Together:**

Add tasks to `.mise.toml`:
```toml
[tasks.install]
run = "npm install"

[tasks.build]
run = "npm run build"
depends = ["install"]

[tasks.dev]
run = "npm run dev"
depends = ["install"]

[tasks.test]
run = "npm test"
depends = ["install"]
```

List available tasks:
```bash
mise tasks ls
```

Run a task:
```bash
mise run build
```

Point out: "Notice that `build` depends on `install`. mise ran `npm install` first, then `npm run build`. Dependencies are resolved automatically."

Run multiple tasks:
```bash
mise run lint test
```

**File-Based Tasks:**
"For tasks that are more than a one-liner, you can create executable scripts in `.mise/tasks/`:"

```bash
mkdir -p .mise/tasks
```

Create `.mise/tasks/setup`:
```bash
#!/bin/bash
# Full project setup for new developers
echo "Installing dependencies..."
npm install
echo "Setting up database..."
createdb myapp_dev || true
echo "Running migrations..."
npm run migrate
echo "Setup complete!"
```

Make it executable:
```bash
chmod +x .mise/tasks/setup
```

Run it:
```bash
mise run setup
```

Point out: "File-based tasks are discovered automatically. Any executable file in `.mise/tasks/` becomes a task you can run with `mise run`."

**Common Misconceptions:**
- Misconception: "mise tasks replace npm entirely" -> Clarify: "mise tasks are for project-level workflow commands. You still use npm to manage Node packages. But you can wrap npm scripts in mise tasks for a unified interface across languages"
- Misconception: "Tasks only work with Node/npm" -> Clarify: "Tasks run any shell command. Use them for Python projects, Go projects, multi-language projects — anything"
- Misconception: "File-based tasks need to be registered in .mise.toml" -> Clarify: "mise discovers them automatically from the `.mise/tasks/` directory. Just make them executable"

**Verification Questions:**
1. "What's the advantage of mise tasks over npm scripts or Makefiles?"
2. "How do task dependencies work?"
3. Multiple choice: "If task `build` depends on `install`, what happens when you run `mise run build`? A) Only build runs B) install runs first, then build C) You get an error D) Both run in parallel"

**Good answer indicators:**
- They understand tasks provide a unified interface across tools/languages
- They know dependencies run before the dependent task
- They can answer B (install runs first, then build)

**If they struggle:**
- Compare to npm scripts: "It's like `\"prebuild\": \"npm install\"` in package.json, but explicit and language-agnostic"
- Compare to Makefiles: "Similar to Make targets with dependencies, but the config is TOML instead of Makefile syntax"

**Exercise 5.1:**
"Define three tasks in your `.mise.toml`: `greet` (echoes 'Hello from mise!'), `shout` (echoes 'HELLO FROM MISE!'), and `both` (depends on `greet` and `shout`). Run `mise run both` and observe the dependency order."

**How to Guide Them:**
1. "Add `[tasks.greet]`, `[tasks.shout]`, and `[tasks.both]` sections"
2. "For `both`, set `depends = [\"greet\", \"shout\"]`"
3. "Run `mise run both` and observe that greet and shout run before both"

**Exercise 5.2:**
"Create a file-based task at `.mise/tasks/info` that prints the current Node version, Python version, and all environment variables set by mise. Run it with `mise run info`."

**How to Guide Them:**
1. "Create the `.mise/tasks/` directory"
2. "Write a bash script that runs `node --version`, `python --version`, and `env | grep -E 'DATABASE_URL|NODE_ENV'`"
3. "Make it executable with `chmod +x`"
4. "Run it with `mise run info`"

---

## Practice Project

**Project Introduction:**
"Let's put everything together. You'll set up a project directory from scratch with multiple tools, environment variables, and tasks — the kind of setup you'd use for a real application."

**Requirements:**
Present one at a time:
1. "Create a new project directory called `mise-demo`"
2. "Configure it to use Node 20 and Python 3.12"
3. "Add environment variables: `APP_NAME`, `APP_ENV` (set to development), and a `DATABASE_URL`"
4. "Create a `.env` file with a `SECRET_KEY` and load it via `_.file`"
5. "Define tasks: `check` (prints tool versions), `dev` (echoes a simulated dev server start message), and `info` (prints env vars). Make `dev` depend on `check`"
6. "Verify the whole setup works by running `mise run dev` and `mise run info`"

**Scaffolding Strategy:**
- Let them work independently first
- Check in after tool setup: "Got Node and Python configured? What does `mise ls` show?"
- Check in after env vars: "Try `echo $APP_NAME` — does it show up?"
- Check in after tasks: "Run `mise tasks ls` — see all three?"
- Final check: "Run `mise run dev`. Did `check` run first?"

**Checkpoints During Project:**
- After tool setup: `mise ls` shows both Node 20 and Python 3.12
- After env vars: `echo $APP_NAME` returns the expected value
- After tasks: `mise tasks ls` shows all defined tasks
- After running: `mise run dev` executes `check` first, then `dev`

**The Complete .mise.toml Should Look Like:**
```toml
[tools]
node = "20"
python = "3.12"

[env]
APP_NAME = "mise-demo"
APP_ENV = "development"
DATABASE_URL = "postgres://localhost/mise_demo"
_.file = ".env"

[tasks.check]
run = "echo \"Node: $(node --version), Python: $(python --version)\""

[tasks.dev]
run = "echo \"Starting development server for $APP_NAME...\""
depends = ["check"]

[tasks.info]
run = "echo \"APP_NAME=$APP_NAME\nAPP_ENV=$APP_ENV\nDATABASE_URL=$DATABASE_URL\""
```

**If They Get Stuck:**
- "Which part are you on — tools, env vars, or tasks?"
- "What does your `.mise.toml` look like so far?"
- If really stuck: "Let's build the `[tools]` section together, then you do `[env]` and `[tasks]`"

**Extension Ideas if They Finish Early:**
- "Add a file-based task in `.mise/tasks/` that does something multi-step"
- "Try `mise use --global` to set your default tool versions"
- "Explore `mise plugins ls-remote` to see what other tools mise can manage"
- "Set up `_.path` to include a local `scripts/` directory"

---

## Wrap-Up and Next Steps

**Review Key Takeaways:**
"Let's review what you learned today:"
- mise replaces nvm, pyenv, rbenv, and other version managers with one tool
- `.mise.toml` configures tools, env vars, and tasks in one file
- `mise install` downloads a tool; `mise use` downloads and pins it to the project
- Version resolution goes from local `.mise.toml` up to global config
- Environment variables are directory-scoped — they activate/deactivate as you navigate
- Tasks provide a unified project command interface

**Ask them to explain one concept:**
"Can you walk me through what happens when someone clones a project with a `.mise.toml` and cds into it for the first time?"
(This reinforces the end-to-end workflow: mise reads the config, installs/activates the right tool versions, sets env vars)

**Assess Confidence:**
"On a scale of 1-10, how confident do you feel using mise for your projects?"

**Respond based on answer:**
- 1-4: "That's okay! Start by using it on one project. Replace whatever version manager you use today for that project with mise, and get comfortable with `mise use` and `mise ls`"
- 5-7: "Good progress! Try adding `.mise.toml` to your next project from the start. The env vars and tasks are where the real convenience kicks in"
- 8-10: "Nice. You're ready to make mise your default for all projects. Next up: explore the plugin ecosystem and consider Nix for full environment reproducibility"

**Suggest Next Steps:**
Based on their progress and interests:
- "To practice: Convert an existing project from nvm/pyenv to mise"
- "For team workflows: Add `.mise.toml` to a shared repo and document the setup"
- "For advanced use: Explore Nix integration or custom plugins"
- "For CI/CD: Look into `mise install` in CI pipelines to match local tool versions"

**Encourage Questions:**
"Do you have any questions about anything we covered?"
"Is there a project you want to migrate to mise?"
"What other tools would you like mise to manage?"

---

## Adaptive Teaching Strategies

### If Learner is Struggling

**Signs:**
- Confused about the difference between `install` and `use`
- Not clear on where `.mise.toml` fits in
- Shell activation isn't working

**Strategies:**
- Slow down significantly
- Focus on just one tool (Node) until they're comfortable, then add Python
- Use direct nvm comparisons: "`mise use node@20` is like `nvm use 20` + writing to `.nvmrc` combined"
- Walk through `mise doctor` output together to fix setup issues
- Have them check `cat .mise.toml` after every command to see what changed
- Stick to `mise use` and `mise ls` — skip `mise install` as a separate concept for now
- Defer env vars and tasks to a follow-up session if needed

### If Learner is Excelling

**Signs:**
- Completes exercises quickly
- Asks about advanced features
- Already thinking about team adoption

**Strategies:**
- Move at faster pace, less explanation
- Show `mise trust` for security model
- Discuss `mise.lock` for pinning exact versions
- Show `mise exec` for running commands with specific tool versions without setting them
- Introduce `mise watch` for file-watching tasks
- Discuss CI/CD integration patterns
- Challenge: "Migrate one of your real projects to mise and set up the complete `.mise.toml`"

### If Learner Seems Disengaged

**Signs:**
- Short responses
- Not asking questions
- Seems unconvinced of the value

**Strategies:**
- Check in: "Is this relevant to your day-to-day work? What tools do you actually use?"
- Focus on their specific stack: if they only use Node, show how mise simplifies just that
- Demonstrate the "clone and go" workflow: show how a `.mise.toml` means zero setup for new team members
- Make it concrete: "Let's set up your actual project with mise right now"
- If they don't see the value over nvm: acknowledge it — mise's advantages are more pronounced with multiple languages or when you add env vars and tasks

### Different Learning Styles

**Visual learners:**
- Show the `.mise.toml` file content after each change
- Use `mise ls` frequently to show the state of installed tools
- Point out `mise doctor` output as a visual health check

**Hands-on learners:**
- Less explanation upfront, get them running `mise use` immediately
- "Install Node 20, then install Python 3.12. Now check `mise ls`"
- Learn by configuring a real project

**Conceptual learners:**
- Explain the version resolution algorithm in detail
- Discuss how PATH manipulation works vs shims
- Compare mise's design philosophy to asdf, nvm, and Nix
- Explain why TOML was chosen over YAML or JSON

---

## Troubleshooting Common Issues

### mise Not Found After Installation
- Did they add the activation line to their shell config?
- Did they restart their terminal or source the config file?
- Check: `which mise` — if nothing, the binary isn't on PATH
- Try: `~/.local/bin/mise --version` — the default install location

### Shell Activation Not Working
- Verify the `eval` line is in the right file (`~/.zshrc` for zsh, `~/.bashrc` for bash)
- Make sure it's not inside an `if` block that might not execute
- Run `mise doctor` to check shell integration status
- If using tmux: make sure the shell config is sourced in tmux sessions

### Tool Installation Fails
- Check internet connectivity
- Run `mise doctor` for diagnostics
- Try a specific version: `mise install node@20.11.0` instead of just `@20`
- Check if the tool/plugin exists: `mise plugins ls-remote | grep toolname`

### .mise.toml Not Being Picked Up
- Confirm you're in the right directory: `pwd` then `ls .mise.toml`
- Check if the file is trusted: `mise trust`
- Run `mise doctor` to see if it detects the config
- Check for TOML syntax errors: malformed TOML is silently ignored

### Environment Variables Not Set
- Make sure shell activation is working (test with a tool version first)
- Check TOML syntax in the `[env]` section
- Verify with `mise env` to see what mise thinks should be set
- If loading `.env` via `_.file`: make sure the `.env` file exists and has valid syntax

### Tasks Not Running
- Check `mise tasks ls` to see if the task is recognized
- For file-based tasks: verify the file is executable (`chmod +x`)
- Check for TOML syntax errors in task definitions
- Verify task name matches: `mise run taskname` (case-sensitive)

---

## Teaching Notes

**Key Emphasis Points:**
- The `install` vs `use` distinction is the most important concept — spend time on it
- Showing `.mise.toml` after each change builds understanding of what mise is doing
- The "cd in/cd out" demonstration for env vars is the "aha moment"
- Start with just tool versions, layer on env vars, then tasks

**Pacing Guidance:**
- Don't rush Section 3 (managing tool versions) — this is the foundation
- Env vars and tasks can be covered faster once tool management clicks
- Give plenty of time for the practice project
- Better to master tool management than to rush through everything

**Success Indicators:**
You'll know they've got it when they:
- Use `mise use` without hesitation to set up tool versions
- Check `.mise.toml` to understand project requirements
- Start thinking about how to convert existing projects
- Ask questions like "can mise handle [tool X]?" (shows they're adopting it)
- Mention removing nvm/pyenv from their setup

**Most Common Confusion Points:**
1. **install vs use**: `install` downloads, `use` downloads AND configures
2. **Version resolution**: local wins over parent wins over global
3. **Shell activation**: must be set up for automatic switching to work
4. **Secrets in .mise.toml**: should use `.env` via `_.file` instead

**Teaching Philosophy:**
- mise's value becomes obvious when you see it replace multiple tools
- Get them using `mise use` early — the `.mise.toml` generation is satisfying
- The env var "cd in/cd out" demo is the most convincing moment
- Don't oversell it — for someone who only uses one language, the value is smaller (but still real with env vars and tasks)
- Real projects make the best teaching examples
