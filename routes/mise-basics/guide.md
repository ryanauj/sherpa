---
title: mise Basics - Dev Tool Version Management
route_map: /routes/mise-basics/map.md
paired_sherpa: /routes/mise-basics/sherpa.md
prerequisites:
  - Basic command line usage
topics:
  - mise
  - Version Management
  - Environment Variables
  - Task Running
---

# mise Basics - Guide (Human-Focused Content)

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/mise-basics/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/mise-basics/map.md` for the high-level overview.

## Overview

mise (pronounced "meez") is a polyglot dev tool version manager. It replaces the collection of language-specific version managers you've accumulated — nvm for Node, pyenv for Python, rbenv for Ruby — with a single tool. Beyond version management, mise handles project-scoped environment variables and task running, combining what you'd otherwise piece together from direnv, Makefiles, and npm scripts.

This tutorial will teach you how to install mise, manage tool versions across projects, configure per-project environment variables, and define project tasks — all from a single `.mise.toml` configuration file.

## Learning Objectives

By the end of this tutorial, you will be able to:
- Explain what mise replaces and why a unified tool manager matters
- Install mise and activate it in your shell
- Install and manage tool versions for any language
- Configure per-project tool versions with `.mise.toml`
- Set project-scoped environment variables
- Define and run tasks

## Prerequisites

Before starting this tutorial, you should be familiar with:
- Using the command line (cd, ls, running programs)

Helpful but not required:
- Experience with any version manager (nvm, pyenv, rbenv)

---

## Section 1: The Problem mise Solves

### Version Manager Sprawl

If you work across multiple languages, you've probably ended up with something like this:

- **nvm** for Node.js, configured in `~/.nvm/`, reading `.nvmrc` files
- **pyenv** for Python, configured in `~/.pyenv/`, reading `.python-version` files
- **rbenv** for Ruby, configured in `~/.rbenv/`, reading `.ruby-version` files

Each one:
- Has its own installation process
- Adds its own hooks to your shell startup (slowing it down)
- Uses its own config file format
- Has its own set of commands to memorize

That's three tools doing the same thing — managing which version of a runtime is active — with three different interfaces.

### What mise Does

mise replaces all of them with one tool:

| Before | After |
|--------|-------|
| nvm, pyenv, rbenv, ... | mise |
| `.nvmrc`, `.python-version`, `.ruby-version` | `.mise.toml` |
| Multiple shell hooks | One `eval` line |
| Separate commands per tool | `mise use`, `mise install`, `mise ls` |

mise also does two things those tools don't:
- **Environment variables** — set per-project env vars (like direnv)
- **Task running** — define project commands (like Makefiles or npm scripts)

### How mise Works: PATH Manipulation vs Shims

Tools like nvm and asdf use **shims** — small wrapper scripts that intercept commands like `node` and redirect them to the correct version. Every time you run `node`, you're actually running a shim that looks up the right version.

mise takes a different approach: it directly modifies your **PATH** to point at the correct version's actual binary. No wrapper, no lookup overhead. When you cd into a project directory, mise's shell hook adjusts your PATH so `node` resolves directly to the right binary.

### What mise is NOT

To avoid confusion:
- **mise is not a package manager**. It doesn't replace npm, pip, or bundler. It manages the runtime itself (which version of Node), not the packages installed into it. You still run `npm install` for Node packages.
- **mise is not a sandbox or container**. Your tools run directly on your system, not in isolation. For containerized environments, look at Docker or Nix.

### Checkpoint 1

Before moving on, make sure you understand:
- [ ] Why having multiple version managers is a problem
- [ ] What mise replaces (nvm, pyenv, rbenv, direnv, etc.)
- [ ] The difference between mise and a package manager like npm
- [ ] How PATH manipulation differs from shims

---

## Section 2: Installation and Shell Setup

### Installing mise

Install mise with a single command:

```bash
curl https://mise.run | sh
```

**Expected output:**
```
mise: installing mise...
mise: installed successfully to ~/.local/bin/mise
```

On macOS, you can alternatively use Homebrew:
```bash
brew install mise
```

### Activating mise in Your Shell

Installing the binary isn't enough. You also need to activate mise in your shell so it can automatically switch tool versions when you cd into a project directory.

**For zsh** (default on macOS):
```bash
echo 'eval "$(mise activate zsh)"' >> ~/.zshrc
```

**For bash:**
```bash
echo 'eval "$(mise activate bash)"' >> ~/.bashrc
```

**For fish:**
```bash
echo 'mise activate fish | source' >> ~/.config/fish/config.fish
```

Then reload your shell:
```bash
source ~/.zshrc   # or restart your terminal
```

### Verifying Your Setup

Run the diagnostic command:

```bash
mise doctor
```

**Expected output** (abbreviated):
```
mise version: 2025.x.x
activated: yes
shims_on_path: yes
```

`mise doctor` checks your entire setup — shell integration, configuration, and potential issues. If anything is wrong, it tells you what to fix.

### Why Activation Matters

Without shell activation, mise still works — but you'd have to manually run commands like `mise shell node@20` every time you open a terminal. Activation adds a hook that runs every time you cd, automatically reading `.mise.toml` and adjusting your PATH. This is what makes version switching seamless.

### Checkpoint 2

Before moving on, make sure you can:
- [ ] Run `mise --version` and see output
- [ ] Run `mise doctor` with no errors
- [ ] Explain why shell activation is needed

---

## Section 3: Managing Tool Versions

This is the core of mise — installing and managing tool versions.

### Key Distinction: install vs use

These two commands look similar but do different things:

| Command | What it does |
|---------|-------------|
| `mise install node@20` | Downloads Node 20 to disk. Does NOT activate it or write config. |
| `mise use node@20` | Downloads Node 20 (if needed) AND writes `node = "20"` to `.mise.toml`. Activates it in the current directory. |

For most workflows, **`mise use` is the command you want**. It installs the tool and configures your project in one step.

### Installing and Setting a Tool Version

Let's set up Node 20 for a project:

```bash
mkdir ~/mise-tutorial && cd ~/mise-tutorial
mise use node@20
```

**Expected output:**
```
mise node@20.x.x ✓ installed
mise ~/mise-tutorial/.mise.toml tools: node@20.x.x
```

mise created a `.mise.toml` file. Let's look at it:

```bash
cat .mise.toml
```

**Expected output:**
```toml
[tools]
node = "20"
```

Verify the version is active:

```bash
node --version
```

**Expected output:**
```
v20.x.x
```

### Adding More Tools

You can manage multiple tools in the same project. Add Python:

```bash
mise use python@3.12
```

Check `.mise.toml` again:

```bash
cat .mise.toml
```

**Expected output:**
```toml
[tools]
node = "20"
python = "3.12"
```

Both tools are now configured for this project directory.

### Listing Installed Versions

See what's installed and active:

```bash
mise ls
```

**Expected output:**
```
Tool    Version    Source                              Requested
node    20.x.x     ~/mise-tutorial/.mise.toml         20
python  3.12.x     ~/mise-tutorial/.mise.toml         3.12
```

### Setting Global Defaults

Set a default version that applies everywhere (when no local `.mise.toml` exists):

```bash
mise use --global node@20
```

This writes to `~/.config/mise/config.toml` instead of the local `.mise.toml`.

### Version Resolution Order

When you run a tool like `node`, mise determines which version to use by checking these locations in order:

1. `.mise.toml` in the **current directory**
2. `.mise.toml` in **parent directories** (walking up to `/`)
3. `~/.config/mise/config.toml` (**global config**)

The most specific (closest to your current directory) wins. This means a project can pin its own version, and it overrides your global default.

### Compatibility with Other Config Files

mise also reads config files from other version managers:
- `.tool-versions` (asdf format)
- `.nvmrc` and `.node-version` (nvm/fnm)
- `.python-version` (pyenv)

If a project already has an `.nvmrc`, mise respects it. For new projects, prefer `.mise.toml` — it supports everything (tools, env vars, tasks) in one file.

### Exercise 3.1: Multi-Tool Project

**Task:** Create a new directory, configure it with Node 20 and Python 3.12, and verify both tools are active.

<details>
<summary>Hint 1</summary>

Use `mise use` for each tool. You don't need to `mise install` separately.
</details>

<details>
<summary>Hint 2</summary>

Check your work with `mise ls` and `cat .mise.toml`.
</details>

<details>
<summary>Solution</summary>

```bash
mkdir ~/multi-tool && cd ~/multi-tool
mise use node@20
mise use python@3.12

# Verify
mise ls
node --version     # v20.x.x
python --version   # Python 3.12.x
cat .mise.toml
```

**Expected `.mise.toml`:**
```toml
[tools]
node = "20"
python = "3.12"
```
</details>

### Exercise 3.2: Version Resolution

**Task:** In your project directory (with Node 20), create a subdirectory and set Node 18 there. Verify the subdirectory uses Node 18 while the parent uses Node 20. Then cd back to the parent to confirm.

<details>
<summary>Hint 1</summary>

Create a subdirectory with `mkdir sub`, cd into it, and run `mise use node@18`.
</details>

<details>
<summary>Hint 2</summary>

After setting up the subdirectory, use `node --version` in both directories to compare.
</details>

<details>
<summary>Solution</summary>

```bash
# In the project root (Node 20)
node --version     # v20.x.x

# Create a subdirectory with a different version
mkdir sub && cd sub
mise use node@18
node --version     # v18.x.x

# Go back to the parent
cd ..
node --version     # v20.x.x
```

The subdirectory's `.mise.toml` overrides the parent's. This is the version resolution order in action.
</details>

### Checkpoint 3

Before moving on, make sure you can:
- [ ] Explain the difference between `mise install` and `mise use`
- [ ] Set tool versions for a project with `mise use`
- [ ] List installed tools with `mise ls`
- [ ] Describe the version resolution order
- [ ] Set a global default with `mise use --global`

---

## Section 4: Environment Variables

mise can manage project-scoped environment variables, replacing tools like direnv for common cases.

### Setting Environment Variables

Add an `[env]` section to your `.mise.toml`:

```toml
[tools]
node = "20"
python = "3.12"

[env]
DATABASE_URL = "postgres://localhost/myapp"
NODE_ENV = "development"
```

After saving the file, the variables are immediately available (mise's shell hook picks up the change):

```bash
echo $DATABASE_URL
```

**Expected output:**
```
postgres://localhost/myapp
```

### Directory Scoping

Environment variables set by mise are scoped to the directory. When you cd out, they're unset. When you cd back in, they're restored.

```bash
# Inside the project directory
echo $NODE_ENV
# development

cd ~
echo $NODE_ENV
# (empty)

cd ~/mise-tutorial
echo $NODE_ENV
# development
```

This is the same behavior as direnv — your project gets its own isolated set of environment variables.

### Loading .env Files

If you already have a `.env` file (or you want to keep secrets out of `.mise.toml`), mise can load it:

```toml
[env]
DATABASE_URL = "postgres://localhost/myapp"
NODE_ENV = "development"
_.file = ".env"
```

Create a `.env` file:
```
SECRET_KEY=my-dev-secret-key
API_TOKEN=dev-token-12345
```

Now both the explicit values in `.mise.toml` and the values from `.env` are available. Explicit values in `.mise.toml` take precedence over `.env` values if there's a conflict.

**Important:** `.mise.toml` is meant to be checked into version control. `.env` files (containing secrets) should be gitignored. Use `_.file` to bridge the two.

### PATH Manipulation

You can prepend directories to your PATH:

```toml
[env]
_.path = ["./node_modules/.bin", "./scripts"]
```

This adds `./node_modules/.bin` and `./scripts` to the front of your PATH when you're in the project directory. This lets you run project-local binaries directly (like `eslint` instead of `npx eslint`).

### Exercise 4.1: Project Environment Variables

**Task:** Add `APP_NAME` and `APP_ENV` environment variables to your project's `.mise.toml`. Verify they're set inside the project directory and unset when you cd to your home directory.

<details>
<summary>Hint 1</summary>

Add the variables under the `[env]` section in `.mise.toml`.
</details>

<details>
<summary>Hint 2</summary>

Use `echo $APP_NAME` to check the value. cd to `~` and check again.
</details>

<details>
<summary>Solution</summary>

Add to `.mise.toml`:
```toml
[env]
APP_NAME = "my-project"
APP_ENV = "development"
```

```bash
# In the project directory
echo $APP_NAME
# my-project

echo $APP_ENV
# development

# Leave the project
cd ~
echo $APP_NAME
# (empty)

# Return
cd ~/mise-tutorial
echo $APP_NAME
# my-project
```
</details>

### Exercise 4.2: Loading Secrets from .env

**Task:** Create a `.env` file with a `SECRET_KEY` variable. Configure `.mise.toml` to load it. Verify the secret is available as an environment variable. Then add `_.path` to include a `./scripts` directory in your PATH.

<details>
<summary>Hint 1</summary>

Add `_.file = ".env"` to the `[env]` section of `.mise.toml`.
</details>

<details>
<summary>Hint 2</summary>

Add `_.path = ["./scripts"]` to the `[env]` section. Verify with `echo $PATH | tr ':' '\n' | head -5`.
</details>

<details>
<summary>Solution</summary>

Create `.env`:
```
SECRET_KEY=super-secret-dev-key
```

Update `.mise.toml`:
```toml
[env]
APP_NAME = "my-project"
APP_ENV = "development"
_.file = ".env"
_.path = ["./scripts"]
```

```bash
echo $SECRET_KEY
# super-secret-dev-key

# Check PATH includes ./scripts
echo $PATH | tr ':' '\n' | head -5
# /path/to/mise-tutorial/scripts
# ... (other PATH entries)
```
</details>

### Checkpoint 4

Before moving on, make sure you can:
- [ ] Set environment variables in `.mise.toml`
- [ ] Explain that env vars are directory-scoped (set on cd in, unset on cd out)
- [ ] Load a `.env` file with `_.file`
- [ ] Manipulate PATH with `_.path`
- [ ] Explain why secrets should go in `.env` (gitignored) rather than `.mise.toml`

---

## Section 5: Tasks

mise can define and run project tasks, providing a unified command interface regardless of what languages or tools your project uses.

### Defining Tasks in .mise.toml

Add a `[tasks]` section to your `.mise.toml`:

```toml
[tasks.install]
run = "npm install"

[tasks.build]
run = "npm run build"
depends = ["install"]

[tasks.dev]
run = "npm run dev"
depends = ["install"]
```

### Listing Tasks

See all available tasks:

```bash
mise tasks ls
```

**Expected output:**
```
Name      Description  Source
build                  ~/mise-tutorial/.mise.toml
dev                    ~/mise-tutorial/.mise.toml
install                ~/mise-tutorial/.mise.toml
```

### Running Tasks

```bash
mise run build
```

Because `build` depends on `install`, mise runs `npm install` first, then `npm run build`. Dependencies are resolved automatically.

You can also run multiple tasks:

```bash
mise run lint test
```

### Task Dependencies

Dependencies let you express relationships between tasks. When you run a task, its dependencies run first:

```toml
[tasks.install]
run = "npm install"

[tasks.lint]
run = "npm run lint"
depends = ["install"]

[tasks.test]
run = "npm test"
depends = ["install"]

[tasks.ci]
run = "echo 'All checks passed!'"
depends = ["lint", "test"]
```

Running `mise run ci` will:
1. Run `install` (dependency of both lint and test)
2. Run `lint`
3. Run `test`
4. Run `ci` (echo the message)

### Comparing to Other Task Runners

| Feature | npm scripts | Makefile | mise tasks |
|---------|------------|----------|------------|
| Config format | JSON (package.json) | Makefile syntax | TOML (.mise.toml) |
| Language-specific | Node only | Language-agnostic | Language-agnostic |
| Dependencies | Via pre/post hooks | Via target deps | Via `depends` |
| Part of tool config | No (separate file) | No (separate file) | Yes (same .mise.toml) |

The advantage of mise tasks: they live alongside your tool versions and env vars in one file, and they work for any language.

### File-Based Tasks

For tasks that are more than a one-liner, create executable scripts in `.mise/tasks/`:

```bash
mkdir -p .mise/tasks
```

Create `.mise/tasks/setup`:

```bash
#!/bin/bash
echo "Installing dependencies..."
npm install

echo "Setting up database..."
createdb myapp_dev 2>/dev/null || echo "Database already exists"

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

File-based tasks are discovered automatically — any executable file in `.mise/tasks/` becomes a runnable task. You don't need to register them in `.mise.toml`.

### Exercise 5.1: Define and Run Tasks

**Task:** Define three tasks in `.mise.toml`: `greet` (echoes "Hello from mise!"), `shout` (echoes "HELLO FROM MISE!"), and `both` (depends on `greet` and `shout`). Run `mise run both` and observe the dependency order.

<details>
<summary>Hint 1</summary>

Each task needs its own `[tasks.name]` section with a `run` key.
</details>

<details>
<summary>Hint 2</summary>

For `both`, add `depends = ["greet", "shout"]` and a `run` command (even just `echo "Done!"`).
</details>

<details>
<summary>Solution</summary>

Add to `.mise.toml`:
```toml
[tasks.greet]
run = "echo 'Hello from mise!'"

[tasks.shout]
run = "echo 'HELLO FROM MISE!'"

[tasks.both]
run = "echo 'Both complete!'"
depends = ["greet", "shout"]
```

```bash
mise run both
```

**Expected output:**
```
Hello from mise!
HELLO FROM MISE!
Both complete!
```

The dependencies (`greet` and `shout`) run before `both`.
</details>

### Exercise 5.2: File-Based Task

**Task:** Create a file-based task at `.mise/tasks/info` that prints the current Node version, Python version, and any environment variables you've set. Run it with `mise run info`.

<details>
<summary>Hint 1</summary>

Create the `.mise/tasks/` directory and write a bash script. Don't forget the shebang line (`#!/bin/bash`).
</details>

<details>
<summary>Hint 2</summary>

Make the script executable with `chmod +x .mise/tasks/info`.
</details>

<details>
<summary>Solution</summary>

```bash
mkdir -p .mise/tasks
```

Create `.mise/tasks/info`:
```bash
#!/bin/bash
echo "=== Tool Versions ==="
echo "Node: $(node --version)"
echo "Python: $(python --version)"
echo ""
echo "=== Environment ==="
echo "APP_NAME: $APP_NAME"
echo "APP_ENV: $APP_ENV"
echo "DATABASE_URL: $DATABASE_URL"
```

```bash
chmod +x .mise/tasks/info
mise run info
```

**Expected output:**
```
=== Tool Versions ===
Node: v20.x.x
Python: Python 3.12.x

=== Environment ===
APP_NAME: my-project
APP_ENV: development
DATABASE_URL: postgres://localhost/myapp
```
</details>

### Checkpoint 5

Before moving on, make sure you can:
- [ ] Define tasks in `.mise.toml` with `[tasks.name]`
- [ ] Run tasks with `mise run`
- [ ] Express task dependencies with `depends`
- [ ] Create file-based tasks in `.mise/tasks/`
- [ ] Explain the advantage of mise tasks over language-specific alternatives

---

## Practice Project

### Project Description

Build a complete project setup from scratch that uses all of mise's features: multi-tool version management, environment variables, and tasks. This simulates setting up a real application's development environment.

### Requirements

- Create a new project directory called `mise-demo`
- Configure Node 20 and Python 3.12 as project tools
- Set environment variables: `APP_NAME`, `APP_ENV` (development), and `DATABASE_URL`
- Create a `.env` file with a `SECRET_KEY` and load it via `_.file`
- Define tasks: `check` (prints tool versions), `dev` (simulates starting a dev server), and `info` (prints env vars). Make `dev` depend on `check`
- Verify the whole setup works

### Getting Started

```bash
mkdir ~/mise-demo && cd ~/mise-demo
```

Now configure your tools, environment variables, and tasks!

### Validation

After setting everything up, verify:

**Tools are configured:**
```bash
mise ls
# Should show node 20.x.x and python 3.12.x
```

**Environment variables are active:**
```bash
echo $APP_NAME
# mise-demo

echo $SECRET_KEY
# (your secret from .env)
```

**Tasks work:**
```bash
mise tasks ls
# Should show check, dev, info

mise run dev
# Should run check first (dependency), then dev
```

**Directory scoping works:**
```bash
cd ~
echo $APP_NAME
# (empty)

cd ~/mise-demo
echo $APP_NAME
# mise-demo
```

### Complete .mise.toml Reference

After completing the project, your `.mise.toml` should look something like this:

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

And your `.env`:
```
SECRET_KEY=my-dev-secret-key
```

---

## Quick Reference

| Command | What it does |
|---------|-------------|
| `mise install node@20` | Download Node 20 (doesn't activate or write config) |
| `mise use node@20` | Download Node 20 AND write to `.mise.toml` |
| `mise use --global node@20` | Set Node 20 as global default |
| `mise ls` | List installed and active tool versions |
| `mise run taskname` | Run a task defined in `.mise.toml` or `.mise/tasks/` |
| `mise tasks ls` | List all available tasks |
| `mise doctor` | Check mise setup for problems |
| `mise trust` | Trust the current directory's `.mise.toml` |
| `mise env` | Show environment variables mise would set |

### Configuration File Locations

| File | Purpose |
|------|---------|
| `.mise.toml` | Per-project tools, env vars, and tasks (check into git) |
| `~/.config/mise/config.toml` | Global tool defaults |
| `.env` | Project secrets (gitignore this) |
| `.mise/tasks/` | File-based task scripts |

### Version Resolution Order

1. `.mise.toml` in current directory (highest priority)
2. `.mise.toml` in parent directories
3. `~/.config/mise/config.toml` (lowest priority)

## Summary

You've learned the fundamentals of mise:
- mise replaces nvm, pyenv, rbenv, and other version managers with one tool
- `.mise.toml` configures tools, env vars, and tasks in a single file
- `mise use` installs a tool version and pins it to your project
- Version resolution walks from local to parent to global config
- Environment variables are directory-scoped — they activate and deactivate as you navigate
- Tasks provide a language-agnostic project command interface

## Next Steps

Now that you understand mise basics, explore:
- **Converting existing projects**: Replace `.nvmrc` / `.python-version` files with `.mise.toml`
- **Team adoption**: Add `.mise.toml` to shared repositories for consistent tool versions
- **CI/CD integration**: Use `mise install` in CI pipelines to match local tool versions
- **Nix integration**: Combine mise with Nix for fully reproducible environments
- **Custom plugins**: Explore `mise plugins ls-remote` to see all supported tools

## Additional Resources

- [mise documentation](https://mise.jdx.dev): Official docs and guides
- [mise GitHub](https://github.com/jdx/mise): Source code and issue tracker
- [.mise.toml reference](https://mise.jdx.dev/configuration.html): Full configuration options
