---
title: Nix Dev Environments - Reproducible Development Shells
route_map: /routes/nix-dev-environments/map.md
paired_guide: /routes/nix-dev-environments/guide.md
topics:
  - Nix
  - Reproducible Environments
  - Development Shells
  - Flakes
  - direnv
---

# Nix Dev Environments - Sherpa (AI Teaching Guide)

**Purpose**: This sherpa guide helps AI assistants teach Nix-based reproducible development environments through structured interaction.

**Route Map**: See `/routes/nix-dev-environments/map.md` for the high-level overview of this route.
**Paired Guide**: The human-focused content is at `/routes/nix-dev-environments/guide.md`.

---

## Teaching Overview

### Learning Objectives
By the end of this session, the learner should be able to:
- Explain why reproducible environments matter and what makes Nix different from other package managers
- Use Nix for ad-hoc package management
- Read and write basic Nix language expressions (attribute sets, functions, let/in)
- Create a `shell.nix` that provides tools for a project
- Create a `flake.nix` with locked dependencies and multi-system support
- Set up direnv for automatic shell activation

### Prerequisites to Verify
Before starting, verify the learner has:
- Basic command line skills (cd, ls, running programs)
- Comfort reading and writing configuration file syntax
- A macOS or Linux system (Nix does not run natively on Windows — WSL2 is required)

**If prerequisites are missing**: Help them get comfortable with the command line first. If they're on Windows, help them set up WSL2 before proceeding.

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
- Good for checking factual knowledge about Nix concepts
- Example: "What makes a path in the Nix store unique? A) The package name B) The version number C) A hash of all inputs D) The installation date"

**Explanation Questions:**
- Ask learner to explain concepts in their own words
- Assess deeper understanding and ability to apply knowledge
- Example: "Why can two versions of Node.js coexist in the Nix store without conflicting?"

**Mixed Approach (Recommended):**
- Use multiple choice for quick checks on Nix store concepts and command syntax
- Use explanation questions for core concepts like reproducibility and the difference between shell.nix and flakes
- Adapt based on learner responses and confidence level

---

## Teaching Flow

### Introduction

**What to Cover:**
- The "works on my machine" problem — different developers get different results because their environments differ
- Nix solves this by making environments declarative and reproducible
- Nix is a package manager, but it's fundamentally different from brew/apt because it's functional and content-addressed
- The learning curve is real, but the payoff is worth it

**Opening Questions to Assess Level:**
1. "Have you used Nix before, or is this completely new?"
2. "What tools do you currently use to manage your development environment? (brew, mise, asdf, Docker, etc.)"
3. "Have you ever had a project break because a tool version changed on your machine?"

**Adapt based on responses:**
- If they've used Docker: Draw comparisons — both describe environments, but Nix doesn't use containers and is more granular
- If they use mise/asdf: Explain that Nix manages complete environments including system libraries, not just language runtimes
- If they use brew/apt: Great starting point — Nix is like brew but with reproducibility guarantees
- If they've had version conflicts: Use their experience as motivation — Nix prevents exactly that

**Good opening analogy:**
"Think of brew or apt like a shared filing cabinet — everyone installs into the same system directories, and upgrading one package can break another. Nix is more like giving each project its own isolated filing cabinet. Two projects can use different versions of Node.js simultaneously because they're stored at different paths, identified by the hash of everything that went into building them."

---

### Setup Verification

**Check if Nix is Installed:**
Ask them to run: `nix --version`

**If not installed:**
Recommend the Determinate Systems installer:
```bash
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install
```

"The Determinate Systems installer is the recommended way to install Nix. It enables flakes by default and provides a clean uninstall path. After installation, open a new terminal or source your shell profile."

**Verify flakes are enabled:**
```bash
nix flake --help
```

If this shows an error about experimental features, add to `~/.config/nix/nix.conf`:
```
experimental-features = nix-command flakes
```

**Quick Orientation:**
"Nix has two generations of commands. The older commands like `nix-shell` and `nix-build` are stable and widely documented. The newer commands like `nix develop`, `nix shell`, and `nix search` use the `nix` subcommand style and require flakes to be enabled. We'll use both — older commands for `shell.nix`, newer commands for flakes."

---

### Section 1: Why Nix? The Reproducibility Problem

**Core Concept to Teach:**
Nix is a functional package manager. Each package is stored at a unique path in `/nix/store` determined by a hash of all its inputs (source code, dependencies, build flags). This means packages never conflict, multiple versions coexist, and environments are reproducible by construction.

**How to Explain:**
1. Start with the problem: "You join a project. The README says 'install Node.js'. Which version? You install the latest, but the project needs v18. Your coworker has v20. The CI server has v16. Nothing works the same way for anyone."
2. Explain the Nix store: "In `/nix/store`, every package lives at its own path that includes a hash: `/nix/store/abc123-nodejs-20.11.0/`. That hash is computed from everything that went into building the package — source code, compiler, flags, dependencies. If any input changes, the hash changes, and it becomes a different package."
3. Explain the consequence: "This means Node.js 18 and Node.js 20 live at completely different paths. They don't conflict. They can both exist at the same time. And if two machines compute the same hash, they have the exact same package — bit for bit."

**Key Comparisons:**

| Feature | brew/apt | mise/asdf | Docker | Nix |
|---------|----------|-----------|--------|-----|
| Multiple versions | No (workarounds) | Yes | Yes (containers) | Yes (Nix store) |
| Reproducible | No | Partially | Yes (Dockerfiles) | Yes (by construction) |
| System libraries | Yes | No | Yes | Yes |
| Containers needed | No | No | Yes | No |
| Declarative config | No | `.tool-versions` | `Dockerfile` | `shell.nix` / `flake.nix` |

**Discussion Points:**
- "Have you ever had `brew upgrade` break something on your machine?"
- "What happens when a new team member sets up their machine — do they get the exact same environment as everyone else?"

**Common Misconceptions:**
- Misconception: "Nix is like Docker" -> Clarify: "Docker creates isolated containers with their own filesystem. Nix just manages packages on your host system. You get a shell with the right tools, not a container. Your files, editor, and everything else stay exactly the same."
- Misconception: "Nix replaces my system package manager" -> Clarify: "Nix installs alongside your existing system. It doesn't touch `/usr/bin` or your brew-installed packages. Everything goes into `/nix/store`."
- Misconception: "If two packages have the same version, they're identical" -> Clarify: "In Nix, the hash includes everything — build flags, dependencies, patches. Same version with different inputs produces different store paths."

**Verification Questions:**
1. "What makes a Nix store path unique? What goes into that hash?"
2. "Why can two versions of the same package coexist in Nix but not in brew?"
3. Multiple choice: "Developer A and Developer B both use a flake.nix with the same lock file. What can they expect? A) Similar but not identical tools B) Identical tools, bit-for-bit C) The same tool names but possibly different builds D) It depends on their OS"

**Good answer indicators:**
- They understand the hash captures all inputs, not just the version
- They can explain why content-addressing prevents conflicts
- They can answer B (identical, bit-for-bit)

**If they struggle:**
- Use the filing cabinet analogy again
- Compare to git commits: "A git commit hash uniquely identifies a state of code. A Nix store hash uniquely identifies a state of a package."
- Draw it out: show two paths in the store with different hashes for different Node.js versions

---

### Section 2: Installing Nix and First Commands

**Core Concept to Teach:**
Nix can be used immediately for ad-hoc package management — getting tools temporarily without installing them permanently. This is the fastest way to see Nix's value.

**How to Explain:**
1. "Before we write any config files, let's use Nix like a tool vending machine. Need Node.js for 5 minutes? Nix can give it to you and clean up when you're done."
2. "This is the `nix-shell -p` workflow — it drops you into a shell with the specified packages available."

**Walk Through Together:**

Try an ad-hoc shell:
```bash
# Get a temporary shell with Node.js
nix-shell -p nodejs

# Verify it's there
node --version
which node

# Exit the temporary shell
exit

# Node.js is gone from your PATH
which node
```

"See what happened? Node.js was available inside the nix-shell, but after you exited, it's gone from your PATH. The package is still cached in the Nix store (so it'll be instant next time), but it's not polluting your regular environment."

**Modern equivalent:**
```bash
# Same thing with the newer command syntax
nix shell nixpkgs#nodejs

# Or run a command without entering a shell
nix run nixpkgs#nodejs -- --version
```

**Finding packages:**
```bash
# Search for packages
nix search nixpkgs python

# Search for a specific package
nix search nixpkgs nodejs
```

"The nixpkgs repository contains over 100,000 packages. If a tool exists, it's probably in nixpkgs."

**Common Misconceptions:**
- Misconception: "`nix-shell -p` installs the package permanently" -> Clarify: "It only makes the package available in the current shell. Exit the shell and it's gone from your PATH. The package remains cached in the Nix store for speed, but it's not installed in the traditional sense."
- Misconception: "`nix-shell` and `nix shell` are the same" -> Clarify: "They're different commands. `nix-shell` is the older command that works with `shell.nix` files and `-p` flag. `nix shell` is the newer command that works with flake references like `nixpkgs#nodejs`."

**Verification Questions:**
1. "What happens to packages from `nix-shell -p` after you exit the shell?"
2. "How do you find a package in nixpkgs?"
3. Multiple choice: "You run `nix-shell -p python3 curl jq`. What do you get? A) Three new permanent installations B) A shell with all three tools available temporarily C) Three separate shells, one for each tool D) An error because you can only specify one package"

**Good answer indicators:**
- They understand ad-hoc shells are temporary
- They know `nix search nixpkgs` for finding packages
- They can answer B

**If they struggle:**
- Have them run the demo again — seeing a tool appear and disappear is compelling
- "Think of it like borrowing a tool from a neighbor. You use it, give it back, but they keep it in their shed (the Nix store) in case you want to borrow it again."

**Exercise 2.1:**
"Use `nix-shell -p` to get `cowsay` and `lolcat` in a temporary shell. Run `echo 'Nix is neat' | cowsay | lolcat`. Then exit and verify they're gone."

**How to Guide Them:**
1. "Use `nix-shell -p cowsay lolcat` to get both tools"
2. "Run the piped command"
3. "Exit and try `which cowsay` — it should not be found"

**Exercise 2.2:**
"Use `nix search nixpkgs` to find the package name for the `ripgrep` search tool. Then use `nix-shell -p` to try it out with `rg --version`."

**How to Guide Them:**
1. "Run `nix search nixpkgs ripgrep`"
2. "The package name is `ripgrep` — use `nix-shell -p ripgrep`"
3. "Run `rg --version` inside the shell"

---

### Section 3: Nix Language Fundamentals

**Core Concept to Teach:**
The Nix language is a simple functional language used to write configuration files. You don't need to learn it deeply — just enough to read and write `shell.nix` and `flake.nix` files.

**How to Explain:**
1. "Nix has its own language for configuration. It looks a bit like JSON mixed with a functional language. Don't worry — you only need a small subset to write dev environments."
2. "The key things to understand: attribute sets (like objects/dictionaries), functions, let/in for local variables, and string interpolation."

**Walk Through Together:**

Use `nix repl` to experiment interactively:
```bash
nix repl
```

**Attribute sets** (like JSON objects or Python dicts):
```nix
nix-repl> { name = "hello"; version = "1.0"; }
{ name = "hello"; version = "1.0"; }

nix-repl> { a = 1; b = 2; }.a
1
```

"Attribute sets are the core data structure. Note the semicolons after each attribute — that's the most common syntax error for beginners."

**Functions:**
```nix
nix-repl> greet = name: "Hello, ${name}!"
nix-repl> greet "world"
"Hello, world!"
```

"Functions use a colon syntax: `argument: body`. The argument is before the colon, the body is after. There are no parentheses like most languages."

**Functions with attribute set arguments** (this is what you'll see in shell.nix):
```nix
nix-repl> f = { x, y }: x + y
nix-repl> f { x = 1; y = 2; }
3
```

"This pattern — a function taking an attribute set — is how `shell.nix` files work. The `{ pkgs ? import <nixpkgs> {} }:` at the top is a function that takes an attribute set with a `pkgs` argument that has a default value."

**Let/in for local variables:**
```nix
nix-repl> let x = 1; y = 2; in x + y
3

nix-repl> let name = "Nix"; in "Hello, ${name}!"
"Hello, Nix!"
```

**`with` for bringing attributes into scope:**
```nix
nix-repl> let attrs = { a = 1; b = 2; }; in with attrs; a + b
3
```

"You'll see `with pkgs;` in Nix configs — it lets you write `nodejs` instead of `pkgs.nodejs`."

**`inherit` for pulling values into an attribute set:**
```nix
nix-repl> let x = 1; in { inherit x; y = 2; }
{ x = 1; y = 2; }
```

"`inherit x;` is shorthand for `x = x;` — it pulls a variable into an attribute set with the same name."

**Multi-line strings:**
```nix
nix-repl> ''
  line one
  line two
''
"line one\nline two\n"
```

"Double single-quotes delimit multi-line strings. You'll use these for `shellHook` scripts."

**Lists:**
```nix
nix-repl> [ 1 2 3 ]
[ 1 2 3 ]
```

"Lists are space-separated, no commas. You'll use these for `buildInputs`."

Exit the repl:
```
nix-repl> :q
```

**Common Misconceptions:**
- Misconception: "Nix language is a general-purpose programming language" -> Clarify: "It's a domain-specific language for package definitions and configuration. You won't write applications in it."
- Misconception: "I need to learn the whole language" -> Clarify: "For dev environments, you need: attribute sets, functions with default arguments, `with`, `let/in`, and strings. That's it."
- Misconception: "Semicolons are optional" -> Clarify: "They're required after every attribute in a set. Missing semicolons are the #1 syntax error."

**Verification Questions:**
1. "What does `{ pkgs ? import <nixpkgs> {} }:` mean at the top of a shell.nix?"
2. "What does `with pkgs;` do?"
3. Multiple choice: "Which of these is a valid Nix attribute set? A) {name: 'hello'} B) { name = 'hello'; } C) { name = \"hello\"; } D) {\"name\": \"hello\"}"

**Good answer indicators:**
- They understand it's a function taking an attribute set argument with a default value
- They know `with` brings attributes into scope
- They can answer C (double quotes and semicolons)

**If they struggle:**
- Focus on pattern recognition: "You don't need to write Nix from scratch. You need to recognize the patterns in config files and modify them."
- Have them modify existing examples rather than writing from scratch
- "Think of `{ pkgs ? import <nixpkgs> {} }:` as boilerplate for now. It means 'give me the package set'."

**Exercise 3.1:**
"In `nix repl`, create an attribute set representing a project with `name`, `version`, and `tools` (a list of strings). Then access the `name` attribute."

**How to Guide Them:**
1. "Enter `nix repl`"
2. "Create the set: `project = { name = \"myapp\"; version = \"1.0\"; tools = [ \"nodejs\" \"python\" ]; }`"
3. "Access name: `project.name`"

**Exercise 3.2:**
"Write a function that takes a name and returns an attribute set with a `greeting` attribute. Call it with your name."

**How to Guide Them:**
1. "Functions use the colon syntax: `argument: body`"
2. "Solution: `mkGreeting = name: { greeting = \"Hello, ${name}!\"; };`"
3. "Call it: `mkGreeting \"Alice\"`"
4. "Access the greeting: `(mkGreeting \"Alice\").greeting`"

---

### Section 4: Development Shells with shell.nix

**Core Concept to Teach:**
`shell.nix` defines a reproducible development shell — a list of tools and setup commands that Nix provides when you run `nix-shell`. It's the simplest way to give a project a declarative dev environment.

**How to Explain:**
1. "A `shell.nix` file says: 'this project needs Node.js 20, Python 3.12, and npm. When the shell starts, run this setup script.' Anyone who runs `nix-shell` gets exactly those tools."
2. "Think of it as a Dockerfile for your development shell, but without the container. You get the tools directly on your host system."

**Walk Through Together:**

Create a project directory and shell.nix:
```bash
mkdir ~/nix-demo && cd ~/nix-demo
```

Write this `shell.nix`:
```nix
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    nodejs_20
    python312
    nodePackages.npm
  ];

  shellHook = ''
    echo "Dev environment loaded!"
    echo "Node: $(node --version)"
    echo "Python: $(python3 --version)"
    export PROJECT_ROOT=$(pwd)
  '';
}
```

Explain each part:
- `{ pkgs ? import <nixpkgs> {} }:` — "This is a function that takes a package set. The default is your system's nixpkgs channel."
- `pkgs.mkShell` — "This function creates a development shell environment."
- `buildInputs` — "The list of packages to make available in the shell."
- `with pkgs;` — "So we can write `nodejs_20` instead of `pkgs.nodejs_20`."
- `shellHook` — "A bash script that runs when you enter the shell. Good for environment variables and setup."

Run it:
```bash
nix-shell
```

"You should see the shell hook output, and now `node`, `python3`, and `npm` are all available. Exit with `exit` or Ctrl-d."

**Pinning nixpkgs for reproducibility:**

"There's a problem with the basic `shell.nix` above: `<nixpkgs>` refers to whatever channel your system has, which can differ between machines. To make it truly reproducible, pin to a specific nixpkgs revision:"

```nix
{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-24.05.tar.gz") {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    nodejs_20
    python312
  ];

  shellHook = ''
    echo "Dev environment loaded!"
  '';
}
```

"Now everyone gets the exact same packages, regardless of what's on their system. This URL points to a specific release of nixpkgs."

**Common Misconceptions:**
- Misconception: "`buildInputs` installs packages globally" -> Clarify: "They're only available inside the nix-shell. Exit and they're gone from your PATH."
- Misconception: "`shellHook` runs every time I run a command" -> Clarify: "It runs once when you enter the shell, not on every command."
- Misconception: "I need to commit shell.nix before using it" -> Clarify: "It works from any directory. Just run `nix-shell` where the file is."
- Misconception: "The unpinned `<nixpkgs>` is fine for team use" -> Clarify: "It works for personal use, but for a team you need pinning. Without it, different machines can get different package versions."

**Verification Questions:**
1. "What's the difference between `buildInputs` and `shellHook`?"
2. "Why should you pin nixpkgs in a team project?"
3. Multiple choice: "You have a shell.nix with `nodejs_20` in `buildInputs`. You run `nix-shell`, then `exit`. What happens to Node.js? A) It stays installed B) It's removed from your PATH C) It's deleted from disk D) It causes an error"

**Good answer indicators:**
- They understand `buildInputs` is for packages, `shellHook` is for setup commands
- They understand pinning prevents version drift between machines
- They can answer B (removed from PATH, but still cached in Nix store)

**If they struggle:**
- Have them write a minimal shell.nix with just one package and verify it works
- "Focus on the two important parts: `buildInputs` (what tools) and `shellHook` (what setup)"
- Compare to a package.json: "`buildInputs` is like `dependencies`, `shellHook` is like a `postinstall` script"

**Exercise 4.1:**
"Create a `shell.nix` for a Python project that provides Python 3.12 and `pip`. Add a `shellHook` that creates a virtual environment if one doesn't exist and activates it."

**How to Guide Them:**
1. "Start with the template we walked through"
2. "For the shellHook, check if `.venv` exists with an `if` statement"
3. Solution:
```nix
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python312
    python312Packages.pip
  ];

  shellHook = ''
    if [ ! -d ".venv" ]; then
      echo "Creating virtual environment..."
      python3 -m venv .venv
    fi
    source .venv/bin/activate
    echo "Python dev environment ready!"
  '';
}
```

**Exercise 4.2:**
"Modify your shell.nix to also include `curl` and `jq`, and add environment variables to the shellHook that set `API_URL=http://localhost:3000` and `DEBUG=1`."

**How to Guide Them:**
1. "Add `curl` and `jq` to the `buildInputs` list"
2. "Add `export` lines to the `shellHook`"
3. "Verify with `echo $API_URL` inside the shell"

---

### Section 5: Flakes: The Modern Approach

**Core Concept to Teach:**
Flakes are Nix's solution to the reproducibility gaps in plain `shell.nix`. A `flake.nix` defines both inputs (where packages come from) and outputs (what the flake provides). A `flake.lock` file pins the exact version of every input, similar to `package-lock.json`.

**How to Explain:**
1. "Remember how we pinned nixpkgs with a URL in `shell.nix`? Flakes make that automatic and rigorous. Every input gets locked to a specific revision."
2. "A flake has two parts: `inputs` (your dependencies — usually just nixpkgs) and `outputs` (what you're providing — in our case, a dev shell)."
3. "Compare `flake.lock` to `package-lock.json` — same concept. It records the exact revision of every input so builds are reproducible."

**Why flakes are better than plain shell.nix:**
- Automatic locking via `flake.lock` (no manual URL pinning)
- Standardized structure (inputs/outputs)
- Evaluation is hermetic — no references to `<nixpkgs>` or other impure values
- Better caching and evaluation performance
- Composable — flakes can depend on other flakes

**Walk Through Together:**

Create a flake:
```bash
mkdir ~/flake-demo && cd ~/flake-demo
git init  # Flakes require a git repo
```

Write `flake.nix`:
```nix
{
  description = "My project dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
  };

  outputs = { self, nixpkgs }:
    let
      system = "aarch64-darwin";
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          nodejs_20
          python312
        ];

        shellHook = ''
          echo "Flake dev environment ready!"
        '';
      };
    };
}
```

"Note: change `system` to match your machine — `aarch64-darwin` for Apple Silicon Mac, `x86_64-darwin` for Intel Mac, `x86_64-linux` for most Linux."

Explain each part:
- `description` — "Human-readable description of what this flake provides."
- `inputs.nixpkgs.url` — "Where to get packages from. This points to a specific nixpkgs branch."
- `outputs` — "A function that receives the resolved inputs and returns what the flake provides."
- `devShells.${system}.default` — "The dev shell for this system. `default` means it's used when you run `nix develop` without specifying a name."

Enter the dev shell:
```bash
nix develop
```

"The first time, Nix downloads and evaluates nixpkgs. This takes a while. Subsequent runs are fast because everything is cached."

**Point out the lock file:**
```bash
cat flake.lock
```

"Nix created `flake.lock` automatically. It records the exact git revision of nixpkgs. Commit this file! It's what makes your environment reproducible."

**Updating inputs:**
```bash
nix flake update
```

"This updates all inputs to their latest revision and rewrites `flake.lock`. It's like running `npm update` — you choose when to update, and the lock file records exactly what changed."

**Multi-system support:**

"If your team uses both macOS and Linux, you need to support multiple systems:"

```nix
{
  description = "My project dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-darwin" "x86_64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      pkgsFor = system: nixpkgs.legacyPackages.${system};
    in {
      devShells = forAllSystems (system:
        let pkgs = pkgsFor system;
        in {
          default = pkgs.mkShell {
            buildInputs = with pkgs; [
              nodejs_20
              python312
            ];

            shellHook = ''
              echo "Dev environment ready on ${system}!"
            '';
          };
        }
      );
    };
}
```

"This pattern uses `genAttrs` to generate dev shell definitions for each supported system. Now `nix develop` works on both macOS and Linux."

**Common Misconceptions:**
- Misconception: "Flakes replace shell.nix entirely" -> Clarify: "shell.nix still works and is simpler for personal use. Flakes are better for team projects because of the lock file and hermetic evaluation."
- Misconception: "I don't need to commit flake.lock" -> Clarify: "You absolutely must commit flake.lock. It's the mechanism that makes your environment reproducible. Without it, different machines can resolve different versions."
- Misconception: "Flakes don't need a git repo" -> Clarify: "Flakes require a git repo. Nix uses git to determine which files are part of the flake. Untracked files are invisible to the flake."
- Misconception: "`nix develop` and `nix-shell` are interchangeable" -> Clarify: "`nix develop` is for flakes. `nix-shell` is for shell.nix. They serve similar purposes but use different mechanisms."

**Verification Questions:**
1. "What's the difference between `flake.nix` and `flake.lock`?"
2. "Why do flakes require a git repo?"
3. "How do you update your pinned dependencies?"
4. Multiple choice: "Your flake.lock pins nixpkgs to revision abc123. A teammate clones your repo and runs `nix develop`. What nixpkgs revision do they get? A) Their system's default B) The latest C) abc123 D) It depends on their Nix version"

**Good answer indicators:**
- They understand flake.nix declares what you want, flake.lock records exact versions
- They know git tracking determines what files the flake sees
- They know `nix flake update` updates the lock
- They can answer C (abc123 — that's the point of the lock file)

**If they struggle:**
- Compare directly to npm: "flake.nix is package.json, flake.lock is package-lock.json, nix develop is npm install + npm start"
- Start with the single-system example, add multi-system later
- "The inputs/outputs structure is the key thing. Everything else is details."

**Exercise 5.1:**
"Convert the shell.nix from Exercise 4.1 into a flake.nix. Use the multi-system pattern. Enter the dev shell with `nix develop`."

**How to Guide Them:**
1. "Start with the multi-system template"
2. "Move your `buildInputs` and `shellHook` into the `mkShell` call"
3. "Don't forget to `git init` and `git add flake.nix`"
4. "Run `nix develop` and verify your tools are available"

**Exercise 5.2:**
"Add a second dev shell called `ci` to your flake that only includes the testing tools (no editor integration or development-only tools). Enter it with `nix develop .#ci`."

**How to Guide Them:**
1. "In the `devShells` output, add another attribute alongside `default`"
2. "Example: `ci = pkgs.mkShell { buildInputs = with pkgs; [ nodejs_20 ]; };`"
3. "Run `nix develop .#ci` to enter the ci shell"

---

### Section 6: direnv Integration

**Core Concept to Teach:**
direnv automatically activates and deactivates environments when you `cd` into and out of project directories. Combined with nix-direnv, it gives you seamless Nix dev shells — just cd into a project and your tools are there.

**How to Explain:**
1. "Right now, you have to manually run `nix develop` every time you enter a project. direnv automates that."
2. "When you `cd` into a directory with an `.envrc` file, direnv runs it. With nix-direnv, that `.envrc` can say `use flake` and your entire Nix dev shell activates automatically."
3. "It's the same UX as mise's auto-activation, but powered by Nix environments underneath."

**Walk Through Together:**

Install direnv and nix-direnv:
```bash
# Get direnv and nix-direnv
nix profile install nixpkgs#direnv nixpkgs#nix-direnv
```

Hook direnv into your shell. Add to your shell config (`~/.zshrc` or `~/.bashrc`):
```bash
eval "$(direnv hook zsh)"   # for zsh
# or
eval "$(direnv hook bash)"  # for bash
```

Configure nix-direnv. Create or edit `~/.config/direnv/direnvrc`:
```bash
source $HOME/.nix-profile/share/nix-direnv/direnvrc
```

"Reload your shell after these changes."

Create an `.envrc` in your project:
```bash
cd ~/flake-demo
echo "use flake" > .envrc
```

Approve the .envrc:
```bash
direnv allow
```

"Now the magic: leave the directory and come back."
```bash
cd ~
cd ~/flake-demo
```

"Your dev shell activated automatically. Node.js and Python are available without running `nix develop`. Leave the directory and they disappear."

**The `.envrc` file:**
The file is typically just two words:
```bash
use flake
```

"That's it. `use flake` tells nix-direnv to evaluate the flake in the current directory and load its dev shell."

You can also pass arguments:
```bash
# Use a specific dev shell output
use flake .#ci

# Use a flake from a different path
use flake ../other-project
```

**Security model:**
"direnv requires explicit approval for each `.envrc` file. The first time (or when the file changes), you must run `direnv allow`. This prevents malicious repos from running arbitrary code when you cd into them."

**Common Misconceptions:**
- Misconception: "direnv replaces Nix" -> Clarify: "direnv is just the activation mechanism. Nix still does all the package management. direnv just triggers `nix develop` automatically."
- Misconception: "I don't need to run `direnv allow` after changing .envrc" -> Clarify: "Any change to .envrc requires re-approval. This is a security feature."
- Misconception: "direnv is slow because it runs `nix develop` every time" -> Clarify: "nix-direnv caches the environment. The first activation is slow, but subsequent activations are nearly instant."

**Verification Questions:**
1. "What does `use flake` in an `.envrc` do?"
2. "Why does direnv require `direnv allow` for each project?"
3. "What's the role of nix-direnv specifically?"

**Good answer indicators:**
- They understand `.envrc` triggers environment activation on directory entry
- They understand the security model (explicit approval)
- They know nix-direnv provides caching and the `use flake` command

**If they struggle:**
- Compare to mise: "mise activates when you cd into a directory with .tool-versions. direnv does the same thing with .envrc."
- Focus on the end result: "The goal is: cd into project, tools appear. cd out, tools disappear. That's it."

**Exercise 6.1:**
"Set up direnv for your flake-demo project. Verify that your tools activate when you cd into the directory and deactivate when you leave."

**How to Guide Them:**
1. "Create `.envrc` with `use flake`"
2. "Run `direnv allow`"
3. "Leave and re-enter the directory"
4. "Check `which node` — it should point to a Nix store path"
5. "cd out and check again — it should be gone (or point to a system node)"

---

## Practice Project

**Project Introduction:**
"Let's put everything together. Build a reproducible multi-language dev environment that a real team could use."

**Requirements:**
Present one at a time:
1. "Create a new project directory with a git repo"
2. "Write a `flake.nix` that provides: Node.js 20, Python 3.12, and `jq`"
3. "Support at least two systems (e.g., `aarch64-darwin` and `x86_64-linux`)"
4. "Add a `shellHook` that prints available tool versions and sets a `PROJECT_NAME` environment variable"
5. "Create an `.envrc` with `use flake`"
6. "Verify: cd out and back in — all tools should auto-activate"
7. "Commit `flake.nix`, `flake.lock`, and `.envrc`"

**Scaffolding Strategy:**
- Let them work independently first
- Check in after flake creation: "Does `nix develop` work?"
- Check in after direnv setup: "Does cd-ing in and out activate/deactivate?"
- Final check: "Are flake.nix, flake.lock, and .envrc committed?"

**Checkpoints During Project:**
- After flake.nix creation: Run `nix flake check` to validate
- After `nix develop`: Verify `node --version`, `python3 --version`, and `jq --version` all work
- After direnv setup: Leave and re-enter the directory, verify tools appear
- After commit: `git log` should show the three files

**If They Get Stuck:**
- "Which step are you on? Flake, direnv, or testing?"
- "What error are you seeing?"
- If really stuck on the flake: "Let's start with the single-system template and get it working, then add multi-system"

**Extension Ideas if They Finish Early:**
- "Add a second dev shell called `ci` with a subset of tools"
- "Add a `packages` output that builds a simple script"
- "Try adding a flake input from a different source (e.g., a GitHub repo)"
- "Add `pre-commit` hooks to the shell hook"
- "Create a `.gitignore` that excludes `.direnv/` (the nix-direnv cache directory)"

---

## Wrap-Up and Next Steps

**Review Key Takeaways:**
"Let's review what you learned today:"
- Nix is a functional package manager that stores packages by hash, enabling multiple versions and reproducibility
- `nix-shell -p` gives you ad-hoc temporary access to packages
- `shell.nix` defines a declarative dev environment with `mkShell`
- Flakes add automatic locking and hermetic evaluation for team-grade reproducibility
- direnv + nix-direnv automates shell activation when you enter a project

**Ask them to explain one concept:**
"Can you walk me through what happens when someone clones a repo with a flake.nix and flake.lock, then cd's into it with direnv set up?"
(This ties together all the concepts: flake inputs, lock file, direnv activation, package resolution)

**Assess Confidence:**
"On a scale of 1-10, how confident do you feel setting up a Nix dev environment for a project?"

**Respond based on answer:**
- 1-4: "That's normal — Nix has a real learning curve. The key concepts to internalize are: flake.nix declares what you want, flake.lock pins versions, and direnv activates automatically. Start by using the templates we built today and modify them for your projects."
- 5-7: "Good progress! You understand the concepts. Now it's about building muscle memory. Try converting one of your real projects to use a flake."
- 8-10: "Excellent! You're ready to introduce Nix to your team. Next steps: explore multi-flake projects, overlays, and NixOS configurations."

**Suggest Next Steps:**
Based on their progress and interests:
- "To practice: Convert an existing project to use a flake.nix + direnv"
- "For deeper Nix: Learn about overlays, custom packages, and the Nix derivation model"
- "For comparison: Try the mise-basics route to understand a simpler approach and when each tool is appropriate"
- "For containers: Try docker-dev-environments to see how Nix and Docker can complement each other"

**Encourage Questions:**
"Do you have any questions about anything we covered?"
"What part of the Nix workflow do you want to explore more?"
"Is there a specific project you want to set up with Nix?"

---

## Adaptive Teaching Strategies

### If Learner is Struggling

**Signs:**
- Confused by Nix language syntax
- Can't remember the difference between shell.nix and flake.nix
- Frustrated by the learning curve
- Getting errors they can't interpret

**Strategies:**
- Slow down significantly — Nix has more concepts to absorb than most tools
- Focus on the end result first: "Let me show you what a working setup looks like, then we'll understand how it works"
- Use templates — have them modify working examples rather than writing from scratch
- Skip the Nix language section if needed and come back to it when they hit a specific syntax question
- Emphasize that Nix's complexity pays off: "You're investing in a tool that will save you hours of 'works on my machine' debugging"
- Use comparison tables frequently — relate everything to tools they already know
- If they're stuck on the language: "Just treat `{ pkgs ? import <nixpkgs> {} }:` as boilerplate for now. Focus on `buildInputs` and `shellHook`."

### If Learner is Excelling

**Signs:**
- Completes exercises quickly
- Asks about advanced features (overlays, custom derivations)
- Already thinking about team adoption

**Strategies:**
- Move at a faster pace, less hand-holding
- Introduce overlays for customizing packages
- Show how to create custom derivations
- Discuss Nix in CI/CD pipelines
- Explore flake inputs from other sources (other flakes, git repos)
- Challenge: "Set up a Nix dev environment for one of your real projects"
- Discuss NixOS and home-manager for full system management
- Show `nix build` for creating reproducible build artifacts

### If Learner Seems Disengaged

**Signs:**
- Short responses
- Not asking questions
- Seems overwhelmed

**Strategies:**
- Check in: "Is the pace okay? Nix has a lot of concepts — it's fine to take it slower"
- Connect to their real pain: "What's the most annoying part of setting up your dev environment? Let me show you how Nix fixes that specific thing"
- Focus on the immediate payoff: skip ahead to direnv integration so they see the end-state UX
- Make it concrete: work with a project they actually care about, not toy examples
- Acknowledge the learning curve: "Nix is famously hard to learn. That's not you — it's the tool. But it's worth it."

### Different Learning Styles

**Visual learners:**
- Draw the Nix store as a tree of hashed paths
- Show the dependency graph between packages
- Use the comparison tables between Nix and other tools

**Hands-on learners:**
- Less explanation upfront, more `nix-shell -p` experimentation
- "Try this and see what happens" approach
- Let them break things — Nix is safe to experiment with

**Conceptual learners:**
- Explain the functional programming model behind Nix
- Discuss content-addressing and how it relates to git
- Talk about the Nix derivation model and build sandboxing
- Compare Nix's approach to other reproducibility solutions (Docker, Guix, Bazel)

---

## Troubleshooting Common Issues

### Nix Not Found After Installation
- Open a new terminal or source your shell profile
- Check if `/nix/var/nix/profiles/default/bin` is in your PATH
- The Determinate installer modifies your shell config — verify the changes were applied

### "experimental feature 'flakes' is disabled"
- Add to `~/.config/nix/nix.conf`:
  ```
  experimental-features = nix-command flakes
  ```
- The Determinate Systems installer enables this by default — if you see this error, you may have used a different installer

### "path not found" or "file not tracked by git"
- Flakes only see files tracked by git
- Run `git add flake.nix` before `nix develop`
- If files aren't showing up in your flake, check `git status`

### "hash mismatch" or Download Failures
- Network issue — check your internet connection
- If behind a proxy, configure Nix proxy settings
- Try `nix flake update` to refresh inputs

### Slow First Evaluation
- The first `nix develop` downloads and evaluates all of nixpkgs — this can take several minutes
- Subsequent runs are fast because results are cached
- "This is normal. It's like the first `npm install` in a large project."

### direnv Not Activating
- Check shell hook: is `eval "$(direnv hook zsh)"` in your `.zshrc`?
- Did you run `direnv allow`?
- Check `direnv status` for diagnostic info
- Make sure nix-direnv is installed and `direnvrc` is configured

### Package Not Found
- Use `nix search nixpkgs <name>` to find the correct attribute name
- Package names in nixpkgs don't always match the command name (e.g., the `rg` command comes from the `ripgrep` package)
- Check [search.nixos.org](https://search.nixos.org/packages) for a web-based search

### "infinite recursion" or Evaluation Errors
- Usually a syntax error in the Nix expression
- Check for missing semicolons in attribute sets
- Check for circular references
- Use `nix flake check` for flake-specific validation

---

## Teaching Notes

**Key Emphasis Points:**
- The reproducibility story is the main selling point — keep coming back to it
- The learning curve is the main obstacle — acknowledge it openly and keep momentum
- Practical examples beat theory — get them into `nix-shell -p` as fast as possible
- Comparison to familiar tools reduces cognitive load
- The ad-hoc `nix-shell -p` is the quickest win and best motivation

**Pacing Guidance:**
- Don't rush Section 1 (why Nix) — the motivation needs to be solid before tackling syntax
- Section 3 (language) can feel dry — keep it practical and skip ahead if they're getting bogged down
- Give plenty of time for Sections 4 and 5 (shell.nix and flakes) — this is the core skill
- Section 6 (direnv) is the payoff moment — make sure they experience the "cd in, tools appear" magic
- Better to master shell.nix + flakes than to rush through direnv

**Success Indicators:**
You'll know they've got it when they:
- Can explain why Nix is different from brew/mise in their own words
- Can write a flake.nix from a template without looking at examples
- Get excited about the direnv integration
- Start thinking about which of their projects to convert
- Ask questions about team adoption or CI integration

**Most Common Confusion Points:**
1. **Nix language syntax**: Semicolons, colons for functions, `with` scope
2. **shell.nix vs flake.nix**: When to use which and how they relate
3. **`nix-shell` vs `nix develop` vs `nix shell`**: Three commands that sound similar but serve different purposes
4. **The system string**: Forgetting to set the right system or not supporting multiple systems
5. **Git tracking with flakes**: Files must be `git add`ed before flakes can see them

**Teaching Philosophy:**
- Nix's value is in the guarantee, not the syntax — keep focus on what it enables
- Start with the simplest thing that works (ad-hoc shells), layer complexity
- Use their existing knowledge: "It's like X, but with Y guarantee"
- Acknowledge the ecosystem's rough edges honestly — Nix documentation is famously fragmented
- The moment they see direnv auto-activate a flake dev shell, they'll understand the vision
