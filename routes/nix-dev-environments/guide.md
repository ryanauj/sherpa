---
title: Nix Dev Environments - Reproducible Development Shells
route_map: /routes/nix-dev-environments/map.md
paired_sherpa: /routes/nix-dev-environments/sherpa.md
prerequisites:
  - Basic command line usage
  - Comfort with configuration file syntax
topics:
  - Nix
  - Reproducible Environments
  - Development Shells
  - Flakes
  - direnv
---

# Nix Dev Environments - Guide (Human-Focused Content)

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/nix-dev-environments/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/nix-dev-environments/map.md` for the high-level overview.

## Overview

Nix is a functional package manager that creates fully reproducible development environments. Instead of hoping everyone on your team has the same tools installed, you declare what you need in a file, and Nix guarantees everyone gets the exact same environment — down to the byte.

This guide teaches you how to go from zero to a complete flake-based dev environment with automatic activation. You'll learn to write `shell.nix` and `flake.nix` files, understand enough of the Nix language to modify them confidently, and set up direnv so your tools appear automatically when you enter a project directory.

Nix has a real learning curve. The syntax is unfamiliar, the documentation is scattered, and the ecosystem has rough edges. This guide acknowledges all of that and focuses on the practical subset you need for dev environments.

## Learning Objectives

By the end of this guide, you will be able to:
- Explain what makes Nix's approach to package management different
- Use `nix-shell -p` for ad-hoc temporary packages
- Read and write basic Nix language expressions
- Create dev environments with `shell.nix`
- Create reproducible, locked dev environments with `flake.nix`
- Set up direnv for automatic shell activation

## Prerequisites

Before starting this guide, you should be familiar with:
- Using the command line (cd, ls, running programs)
- Reading configuration file syntax (YAML, JSON, or similar)
- Using a text editor from the command line

Helpful but not required:
- Experience with any package manager (brew, apt, npm)
- The mise-basics route (for comparison)

## Setup

### Installing Nix

The recommended installer is from Determinate Systems. It enables flakes by default and provides a clean uninstall path.

```bash
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install
```

After installation, open a new terminal (or source your shell profile) and verify:

```bash
nix --version
```

**Expected output:**
```
nix (Nix) 2.x.x
```

### Verify Flakes are Enabled

```bash
nix flake --help
```

If you see help text, flakes are enabled. If you get an error about experimental features, add the following to `~/.config/nix/nix.conf` (create the file if it doesn't exist):

```
experimental-features = nix-command flakes
```

The Determinate Systems installer does this automatically, so you should only need this if you used a different installer.

---

## Section 1: Why Nix? The Reproducibility Problem

### The Problem

You join a project. The README says "install Node.js." Which version? You install the latest. Your coworker installed it six months ago and has an older version. The CI server has yet another version. Everyone runs the same code and gets different results.

This is the "works on my machine" problem, and it happens because traditional package managers (brew, apt) install packages into shared system directories. Upgrading one package can affect everything else that depends on it. There's no isolation and no record of exactly what's installed.

### How Nix is Different

Nix is a **functional package manager**. Every package is stored in the Nix store (`/nix/store`) at a path that includes a cryptographic hash of all its inputs:

```
/nix/store/abc123...-nodejs-20.11.0/
/nix/store/def456...-nodejs-18.19.0/
```

That hash is computed from:
- The source code
- All dependencies
- The compiler and build flags
- Build scripts and patches

If any input changes, the hash changes, and it becomes a different path. This has several consequences:

- **Multiple versions coexist**: Node.js 18 and Node.js 20 live at different paths. They never conflict.
- **Reproducibility by construction**: If two machines compute the same hash, they have the same package — bit for bit.
- **No side effects**: Installing or removing a package can't break other packages because nothing shares paths.

### How Nix Compares to Other Tools

| Feature | brew/apt | mise/asdf | Docker | Nix |
|---------|----------|-----------|--------|-----|
| Multiple versions | No (workarounds) | Yes | Yes (containers) | Yes (Nix store) |
| Reproducible | No | Partially | Yes (Dockerfiles) | Yes (by construction) |
| System libraries | Yes | No | Yes | Yes |
| Containers needed | No | No | Yes | No |
| Declarative config | No | `.tool-versions` | `Dockerfile` | `shell.nix` / `flake.nix` |

- **vs. brew/apt**: Nix provides isolation and reproducibility. Brew installs everything into `/usr/local` (or `/opt/homebrew`); upgrading one package can break another. Nix store paths never interfere with each other.
- **vs. mise/asdf**: mise manages versions of language runtimes (Node, Python, Ruby). Nix manages complete environments including system libraries, build tools, and arbitrary packages.
- **vs. Docker**: Docker gives you a reproducible environment inside a container. Nix gives you a reproducible environment on your host system — no containers, no filesystem isolation, your editor and tools work normally.

### Checkpoint 1

Before moving on, make sure you understand:
- [ ] Why traditional package managers cause "works on my machine" problems
- [ ] What the Nix store is and why hashed paths matter
- [ ] How Nix differs from brew, mise, and Docker

---

## Section 2: Installing Nix and First Commands

### Ad-Hoc Packages

The fastest way to see Nix's value is ad-hoc package management — getting tools temporarily without installing them permanently.

```bash
nix-shell -p nodejs
```

**What happens:**
```
[nix-shell:~]$ node --version
v20.11.0
```

You're now in a shell with Node.js available. Check where it lives:

```bash
which node
```

**Expected output:**
```
/nix/store/...-nodejs-20.11.0/bin/node
```

Exit the shell:
```bash
exit
```

Now check:
```bash
which node
```

**Expected output:**
```
node not found
```

Node.js is gone from your PATH. The package is still cached in the Nix store (so it'll be instant next time you request it), but it's not in your environment.

### Multiple Packages at Once

```bash
nix-shell -p nodejs python3 curl jq
```

All four tools are available inside the shell. Exit and they're all gone.

### Modern Command Syntax

Nix also has a newer command syntax:

```bash
# Enter a shell with a package
nix shell nixpkgs#nodejs

# Run a single command without entering a shell
nix run nixpkgs#cowsay -- "Hello from Nix"
```

The `nixpkgs#package` syntax specifies which flake (nixpkgs) and which package to use.

### Finding Packages

```bash
nix search nixpkgs python
```

**Expected output (abbreviated):**
```
* legacyPackages.aarch64-darwin.python3 (3.12.x)
  The Python interpreter

* legacyPackages.aarch64-darwin.python312 (3.12.x)
  The Python interpreter

* legacyPackages.aarch64-darwin.python311 (3.11.x)
  The Python interpreter
...
```

The nixpkgs repository contains over 100,000 packages. You can also search at [search.nixos.org/packages](https://search.nixos.org/packages) for a web-based interface.

### Exercise 2.1: Ad-Hoc Package Exploration

**Task:** Use `nix-shell -p` to get `cowsay` and `lolcat`. Run `echo "Nix is neat" | cowsay | lolcat`. Then exit and verify both tools are gone.

<details>
<summary>Hint 1</summary>

You can specify multiple packages: `nix-shell -p cowsay lolcat`
</details>

<details>
<summary>Hint 2</summary>

After exiting, `which cowsay` should report that the command is not found.
</details>

<details>
<summary>Solution</summary>

```bash
# Enter shell with both tools
nix-shell -p cowsay lolcat

# Run the piped command
echo "Nix is neat" | cowsay | lolcat

# Exit
exit

# Verify they're gone
which cowsay
# cowsay not found
```
</details>

### Exercise 2.2: Finding and Trying a Package

**Task:** Use `nix search nixpkgs` to find the package for the `rg` command (ripgrep). Then use `nix-shell -p` to try it out.

<details>
<summary>Hint 1</summary>

Search for it: `nix search nixpkgs ripgrep`
</details>

<details>
<summary>Hint 2</summary>

The package name is `ripgrep`, even though the command is `rg`.
</details>

<details>
<summary>Solution</summary>

```bash
# Search for the package
nix search nixpkgs ripgrep

# Try it out
nix-shell -p ripgrep

# Inside the shell
rg --version

# Exit
exit
```
</details>

### Checkpoint 2

Before moving on, make sure you can:
- [ ] Use `nix-shell -p` to get temporary access to packages
- [ ] Search for packages with `nix search nixpkgs`
- [ ] Explain what happens to packages when you exit a nix-shell

---

## Section 3: Nix Language Fundamentals

You don't need to learn the Nix language deeply. You need just enough to read and write `shell.nix` and `flake.nix` files. This section covers that practical subset.

Start an interactive Nix session to experiment:
```bash
nix repl
```

### Attribute Sets

Attribute sets are Nix's core data structure — like JSON objects or Python dictionaries.

```nix
nix-repl> { name = "hello"; version = "1.0"; }
{ name = "hello"; version = "1.0"; }

nix-repl> { a = 1; b = 2; }.a
1
```

Note the **semicolons after each attribute**. This is the most common syntax error for Nix beginners.

Nested attribute sets:
```nix
nix-repl> { server = { host = "localhost"; port = 8080; }; }
{ server = { ... }; }

nix-repl> { server = { host = "localhost"; port = 8080; }; }.server.port
8080
```

### Functions

Functions use a colon syntax — the argument is before the colon, the body is after:

```nix
nix-repl> greet = name: "Hello, ${name}!"
nix-repl> greet "world"
"Hello, world!"
```

Functions can take attribute set arguments (this is the pattern you'll see in shell.nix):

```nix
nix-repl> add = { x, y }: x + y
nix-repl> add { x = 1; y = 2; }
3
```

Functions with default values (the `?` provides a default):

```nix
nix-repl> greetWithDefault = { name ? "stranger" }: "Hello, ${name}!"
nix-repl> greetWithDefault {}
"Hello, stranger!"
nix-repl> greetWithDefault { name = "Alice"; }
"Hello, Alice!"
```

This is how `shell.nix` files work. The `{ pkgs ? import <nixpkgs> {} }:` at the top is a function that takes an attribute set with a `pkgs` argument that defaults to importing nixpkgs.

### Let/In Blocks

`let/in` creates local variables:

```nix
nix-repl> let x = 1; y = 2; in x + y
3

nix-repl> let name = "Nix"; in "Hello, ${name}!"
"Hello, Nix!"
```

Variables defined in `let` are only available in the `in` body.

### `with` for Scope

`with` brings all attributes of a set into scope:

```nix
nix-repl> let mySet = { a = 1; b = 2; }; in with mySet; a + b
3
```

You'll see `with pkgs;` in Nix configs — it lets you write `nodejs` instead of `pkgs.nodejs`.

### `inherit` for Pulling Values

`inherit` pulls a variable into an attribute set with the same name:

```nix
nix-repl> let x = 1; y = 2; in { inherit x y; }
{ x = 1; y = 2; }
```

`inherit x;` is shorthand for `x = x;`.

### Strings and Multi-Line Strings

Single-line strings use double quotes:
```nix
nix-repl> "Hello, ${"world"}!"
"Hello, world!"
```

Multi-line strings use double single-quotes:
```nix
nix-repl> ''
  line one
  line two
''
"line one\nline two\n"
```

You'll use multi-line strings for `shellHook` scripts.

### Lists

Lists are space-separated (no commas):
```nix
nix-repl> [ 1 2 3 ]
[ 1 2 3 ]

nix-repl> [ "nodejs" "python" "curl" ]
[ "nodejs" "python" "curl" ]
```

You'll use lists for `buildInputs`.

Exit the repl:
```
nix-repl> :q
```

### Putting It Together: Reading a shell.nix

Now you can read this:

```nix
{ pkgs ? import <nixpkgs> {} }:     # Function with default argument

pkgs.mkShell {                       # Call mkShell with an attribute set
  buildInputs = with pkgs; [         # List of packages (with pkgs in scope)
    nodejs_20
    python312
  ];

  shellHook = ''                     # Multi-line string for setup script
    echo "Ready!"
  '';
}
```

Every piece of syntax here is something you just learned.

### Exercise 3.1: Exploring in nix repl

**Task:** In `nix repl`, create an attribute set representing a project with `name`, `version`, and `tools` (a list of strings). Access the `name` attribute.

<details>
<summary>Hint</summary>

Remember: attribute sets use `=` and end each attribute with `;`. Lists are space-separated.
</details>

<details>
<summary>Solution</summary>

```nix
nix-repl> project = { name = "myapp"; version = "1.0"; tools = [ "nodejs" "python" ]; }
nix-repl> project.name
"myapp"
nix-repl> project.tools
[ "nodejs" "python" ]
```
</details>

### Exercise 3.2: Writing a Function

**Task:** Write a function that takes a name and returns an attribute set with a `greeting` attribute. Call it with your name.

<details>
<summary>Hint 1</summary>

Functions use the colon syntax: `argument: body`
</details>

<details>
<summary>Hint 2</summary>

String interpolation: `"Hello, ${name}!"`
</details>

<details>
<summary>Solution</summary>

```nix
nix-repl> mkGreeting = name: { greeting = "Hello, ${name}!"; }
nix-repl> mkGreeting "Alice"
{ greeting = "Hello, Alice!"; }
nix-repl> (mkGreeting "Alice").greeting
"Hello, Alice!"
```
</details>

### Checkpoint 3

Before moving on, make sure you can:
- [ ] Read and write attribute sets with proper semicolons
- [ ] Understand function syntax (argument: body)
- [ ] Use `let/in` for local variables
- [ ] Explain what `with pkgs;` does
- [ ] Read a basic `shell.nix` file

---

## Section 4: Development Shells with shell.nix

### What is a Development Shell?

A `shell.nix` file declares what tools a project needs. When you run `nix-shell` in that directory, Nix gives you a shell with exactly those tools available. It's a declarative, reproducible alternative to installing tools globally.

Think of it as a Dockerfile for your development shell — but without containers. Your files, editor, and everything else stay exactly the same. You just get the right tools in your PATH.

### Your First shell.nix

Create a project directory:
```bash
mkdir ~/nix-demo && cd ~/nix-demo
```

Create `shell.nix`:
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

Enter the shell:
```bash
nix-shell
```

**Expected output:**
```
Dev environment loaded!
Node: v20.11.0
Python: Python 3.12.x
```

Now `node`, `python3`, and `npm` are all available. Exit with `exit` or Ctrl-d, and they're gone from your PATH.

### Understanding the Parts

- **`{ pkgs ? import <nixpkgs> {} }:`** — A function that takes a package set. The `?` provides a default value: import nixpkgs from your system's channel.
- **`pkgs.mkShell`** — Creates a shell environment (as opposed to building a package).
- **`buildInputs`** — The list of packages to make available. `with pkgs;` lets you write `nodejs_20` instead of `pkgs.nodejs_20`.
- **`shellHook`** — A bash script that runs once when you enter the shell. Good for setting environment variables, printing status, or activating tools.

### Pinning nixpkgs

The basic `shell.nix` above has a reproducibility gap: `<nixpkgs>` refers to whatever channel your system has configured. Different machines can have different channels, meaning different package versions.

To fix this, pin nixpkgs to a specific release:

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

Now everyone gets packages from the same nixpkgs release, regardless of their local system configuration.

This pinning approach works but is manual. Flakes (Section 5) automate and improve on this.

### Exercise 4.1: A Python Development Shell

**Task:** Create a `shell.nix` for a Python project that provides Python 3.12. Add a `shellHook` that creates a virtual environment if one doesn't exist and activates it.

<details>
<summary>Hint 1</summary>

Use an `if` statement in the shellHook to check for the `.venv` directory.
</details>

<details>
<summary>Hint 2</summary>

`python3 -m venv .venv` creates a virtual environment. `source .venv/bin/activate` activates it.
</details>

<details>
<summary>Solution</summary>

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
    python3 --version
  '';
}
```

Run `nix-shell` and verify:
- The first time, it creates `.venv` and activates it
- Subsequent times, it just activates the existing `.venv`
- `pip --version` works inside the shell
</details>

### Exercise 4.2: Adding Tools and Environment Variables

**Task:** Modify your shell.nix to also include `curl` and `jq` in `buildInputs`, and add environment variables `API_URL=http://localhost:3000` and `DEBUG=1` to the `shellHook`.

<details>
<summary>Hint</summary>

Add the packages to the `buildInputs` list and use `export` in the `shellHook`.
</details>

<details>
<summary>Solution</summary>

```nix
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python312
    python312Packages.pip
    curl
    jq
  ];

  shellHook = ''
    if [ ! -d ".venv" ]; then
      echo "Creating virtual environment..."
      python3 -m venv .venv
    fi
    source .venv/bin/activate
    export API_URL=http://localhost:3000
    export DEBUG=1
    echo "Python dev environment ready!"
    echo "API_URL=$API_URL"
  '';
}
```

Verify with:
```bash
nix-shell
echo $API_URL    # http://localhost:3000
echo $DEBUG      # 1
which curl       # /nix/store/...-curl-.../bin/curl
which jq         # /nix/store/...-jq-.../bin/jq
```
</details>

### Checkpoint 4

Before moving on, make sure you can:
- [ ] Create a `shell.nix` with `mkShell`, `buildInputs`, and `shellHook`
- [ ] Enter the shell with `nix-shell`
- [ ] Explain why pinning nixpkgs matters
- [ ] Add packages and environment variables to a shell

---

## Section 5: Flakes: The Modern Approach

### Why Flakes?

Plain `shell.nix` works, but it has gaps:

- **Manual pinning**: You have to manage the nixpkgs URL yourself.
- **Impure evaluation**: `<nixpkgs>` refers to a system-level channel that can change.
- **No lock file**: There's no automatic record of exactly which nixpkgs revision you're using.

Flakes solve all of these. A `flake.nix` declares its inputs explicitly, and Nix generates a `flake.lock` that pins every input to an exact revision — like `package-lock.json` for your entire dev environment.

### Flake Structure

A flake has two parts:

- **`inputs`**: Where your dependencies come from (usually just nixpkgs)
- **`outputs`**: What the flake provides (in our case, dev shells)

### Your First Flake

Flakes require a git repository:
```bash
mkdir ~/flake-demo && cd ~/flake-demo
git init
```

Create `flake.nix`:
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
          echo "Node: $(node --version)"
          echo "Python: $(python3 --version)"
        '';
      };
    };
}
```

**Important**: Change `system` to match your machine:
- `aarch64-darwin` — Apple Silicon Mac
- `x86_64-darwin` — Intel Mac
- `x86_64-linux` — Most Linux systems

Add the file to git (flakes only see tracked files):
```bash
git add flake.nix
```

Enter the dev shell:
```bash
nix develop
```

**Expected output (first run — takes a while):**
```
Flake dev environment ready!
Node: v20.11.0
Python: Python 3.12.x
```

### Understanding the Parts

- **`description`** — Human-readable description of the flake.
- **`inputs.nixpkgs.url`** — Where to get packages. `github:NixOS/nixpkgs/nixos-24.05` points to the 24.05 release branch of nixpkgs.
- **`outputs`** — A function that receives resolved inputs and returns what the flake provides. `self` is the flake itself; `nixpkgs` is the resolved nixpkgs input.
- **`devShells.${system}.default`** — The dev shell for this system architecture. `default` means it's what `nix develop` uses when you don't specify a name.

### The Lock File

After the first `nix develop`, check your directory:
```bash
ls
```

**Expected output:**
```
flake.lock  flake.nix
```

Look at the lock file:
```bash
cat flake.lock
```

It contains a JSON structure that records the exact git revision of nixpkgs that was used. **Commit this file.** It's what makes your environment reproducible across machines.

```bash
git add flake.lock
git commit -m "Add flake with Node.js and Python dev environment"
```

### Updating Dependencies

To update nixpkgs to the latest revision on the pinned branch:
```bash
nix flake update
```

This rewrites `flake.lock` with the latest revisions. It's like running `npm update` — you choose when to update, and the lock file records what changed.

### Supporting Multiple Systems

If your team includes macOS and Linux developers, you need to support multiple systems. Here's the pattern:

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
              echo "Dev environment ready!"
            '';
          };
        }
      );
    };
}
```

This uses `genAttrs` to generate a dev shell definition for each system in `supportedSystems`. Now `nix develop` works on any of those systems using the same flake.

### The Three Nix Shell Commands

These three commands sound similar but serve different purposes:

| Command | Works with | Purpose |
|---------|-----------|---------|
| `nix-shell` | `shell.nix`, `-p` flag | Enter a shell defined by shell.nix, or ad-hoc packages |
| `nix develop` | `flake.nix` | Enter a dev shell defined by a flake |
| `nix shell` | Flake package references | Get specific packages from a flake (e.g., `nix shell nixpkgs#nodejs`) |

Use `nix develop` for project dev environments (with a flake.nix). Use `nix-shell -p` for quick ad-hoc access to tools.

### Exercise 5.1: Convert shell.nix to a Flake

**Task:** Take the Python dev shell from Exercise 4.1 and convert it to a `flake.nix` using the multi-system pattern. Enter it with `nix develop`.

<details>
<summary>Hint 1</summary>

Start from the multi-system template above. Move your `buildInputs` and `shellHook` into the `mkShell` call.
</details>

<details>
<summary>Hint 2</summary>

Don't forget: `git init`, `git add flake.nix`, then `nix develop`.
</details>

<details>
<summary>Solution</summary>

```nix
{
  description = "Python project dev environment";

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
          };
        }
      );
    };
}
```

```bash
mkdir ~/flake-python && cd ~/flake-python
git init
# (create flake.nix with the content above)
git add flake.nix
nix develop
```
</details>

### Exercise 5.2: Named Dev Shells

**Task:** Add a second dev shell called `ci` alongside `default` in your flake. The `ci` shell should only include Python (no virtual environment setup in the hook). Enter it with `nix develop .#ci`.

<details>
<summary>Hint</summary>

Add another attribute next to `default` in the attribute set returned by the `forAllSystems` function.
</details>

<details>
<summary>Solution</summary>

```nix
devShells = forAllSystems (system:
  let pkgs = pkgsFor system;
  in {
    default = pkgs.mkShell {
      buildInputs = with pkgs; [
        python312
        python312Packages.pip
      ];

      shellHook = ''
        if [ ! -d ".venv" ]; then
          python3 -m venv .venv
        fi
        source .venv/bin/activate
        echo "Dev environment ready!"
      '';
    };

    ci = pkgs.mkShell {
      buildInputs = with pkgs; [
        python312
      ];

      shellHook = ''
        echo "CI environment ready!"
      '';
    };
  }
);
```

```bash
git add flake.nix
nix develop .#ci
# Only Python is available, no virtual environment
```
</details>

### Checkpoint 5

Before moving on, make sure you can:
- [ ] Create a `flake.nix` with inputs and outputs
- [ ] Enter a dev shell with `nix develop`
- [ ] Explain what `flake.lock` does and why you commit it
- [ ] Support multiple systems with `genAttrs`
- [ ] Explain the difference between `nix-shell`, `nix develop`, and `nix shell`

---

## Section 6: direnv Integration

### The Problem direnv Solves

Right now, you have to manually run `nix develop` every time you open a terminal and navigate to your project. direnv automates this: when you `cd` into a project directory, your dev environment activates automatically. When you leave, it deactivates.

If you've used mise, this is the same concept — tools activate automatically based on your current directory.

### Setting Up direnv

Install direnv and nix-direnv:
```bash
nix profile install nixpkgs#direnv nixpkgs#nix-direnv
```

Hook direnv into your shell. Add to `~/.zshrc`:
```bash
eval "$(direnv hook zsh)"
```

Or to `~/.bashrc`:
```bash
eval "$(direnv hook bash)"
```

Configure nix-direnv. Create or edit `~/.config/direnv/direnvrc`:
```bash
source $HOME/.nix-profile/share/nix-direnv/direnvrc
```

Reload your shell:
```bash
source ~/.zshrc  # or source ~/.bashrc
```

### Activating a Flake with direnv

In your flake project, create an `.envrc` file:
```bash
cd ~/flake-demo
echo "use flake" > .envrc
```

The first time, direnv blocks the `.envrc` for security. Approve it:
```bash
direnv allow
```

**Expected output:**
```
direnv: loading ~/flake-demo/.envrc
direnv: using flake
...
direnv: export +PROJECT_ROOT +buildInputs ...
```

Now test the automatic activation:
```bash
# Leave the directory
cd ~

# Come back
cd ~/flake-demo
```

Your tools are available without running `nix develop`. Leave the directory and they disappear.

### The .envrc File

For most projects, the `.envrc` is just two words:
```bash
use flake
```

You can also specify a particular dev shell:
```bash
# Use the "ci" dev shell
use flake .#ci
```

### Security Model

direnv requires explicit approval for each `.envrc` file via `direnv allow`. If the `.envrc` changes (someone modifies it, or you pull changes), you need to re-approve. This prevents malicious repos from running arbitrary code when you cd into them.

### Caching

nix-direnv caches the evaluated environment in `.direnv/` inside your project. The first activation is slow (it evaluates the flake), but subsequent activations are nearly instant. Add `.direnv/` to your `.gitignore`:

```bash
echo ".direnv/" >> .gitignore
```

### Exercise 6.1: Complete direnv Setup

**Task:** Set up direnv for your flake project. Verify that tools activate when you cd into the directory and deactivate when you leave.

<details>
<summary>Hint 1</summary>

Create `.envrc` with `use flake`, then run `direnv allow`.
</details>

<details>
<summary>Hint 2</summary>

Test by running `which node` inside the directory (should show a Nix store path) and outside (should show nothing or a system node).
</details>

<details>
<summary>Solution</summary>

```bash
cd ~/flake-demo

# Create .envrc
echo "use flake" > .envrc

# Approve it
direnv allow

# Leave and return
cd ~
cd ~/flake-demo

# Verify tools are available
which node
# /nix/store/...-nodejs-20.11.0/bin/node

node --version
# v20.11.0

# Leave
cd ~
which node
# node not found (or your system's node, not the Nix one)
```
</details>

### Checkpoint 6

Before moving on, make sure you can:
- [ ] Install and configure direnv with nix-direnv
- [ ] Create an `.envrc` that activates a flake
- [ ] Explain the security model (`direnv allow`)
- [ ] Verify that tools activate/deactivate when entering/leaving a directory

---

## Practice Project

### Project Description

Build a reproducible multi-language development environment that a team could use. You'll create a flake with direnv integration that provides multiple tools and sets up the project on shell entry.

### Requirements

1. Create a new project directory with `git init`
2. Write a `flake.nix` that provides:
   - Node.js 20
   - Python 3.12
   - `jq`
3. Support at least two systems (e.g., `aarch64-darwin` and `x86_64-linux`)
4. Add a `shellHook` that:
   - Prints the versions of all available tools
   - Sets a `PROJECT_NAME` environment variable
5. Create an `.envrc` with `use flake`
6. Create a `.gitignore` that excludes `.direnv/` and `.venv/`
7. Verify: cd out and back in — all tools should auto-activate
8. Commit `flake.nix`, `flake.lock`, `.envrc`, and `.gitignore`

### Getting Started

```bash
mkdir ~/nix-project && cd ~/nix-project
git init
```

Start with the multi-system flake template from Section 5 and customize it.

### Validation

After setup, verify the following:

```bash
# Leave and re-enter the directory
cd ~
cd ~/nix-project

# Check tools
node --version    # Should show v20.x
python3 --version # Should show 3.12.x
jq --version      # Should show jq-1.x

# Check environment variable
echo $PROJECT_NAME  # Should show your project name

# Check that flake.lock exists and is committed
git log --oneline
```

---

## Command Reference

| Command | Purpose |
|---------|---------|
| `nix-shell -p <packages>` | Temporary shell with ad-hoc packages |
| `nix-shell` | Enter shell defined by `shell.nix` in current directory |
| `nix develop` | Enter dev shell defined by `flake.nix` in current directory |
| `nix develop .#name` | Enter a named dev shell from a flake |
| `nix shell nixpkgs#pkg` | Get a specific package from a flake reference |
| `nix run nixpkgs#pkg -- args` | Run a package's default command |
| `nix search nixpkgs term` | Search for packages in nixpkgs |
| `nix flake update` | Update flake inputs and rewrite `flake.lock` |
| `nix flake check` | Validate a flake's structure |
| `nix repl` | Start an interactive Nix expression evaluator |
| `direnv allow` | Approve an `.envrc` file |
| `direnv status` | Show direnv diagnostic info for current directory |

## Summary

You've learned the fundamentals of Nix-based development environments:
- Nix stores packages by hash in `/nix/store`, enabling multiple versions and guaranteeing reproducibility
- `nix-shell -p` gives you ad-hoc temporary access to any package in nixpkgs
- The Nix language provides attribute sets, functions, `let/in`, and `with` — enough to write config files
- `shell.nix` declares a dev environment with `mkShell`, `buildInputs`, and `shellHook`
- Flakes add a `flake.lock` for automatic version pinning and hermetic evaluation
- direnv + nix-direnv automates shell activation when you enter a project directory

The typical team workflow: commit `flake.nix`, `flake.lock`, and `.envrc`. A new team member clones the repo, runs `direnv allow`, and gets the exact same environment as everyone else.

## Next Steps

Now that you understand Nix dev environments, explore:
- **mise-basics route** — A simpler alternative for version management; understand when each tool is appropriate
- **docker-dev-environments route** — How containers and Nix complement each other
- **Custom Nix packages** — Write your own derivations for project-specific tools
- **Overlays** — Customize or patch packages in nixpkgs
- **Home Manager** — Manage your entire user environment (dotfiles, tools, shell config) with Nix
- **NixOS** — A Linux distribution built entirely on Nix

## Additional Resources

- [Nix Reference Manual](https://nixos.org/manual/nix/stable/): Official Nix documentation
- [nixpkgs Manual](https://nixos.org/manual/nixpkgs/stable/): Package set documentation
- [search.nixos.org](https://search.nixos.org/packages): Web-based package search
- [Nix Pills](https://nixos.org/guides/nix-pills/): In-depth tutorial series
- [nix.dev](https://nix.dev/): Community documentation and tutorials
- [Determinate Systems Blog](https://determinate.systems/posts/): Practical Nix guides
- [Zero to Nix](https://zero-to-nix.com/): Beginner-friendly introduction
