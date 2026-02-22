---
title: Docker Dev Environments - Container-Based Development
route_map: /routes/docker-dev-environments/map.md
paired_sherpa: /routes/docker-dev-environments/sherpa.md
prerequisites:
  - Basic command line usage
topics:
  - Docker
  - Containers
  - Dev Containers
  - Docker Compose
  - Development Environments
---

# Docker Dev Environments - Guide (Human-Focused Content)

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/docker-dev-environments/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/docker-dev-environments/map.md` for the high-level overview.

## Overview

Docker lets you run applications inside isolated environments called containers. For development, this means every team member gets an identical setup, project dependencies don't pollute your host system, and onboarding is as simple as `docker compose up`. This guide teaches Docker for development environments — not production deployment. You'll learn to run containers, write Dockerfiles, use bind mounts for live code editing, set up multi-container dev environments with Docker Compose, configure dev containers for editor integration, and compose Docker with tools like mise and Nix.

## Learning Objectives

By the end of this guide, you will be able to:
- Explain the difference between containers and VMs
- Run, inspect, and manage Docker containers from the command line
- Write Dockerfiles with efficient layer caching for fast rebuilds
- Use bind mounts for live code editing inside containers
- Set up multi-container dev environments with Docker Compose
- Configure a dev container with devcontainer.json
- Explain when to use mise or Nix inside a Docker container

## Prerequisites

Before starting this guide, you should be familiar with:
- Using the command line/terminal (cd, ls, pwd, running programs)

Helpful but not required:
- The mise-basics route (for Section 6)
- The nix-dev-environments route (for Section 6)

## Setup

Install Docker on your system:

**macOS:**
Download and install Docker Desktop from https://www.docker.com/products/docker-desktop/

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install docker.io docker-compose-v2
sudo usermod -aG docker $USER
```
Log out and back in after adding yourself to the docker group.

**Linux (Fedora):**
```bash
sudo dnf install docker docker-compose
sudo systemctl start docker
sudo usermod -aG docker $USER
```
Log out and back in after adding yourself to the docker group.

**Verify installation:**
```bash
docker --version
docker compose version
```

**Expected output:**
```
Docker version 27.x.x, build xxxxxxx
Docker Compose version v2.x.x
```

---

## Section 1: What Docker Does

### The Problem

Setting up a development environment is one of the most tedious parts of software development. You need the right runtime version, the right system libraries, the right database, and the right configuration — all matching what the rest of the team uses. "Works on my machine" is the inevitable result when environments drift apart.

Docker solves this by packaging everything a project needs into an isolated container.

### Containers vs VMs

You might have used virtual machines (VMs) before. Containers solve a similar problem but with a different approach:

**Virtual Machines** run a complete operating system with its own kernel. Each VM includes a full OS, which makes them heavy (gigabytes of disk, minutes to start) but provides strong isolation.

**Containers** share the host operating system's kernel and isolate only the filesystem, processes, and network. This makes them lightweight (megabytes of disk, seconds to start) while still providing practical isolation for development.

```
Virtual Machine                     Container
┌────────────────┐                  ┌────────────────┐
│    Your App    │                  │    Your App    │
├────────────────┤                  ├────────────────┤
│   Guest OS     │                  │   Libraries    │
│   (full kernel)│                  │   (no kernel)  │
├────────────────┤                  └───────┬────────┘
│   Hypervisor   │                          │
├────────────────┤                  ┌───────┴────────┐
│    Host OS     │                  │    Host OS     │
└────────────────┘                  │   (shared      │
                                    │    kernel)     │
                                    └────────────────┘
```

Think of it this way: a VM is like renting a whole apartment (with its own plumbing, electricity, and walls). A container is like having your own room in a shared house — you have privacy, but you share the plumbing and electricity.

**On macOS**: Docker Desktop runs a lightweight Linux VM behind the scenes because containers need a Linux kernel. You don't interact with this VM directly — Docker Desktop handles it transparently.

### Images vs Containers

Two terms you'll see constantly:

- **Image**: A read-only snapshot of a filesystem. It contains the OS base, installed packages, your code, and configuration. Think of it as a class definition.
- **Container**: A running instance of an image. It has its own writable layer on top of the image. Think of it as an object — an instance of that class.

You can create many containers from the same image, just like you can create many objects from the same class.

### When Docker Makes Sense for Dev

Docker is worth the setup overhead when:
- Your project has complex dependencies (databases, specific runtime versions, system libraries)
- Team members need identical environments
- You want project dependencies isolated from your host system
- You want new developers to start coding quickly

Docker is probably overkill when:
- Your project has a single, simple dependency (just Node, just Python)
- You're working solo and your local setup works fine
- Your project has no external services (no databases, no message queues)

### Checkpoint 1

Before moving on, make sure you understand:
- [ ] How containers differ from VMs (shared kernel vs full OS)
- [ ] The relationship between images and containers (blueprint vs running instance)
- [ ] When Docker makes sense for development

---

## Section 2: Docker Fundamentals

### Running Your First Container

The most fundamental Docker command is `docker run`. It pulls an image (if needed), creates a container, and starts it:

```bash
docker run -it ubuntu:22.04 bash
```

**What the flags mean:**
- `-i` — keep STDIN open (interactive)
- `-t` — allocate a terminal (so you get a prompt)
- `ubuntu:22.04` — the image name and tag
- `bash` — the command to run inside the container

**You should see:**
```
root@a1b2c3d4e5f6:/#
```

You're now inside an Ubuntu container. Try some commands:

```bash
cat /etc/os-release
# Shows Ubuntu 22.04

ls /
# Shows the container's filesystem

whoami
# root
```

Type `exit` to leave the container.

### Listing Containers

After exiting, the container has stopped but still exists:

```bash
# Show running containers
docker ps
```

**Expected output:**
```
CONTAINER ID   IMAGE   COMMAND   CREATED   STATUS   PORTS   NAMES
```

Nothing running. But the stopped container is still there:

```bash
# Show all containers, including stopped ones
docker ps -a
```

**Expected output:**
```
CONTAINER ID   IMAGE          COMMAND   CREATED          STATUS                     PORTS   NAMES
a1b2c3d4e5f6   ubuntu:22.04   "bash"    2 minutes ago    Exited (0) 1 minute ago            quirky_darwin
```

### Running in the Background

Not every container needs an interactive shell. To run a container in the background:

```bash
docker run -d --name mywebserver -p 8080:80 nginx:latest
```

**What the flags mean:**
- `-d` — run in the background (detached)
- `--name mywebserver` — give the container a name (instead of a random one)
- `-p 8080:80` — map port 8080 on your host to port 80 in the container

**Expected output:**
```
Unable to find image 'nginx:latest' locally
latest: Pulling from library/nginx
...
Status: Downloaded newer image for nginx:latest
e5f6a7b8c9d0...
```

Open http://localhost:8080 in your browser — you should see the Nginx welcome page.

### Running Commands in a Running Container

`docker exec` lets you run commands inside a container that's already running:

```bash
docker exec -it mywebserver bash
```

You're now inside the running Nginx container. You can look around:

```bash
cat /etc/nginx/nginx.conf
ls /usr/share/nginx/html/
```

Type `exit` to leave — the container keeps running.

### Stopping and Removing Containers

```bash
# Stop a running container
docker stop mywebserver

# Remove a stopped container
docker rm mywebserver

# Or force-stop and remove in one step
docker rm -f mywebserver
```

### Listing Images

```bash
docker images
```

**Expected output:**
```
REPOSITORY   TAG       IMAGE ID       CREATED       SIZE
nginx        latest    a1b2c3d4e5f6   2 weeks ago   187MB
ubuntu       22.04     f6e5d4c3b2a1   3 weeks ago   77.8MB
```

Images stay on your machine even after containers are removed. To remove an image:

```bash
docker rmi nginx:latest
```

### Exercise 2.1: Run an Interactive Python Container

**Task:** Run a Python 3.12 container, start the Python REPL, print "Hello from Docker", and exit.

<details>
<summary>Hint 1</summary>

The Python image is called `python` with the tag `3.12`.
</details>

<details>
<summary>Hint 2</summary>

Use `docker run -it python:3.12 python` to start an interactive Python REPL.
</details>

<details>
<summary>Solution</summary>

```bash
docker run -it python:3.12 python
```

Inside the Python REPL:
```python
print('Hello from Docker')
# Output: Hello from Docker
exit()
```

</details>

### Exercise 2.2: Run Nginx in the Background

**Task:** Run an Nginx container in the background with the name "webtest" mapped to port 9090 on your host. Verify it's running with `docker ps`. Open it in a browser. Then stop and remove it.

<details>
<summary>Hint 1</summary>

You need `-d` for background, `--name` for the name, and `-p host:container` for port mapping.
</details>

<details>
<summary>Hint 2</summary>

Nginx listens on port 80 inside the container. You want to map your host's port 9090 to the container's port 80.
</details>

<details>
<summary>Solution</summary>

```bash
# Start the container
docker run -d --name webtest -p 9090:80 nginx:latest

# Verify it's running
docker ps
# Should show: webtest ... 0.0.0.0:9090->80/tcp

# Open http://localhost:9090 in your browser

# Stop and remove
docker stop webtest
docker rm webtest
```

</details>

### Checkpoint 2

Before moving on, make sure you can:
- [ ] Run containers interactively with `docker run -it`
- [ ] Run containers in the background with `docker run -d`
- [ ] List running and stopped containers with `docker ps` and `docker ps -a`
- [ ] Execute commands in a running container with `docker exec`
- [ ] Stop and remove containers

---

## Section 3: Writing Dockerfiles

### What is a Dockerfile?

A Dockerfile is a text file with instructions for building a Docker image. Each instruction creates a layer in the image. Think of it as a recipe: start with a base, add ingredients, configure, and describe what to do when you open the box.

### Basic Dockerfile

Create a file called `Dockerfile` (no extension):

```dockerfile
FROM node:20-slim

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 3000
CMD ["node", "server.js"]
```

**What each instruction does:**

| Instruction | Purpose |
|------------|---------|
| `FROM node:20-slim` | Start from a Node.js 20 base image (slim = smaller) |
| `WORKDIR /app` | Set the working directory inside the container |
| `COPY package*.json ./` | Copy package.json (and package-lock.json if it exists) |
| `RUN npm install` | Install dependencies |
| `COPY . .` | Copy the rest of the source code |
| `EXPOSE 3000` | Document that the app uses port 3000 (informational only) |
| `CMD ["node", "server.js"]` | Default command when the container starts |

Build the image:
```bash
docker build -t myapp .
```

- `-t myapp` tags the image with the name "myapp"
- `.` tells Docker the build context is the current directory

Run it:
```bash
docker run -d --name myapp -p 3000:3000 myapp
```

### Layer Caching: Why Instruction Order Matters

This is the single most important concept for writing efficient Dockerfiles.

Docker caches each layer. When you rebuild, Docker checks each instruction: if the instruction and its inputs haven't changed, it uses the cached layer. But once a layer changes, all subsequent layers must also rebuild.

Watch what happens when you build twice:

```bash
# First build: every step runs
docker build -t myapp .

# Change a source file, then build again
docker build -t myapp .
```

**Second build output (abridged):**
```
Step 1/7 : FROM node:20-slim
 ---> Using cache
Step 2/7 : WORKDIR /app
 ---> Using cache
Step 3/7 : COPY package*.json ./
 ---> Using cache
Step 4/7 : RUN npm install
 ---> Using cache
Step 5/7 : COPY . .
 ---> abc123def456
Step 6/7 : EXPOSE 3000
 ---> Running in ...
Step 7/7 : CMD ["node", "server.js"]
 ---> Running in ...
```

Steps 1-4 used the cache because `package.json` didn't change. Only the source code copy and later steps re-ran. This saves the time of a full `npm install` on every code change.

**The wrong way** (don't do this):
```dockerfile
FROM node:20-slim
WORKDIR /app
COPY . .          # Every code change invalidates this layer...
RUN npm install   # ...which forces this to re-run every time
CMD ["node", "server.js"]
```

**The rule**: put things that change rarely (dependency installation) before things that change often (source code).

### Multi-Stage Builds

Multi-stage builds use multiple `FROM` statements. Each stage can copy files from a previous stage. This lets you separate build tools from the final image:

```dockerfile
FROM node:20 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM node:20-slim
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
CMD ["node", "dist/server.js"]
```

The first stage (`builder`) has everything needed to build. The final stage only has the compiled output and runtime dependencies. The build tools don't end up in the final image.

### .dockerignore

A `.dockerignore` file tells Docker which files to exclude from the build context. Without it, `docker build` sends everything in the directory to the Docker daemon — including `node_modules`, `.git`, and any other large files.

Create a `.dockerignore`:
```
node_modules
.git
.gitignore
*.md
.env
.env.*
.vscode
.idea
dist
coverage
```

This speeds up builds and prevents accidentally including secrets or unnecessary files in the image.

### Exercise 3.1: Write a Python Dockerfile

**Task:** Write a Dockerfile for a Python Flask app with a `requirements.txt` and an `app.py`. Optimize for layer caching so that changing `app.py` doesn't reinstall dependencies.

<details>
<summary>Hint 1</summary>

The Python equivalent of "copy package.json, then install" is "copy requirements.txt, then pip install."
</details>

<details>
<summary>Hint 2</summary>

Use `FROM python:3.12-slim` as the base image and `pip install -r requirements.txt` to install dependencies.
</details>

<details>
<summary>Solution</summary>

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "app.py"]
```

The key is that `requirements.txt` is copied and installed before the rest of the source code. Changing `app.py` doesn't invalidate the dependency installation layer.
</details>

### Exercise 3.2: Create a .dockerignore

**Task:** Create a `.dockerignore` file for a Node.js project. List everything that should be excluded from the Docker build context.

<details>
<summary>Hint</summary>

Think about: installed dependencies (built in the container), version control files, editor configs, environment files, build output, test coverage reports.
</details>

<details>
<summary>Solution</summary>

```
node_modules
.git
.gitignore
*.md
.env
.env.*
.vscode
.idea
dist
coverage
.nyc_output
```

</details>

### Checkpoint 3

Before moving on, make sure you can:
- [ ] Write a Dockerfile with FROM, WORKDIR, COPY, RUN, and CMD
- [ ] Explain why instruction order matters for layer caching
- [ ] Describe what multi-stage builds accomplish
- [ ] Create a .dockerignore file

---

## Section 4: Volumes and Bind Mounts

### The Dev Workflow Problem

So far, your code gets baked into the Docker image at build time. If you change a file on your host, the container still has the old version. You'd have to rebuild the image every time you change a line of code. That's terrible for development.

Bind mounts fix this.

### Bind Mounts

A bind mount maps a directory on your host into the container. Changes on either side are immediately visible to the other — no copying, no rebuilding.

```bash
docker run -it \
  -v $(pwd):/app \
  -w /app \
  -p 3000:3000 \
  node:20 \
  npm run dev
```

**What this does:**
- `-v $(pwd):/app` — maps your current directory to `/app` in the container
- `-w /app` — sets the working directory to `/app`
- Now editing a file on your host is the same as editing it in the container

Try it yourself:

```bash
# Create a simple script
echo "console.log('version 1')" > index.js

# Run it in a container with a bind mount
docker run --rm -v $(pwd):/app -w /app node:20 node index.js
```

**Expected output:**
```
version 1
```

Now change the file and run it again:

```bash
echo "console.log('version 2')" > index.js
docker run --rm -v $(pwd):/app -w /app node:20 node index.js
```

**Expected output:**
```
version 2
```

The container sees the updated file immediately because of the bind mount.

### The node_modules Problem

There's a common gotcha with Node.js projects. If you bind mount your entire project directory, the host's `node_modules` (which might be empty or have macOS-specific binaries) overwrites the container's `node_modules`.

The fix is to use an anonymous volume for `node_modules`:

```bash
docker run -it \
  -v $(pwd):/app \
  -v /app/node_modules \
  -w /app \
  -p 3000:3000 \
  node:20 \
  sh -c "npm install && npm run dev"
```

The second `-v /app/node_modules` (with no host path) creates an anonymous volume that "masks" the bind mount for just that directory. The container uses its own installed modules while your source code comes from the host.

### Named Volumes

Named volumes persist data independently of any container's lifecycle. They're managed by Docker, not mapped to a specific host path. Use them for data that should survive container deletion — like a database's data files.

```bash
# Create a named volume
docker volume create pgdata

# Run Postgres with the volume
docker run -d \
  --name devdb \
  -v pgdata:/var/lib/postgresql/data \
  -e POSTGRES_PASSWORD=devpass \
  postgres:16
```

Even if you `docker rm devdb`, the `pgdata` volume still exists:

```bash
docker volume ls
```

**Expected output:**
```
DRIVER    VOLUME NAME
local     pgdata
```

Start a new container with the same volume, and your data is still there.

To remove a volume when you're done:
```bash
docker volume rm pgdata
```

### tmpfs Mounts

tmpfs mounts store data in memory only — it disappears when the container stops. Useful for temporary data or secrets you don't want written to disk:

```bash
docker run -it --tmpfs /tmp ubuntu:22.04 bash
```

### Quick Reference: When to Use What

| Type | Use For | Persistence |
|------|---------|-------------|
| Bind mount | Source code (live editing) | Matches host filesystem |
| Named volume | Database data, caches | Survives container deletion |
| tmpfs | Temp files, secrets | Gone when container stops |

### Docker Compose

Typing long `docker run` commands gets old fast. Docker Compose lets you define multi-container setups in a `docker-compose.yml` file:

```yaml
services:
  app:
    build: .
    volumes:
      - .:/app
      - /app/node_modules
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=development
    command: npm run dev

  db:
    image: postgres:16
    volumes:
      - pgdata:/var/lib/postgresql/data
    environment:
      - POSTGRES_PASSWORD=devpass
    ports:
      - "5432:5432"

volumes:
  pgdata:
```

**Start everything:**
```bash
docker compose up
```

**Start in the background:**
```bash
docker compose up -d
```

**See running services:**
```bash
docker compose ps
```

**View logs:**
```bash
docker compose logs -f app
```

**Run a command in a service:**
```bash
docker compose exec app bash
```

**Stop everything:**
```bash
docker compose down
```

**Stop everything AND delete volumes (fresh start):**
```bash
docker compose down -v
```

Notice what this replaces: instead of remembering `-v $(pwd):/app -v /app/node_modules -p 3000:3000 -e NODE_ENV=development`, you write it once in YAML and run `docker compose up`.

### Exercise 4.1: Bind Mount for Live Editing

**Task:** Create a simple `index.js` that logs a message. Run it in a Node container with a bind mount. Change the message on your host and run it again to verify the container sees the change.

<details>
<summary>Hint</summary>

Use `docker run --rm -v $(pwd):/app -w /app node:20 node index.js`. The `--rm` flag removes the container after it exits.
</details>

<details>
<summary>Solution</summary>

```bash
# Create the file
echo "console.log('hello from docker')" > index.js

# Run with bind mount
docker run --rm -v $(pwd):/app -w /app node:20 node index.js
# Output: hello from docker

# Change the file
echo "console.log('updated!')" > index.js

# Run again — sees the change
docker run --rm -v $(pwd):/app -w /app node:20 node index.js
# Output: updated!
```

</details>

### Exercise 4.2: Docker Compose with App and Database

**Task:** Write a `docker-compose.yml` that runs a Node.js app with bind mounts and a Postgres database with a named volume. Start it up and verify both services are running.

<details>
<summary>Hint 1</summary>

You need two services under the `services` key: one for the app and one for the database.
</details>

<details>
<summary>Hint 2</summary>

The app service needs `build`, `volumes`, `ports`, and `command`. The db service needs `image`, `volumes`, `environment`, and `ports`. Don't forget to declare the named volume at the top level.
</details>

<details>
<summary>Solution</summary>

`docker-compose.yml`:
```yaml
services:
  app:
    build: .
    volumes:
      - .:/app
      - /app/node_modules
    ports:
      - "3000:3000"
    command: npm run dev

  db:
    image: postgres:16
    volumes:
      - pgdata:/var/lib/postgresql/data
    environment:
      - POSTGRES_PASSWORD=devpass
    ports:
      - "5432:5432"

volumes:
  pgdata:
```

```bash
# Start everything
docker compose up -d

# Verify both services are running
docker compose ps

# Expected output shows both app and db services running
```

</details>

### Checkpoint 4

Before moving on, make sure you can:
- [ ] Explain the dev workflow problem (code changes not reflected in containers)
- [ ] Use bind mounts for live code editing
- [ ] Use named volumes for data persistence
- [ ] Write a docker-compose.yml with multiple services
- [ ] Start and stop services with `docker compose up` and `docker compose down`

---

## Section 5: Dev Containers

### What Are Dev Containers?

Dev containers take Docker for development one step further. Instead of just running your code in a container, your entire editor runs inside the container. Your terminal, debugger, extensions, and settings all run in the container. It feels like working locally, but everything is containerized.

The `devcontainer.json` specification describes what the dev environment should look like. VS Code has the best support (via the Dev Containers extension), but JetBrains IDEs, GitHub Codespaces, and the standalone Dev Container CLI also support the spec.

### Basic devcontainer.json

Create a `.devcontainer/devcontainer.json` in your project:

```json
{
  "name": "My Project",
  "image": "mcr.microsoft.com/devcontainers/javascript-node:20",
  "forwardPorts": [3000],
  "postCreateCommand": "npm install",
  "customizations": {
    "vscode": {
      "extensions": [
        "dbaeumer.vscode-eslint",
        "esbenp.prettier-vscode"
      ],
      "settings": {
        "editor.formatOnSave": true
      }
    }
  }
}
```

**What each field does:**

| Field | Purpose |
|-------|---------|
| `name` | Display name for the dev container |
| `image` | Base Docker image to use |
| `forwardPorts` | Ports to forward from container to host |
| `postCreateCommand` | Runs after the container is created (install deps, etc.) |
| `customizations.vscode.extensions` | VS Code extensions to install inside the container |
| `customizations.vscode.settings` | VS Code settings to apply |

To use it in VS Code:
1. Install the "Dev Containers" extension
2. Open your project
3. Click the blue button in the bottom-left corner, or use the command palette: "Dev Containers: Reopen in Container"
4. VS Code reopens inside the container with all your specified tools and extensions

### Using a Dockerfile

For more control over the environment, point to a Dockerfile instead of a pre-built image:

```json
{
  "name": "My Project",
  "build": {
    "dockerfile": "Dockerfile",
    "context": ".."
  },
  "forwardPorts": [3000],
  "postCreateCommand": "npm install"
}
```

Place the `Dockerfile` in the `.devcontainer/` directory (or point to one elsewhere). The `context` sets the build context relative to the `devcontainer.json` location.

### Dev Container Features

Features are reusable modules that add tools to your dev container without requiring you to write Dockerfile instructions. They're like plugins:

```json
{
  "name": "My Project",
  "image": "mcr.microsoft.com/devcontainers/base:ubuntu",
  "features": {
    "ghcr.io/devcontainers/features/node:1": {
      "version": "20"
    },
    "ghcr.io/devcontainers/features/python:1": {
      "version": "3.12"
    },
    "ghcr.io/devcontainers/features/docker-in-docker:2": {}
  }
}
```

Instead of writing `RUN apt-get install ...` for each tool, you declare what you need and features handle installation. Browse available features at https://containers.dev/features.

### With Docker Compose

Dev containers can use an existing Docker Compose setup:

```json
{
  "name": "Full Stack",
  "dockerComposeFile": "docker-compose.yml",
  "service": "app",
  "workspaceFolder": "/app",
  "forwardPorts": [3000, 5432],
  "postCreateCommand": "npm install"
}
```

This tells the dev container to use your Docker Compose file, attach to the `app` service, and open the editor in the `/app` directory.

### When Dev Containers Make Sense

**Use dev containers when:**
- Your team uses VS Code (or a compatible editor)
- You want standardized extensions and settings across the team
- You want one-click setup for new team members
- You use GitHub Codespaces

**Use plain Docker when:**
- Your team uses diverse editors that don't support the spec
- You only need the runtime environment, not editor integration
- You prefer managing your own editor setup

### Exercise 5.1: Create a Dev Container for Python

**Task:** Create a `devcontainer.json` for a Python project. It should use a Python base image, install the Python VS Code extension, forward port 8000, and run `pip install -r requirements.txt` after creation.

<details>
<summary>Hint 1</summary>

The Microsoft Python dev container image is `mcr.microsoft.com/devcontainers/python:3.12`.
</details>

<details>
<summary>Hint 2</summary>

The VS Code Python extension identifier is `ms-python.python`.
</details>

<details>
<summary>Solution</summary>

`.devcontainer/devcontainer.json`:
```json
{
  "name": "Python Project",
  "image": "mcr.microsoft.com/devcontainers/python:3.12",
  "forwardPorts": [8000],
  "postCreateCommand": "pip install -r requirements.txt",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python"
      ]
    }
  }
}
```

</details>

### Exercise 5.2: Use Features for a Multi-Language Project

**Task:** Create a `devcontainer.json` that starts from a base Ubuntu image and uses features to add Python 3.12 and Node.js 20. Forward ports 8000 and 3000.

<details>
<summary>Hint</summary>

Use `mcr.microsoft.com/devcontainers/base:ubuntu` as the image and add features for Python and Node.
</details>

<details>
<summary>Solution</summary>

`.devcontainer/devcontainer.json`:
```json
{
  "name": "Python + Node Project",
  "image": "mcr.microsoft.com/devcontainers/base:ubuntu",
  "features": {
    "ghcr.io/devcontainers/features/python:1": {
      "version": "3.12"
    },
    "ghcr.io/devcontainers/features/node:1": {
      "version": "20"
    }
  },
  "forwardPorts": [8000, 3000],
  "postCreateCommand": "pip install -r requirements.txt && npm install",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "dbaeumer.vscode-eslint"
      ]
    }
  }
}
```

</details>

### Checkpoint 5

Before moving on, make sure you can:
- [ ] Explain what dev containers add on top of plain Docker
- [ ] Create a basic devcontainer.json with an image, ports, and extensions
- [ ] Describe what features are and when to use them
- [ ] Explain when dev containers make sense vs plain Docker

---

## Section 6: Composing with mise and Nix

### The Problem Inside the Container

Docker gives you an isolated environment, but inside that environment, you still need the right versions of your tools. You can hardcode versions in `RUN` instructions in your Dockerfile, but that gets messy when you need multiple runtimes at specific versions. This is where mise and Nix can help.

### mise Inside Docker

mise is a tool version manager that handles Node, Python, Go, Ruby, and many other runtimes. It reads a `.mise.toml` configuration file and installs the specified versions. Using mise inside Docker means your container uses the same tool versions defined in the same config file you use locally.

**`.mise.toml`:**
```toml
[tools]
node = "20"
python = "3.12"
```

**Dockerfile:**
```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y curl git

# Install mise
RUN curl https://mise.run | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy mise config and install tools
COPY .mise.toml .
RUN mise install

WORKDIR /app
COPY . .
```

Build and verify:
```bash
docker build -t mise-dev .
docker run --rm mise-dev node --version
# v20.x.x

docker run --rm mise-dev python --version
# Python 3.12.x
```

**When mise inside Docker makes sense:**
- You already use mise locally and want the container to match
- You need multiple runtimes (Node + Python + Go) at specific versions
- You want one config file (`.mise.toml`) that works for both local and containerized dev

**When it doesn't:**
- You only need one runtime — just use a specific Docker image tag like `node:20`
- You don't use mise locally — no benefit to adding it to the container

### Nix Inside Docker

Nix takes a fundamentally different approach. Instead of managing tool versions, Nix builds everything from explicit package definitions that produce the same output every time. A Nix expression is like a mathematical function — same inputs, same outputs, always.

**Multi-stage build with Nix:**
```dockerfile
FROM nixos/nix:latest AS builder
COPY . /src
WORKDIR /src
RUN nix build

FROM ubuntu:22.04
COPY --from=builder /src/result/bin/myapp /usr/local/bin/
CMD ["myapp"]
```

The first stage uses Nix to build the project. The second stage copies just the build output into a minimal image. The final image doesn't need Nix installed.

**For dev environments, Nix can create a dev shell:**
```dockerfile
FROM nixos/nix:latest

COPY . /src
WORKDIR /src

# Build the dev shell and cache its dependencies
RUN nix develop --command echo "Dev shell ready"

CMD ["nix", "develop"]
```

**When Nix inside Docker makes sense:**
- You need bit-for-bit reproducibility (same inputs always produce the same outputs)
- Your project already uses Nix (`flake.nix`, `shell.nix`)
- You want to build minimal production images from Nix derivations

**When it doesn't:**
- You're not already using Nix — the learning curve is steep
- Your reproducibility needs are met by pinning versions in a Dockerfile

### Comparison

| | mise | Nix |
|---|---|---|
| Learning curve | Low — similar to nvm/pyenv | High — new paradigm |
| Config file | `.mise.toml` | `flake.nix` / `shell.nix` |
| What it pins | Tool versions (e.g., Node 20, Python 3.12) | Everything, including system libraries |
| How it installs | Downloads pre-built binaries | May build from source (cached after first build) |
| Adoption effort | Drop-in, works alongside existing tools | Requires rethinking dependency management |

### When to Compose Tools vs Use One

- **Docker alone**: sufficient for most dev environments. A well-written Dockerfile with specific image tags handles single-runtime projects.
- **Docker + mise**: good when you need multiple runtimes at specific versions and want a simple, readable config.
- **Docker + Nix**: good when you need maximum reproducibility or your team already uses Nix.
- **Nix alone (no Docker)**: possible for local dev, but Docker adds isolation from the host system.

### Exercise 6.1: mise Inside Docker

**Task:** Write a Dockerfile that uses mise to install Node 20 and Python 3.12. Create the corresponding `.mise.toml`. Build the image and verify both tools are available.

<details>
<summary>Hint 1</summary>

Start with `FROM ubuntu:22.04` and install curl and git with apt-get.
</details>

<details>
<summary>Hint 2</summary>

Install mise with `curl https://mise.run | sh` and add it to PATH with `ENV PATH="/root/.local/bin:$PATH"`.
</details>

<details>
<summary>Solution</summary>

`.mise.toml`:
```toml
[tools]
node = "20"
python = "3.12"
```

`Dockerfile`:
```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y curl git

RUN curl https://mise.run | sh
ENV PATH="/root/.local/bin:$PATH"

COPY .mise.toml .
RUN mise install

WORKDIR /app
COPY . .
```

```bash
docker build -t mise-dev .

docker run --rm mise-dev node --version
# v20.x.x

docker run --rm mise-dev python --version
# Python 3.12.x
```

</details>

### Exercise 6.2: Multi-Stage Nix Build

**Task:** Write a multi-stage Dockerfile where the first stage uses Nix to build a project and the second stage copies just the build output into a minimal Ubuntu image.

<details>
<summary>Hint</summary>

Use `FROM nixos/nix:latest AS builder` for the first stage. Build with `nix build`. In the second stage, `COPY --from=builder` to pull the output.
</details>

<details>
<summary>Solution</summary>

```dockerfile
FROM nixos/nix:latest AS builder
COPY . /src
WORKDIR /src
RUN nix build

FROM ubuntu:22.04
COPY --from=builder /src/result/bin/myapp /usr/local/bin/
CMD ["myapp"]
```

This assumes the project has a `flake.nix` that defines how to build `myapp`. The final image is small because it only contains the build output, not Nix or any build tools.
</details>

### Checkpoint 6

Before moving on, make sure you understand:
- [ ] When to use mise inside Docker (multiple runtimes, shared config with local)
- [ ] When to use Nix inside Docker (full reproducibility, existing Nix usage)
- [ ] The trade-offs between mise and Nix
- [ ] When composing tools adds value vs when Docker alone is enough

---

## Practice Project

### Project Description

Build a development environment for a project with both a Node.js API and a Python data processing script. You'll use Docker Compose with bind mounts for live editing and mise for tool management inside the container.

### Requirements

1. A `Dockerfile` that uses mise to install Node 20 and Python 3.12
2. A `docker-compose.yml` with:
   - An `app` service using bind mounts for live code editing
   - A `db` service running Postgres with a named volume
3. A simple Node.js API (`server.js`) that responds to HTTP requests
4. A Python script (`process.py`) that can be run inside the same container
5. Editing code on the host should be reflected in the container

### Starter Files

**`.mise.toml`:**
```toml
[tools]
node = "20"
python = "3.12"
```

**`server.js`:**
```javascript
const http = require('http');
const server = http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ message: 'Hello from Docker dev environment' }));
});
server.listen(3000, () => console.log('Server running on port 3000'));
```

**`process.py`:**
```python
import json
data = {"processed": True, "source": "python in docker"}
print(json.dumps(data, indent=2))
```

### Building It Step by Step

**Step 1: Write the Dockerfile**

<details>
<summary>Hint</summary>

Use the mise-inside-Docker pattern from Section 6. Start from `ubuntu:22.04`, install mise, copy `.mise.toml`, install tools.
</details>

<details>
<summary>Solution</summary>

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y curl git

RUN curl https://mise.run | sh
ENV PATH="/root/.local/bin:$PATH"

COPY .mise.toml .
RUN mise install

WORKDIR /app
COPY . .

EXPOSE 3000
CMD ["node", "server.js"]
```

</details>

**Step 2: Write docker-compose.yml**

<details>
<summary>Hint</summary>

You need two services: `app` (build from Dockerfile, bind mount, port 3000) and `db` (Postgres image, named volume, port 5432).
</details>

<details>
<summary>Solution</summary>

```yaml
services:
  app:
    build: .
    volumes:
      - .:/app
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=development
    command: node server.js

  db:
    image: postgres:16
    volumes:
      - pgdata:/var/lib/postgresql/data
    environment:
      - POSTGRES_PASSWORD=devpass
    ports:
      - "5432:5432"

volumes:
  pgdata:
```

</details>

**Step 3: Start it up and verify**

```bash
# Build and start
docker compose up -d

# Verify services are running
docker compose ps

# Test the Node.js API
curl http://localhost:3000
# {"message":"Hello from Docker dev environment"}

# Run the Python script inside the container
docker compose exec app python process.py
# {
#   "processed": true,
#   "source": "python in docker"
# }
```

**Step 4: Test live editing**

Edit `server.js` on your host — change the message. Restart the app service to see the change:

```bash
docker compose restart app
curl http://localhost:3000
# Should show your updated message
```

**Step 5: Verify database persistence**

```bash
# Connect to Postgres
docker compose exec db psql -U postgres -c "CREATE TABLE test (id serial PRIMARY KEY, name text);"
docker compose exec db psql -U postgres -c "INSERT INTO test (name) VALUES ('Docker is working');"
docker compose exec db psql -U postgres -c "SELECT * FROM test;"

# Stop and restart — data should persist
docker compose down
docker compose up -d
docker compose exec db psql -U postgres -c "SELECT * FROM test;"
# Should still show the row you inserted
```

### Validation Checklist

- [ ] `docker compose up` starts both services without errors
- [ ] `curl http://localhost:3000` returns a JSON response
- [ ] `docker compose exec app python process.py` runs Python inside the container
- [ ] Editing files on the host changes files in the container
- [ ] Database data persists across `docker compose down` and `docker compose up`

### Clean Up

When you're done:
```bash
# Stop services and remove containers
docker compose down

# Stop services, remove containers, AND delete volumes (database data)
docker compose down -v
```

---

## Summary

You've learned how to use Docker for development environments:
- **Containers** provide isolated environments that share the host kernel — lighter than VMs, faster to start
- **Images** are blueprints; **containers** are running instances
- **Dockerfiles** are recipes for building images — instruction order matters for layer caching
- **Bind mounts** link host directories into containers for live code editing
- **Named volumes** persist data (like databases) independently of containers
- **Docker Compose** defines multi-container setups in YAML — one command to start everything
- **Dev containers** add editor integration on top of Docker for a standardized dev experience
- **mise** and **Nix** can manage tool versions inside containers when a single Dockerfile isn't enough

## Next Steps

Now that you can set up Docker dev environments, explore:
- **mise-basics** — Tool version management with mise (if not already completed)
- **nix-dev-environments** — Fully reproducible development environments with Nix
- **agent-sandboxing** — Running AI agents safely in containers

## Additional Resources

- [Docker Documentation](https://docs.docker.com/) — Official reference
- [Docker Compose Documentation](https://docs.docker.com/compose/) — Compose file reference
- [Dev Containers Specification](https://containers.dev/) — The devcontainer.json spec
- [Dev Container Features](https://containers.dev/features) — Browse available features
- [mise Documentation](https://mise.jdx.dev/) — mise tool version manager
- [Nix Docker Integration](https://nixos.wiki/wiki/Docker) — Using Nix with Docker
