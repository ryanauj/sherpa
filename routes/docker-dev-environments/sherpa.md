---
title: Docker Dev Environments - Container-Based Development
route_map: /routes/docker-dev-environments/map.md
paired_guide: /routes/docker-dev-environments/guide.md
topics:
  - Docker
  - Containers
  - Dev Containers
  - Docker Compose
  - Development Environments
---

# Docker Dev Environments - Sherpa (AI Teaching Guide)

**Purpose**: This sherpa guide helps AI assistants teach Docker for development environments through structured interaction. The focus is on using Docker to create isolated, reproducible dev environments — not production deployment.

**Route Map**: See `/routes/docker-dev-environments/map.md` for the high-level overview of this route.
**Paired Guide**: The human-focused content is at `/routes/docker-dev-environments/guide.md`.

---

## Teaching Overview

### Learning Objectives
By the end of this session, the learner should be able to:
- Explain the difference between containers and VMs
- Run, inspect, and manage Docker containers from the command line
- Write Dockerfiles with efficient layer caching
- Use bind mounts for live code editing inside containers
- Set up multi-container dev environments with Docker Compose
- Configure a dev container with devcontainer.json
- Explain when to use mise or Nix inside a Docker container

### Prerequisites to Verify
Before starting, verify the learner has:
- Basic command line skills (cd, ls, pwd, running programs)
- Docker Desktop installed on their system (or Docker Engine on Linux)

**If prerequisites are missing**: Help them install Docker Desktop first. On macOS, download from docker.com. On Linux, install Docker Engine via their package manager.

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
- Example: "What does a Docker volume persist that a bind mount doesn't? A) Container logs B) Data independent of the host filesystem layout C) Environment variables D) Running processes"

**Explanation Questions:**
- Ask learner to explain concepts in their own words
- Assess deeper understanding and ability to apply knowledge
- Example: "Why do we copy package.json before copying the rest of the source code in a Dockerfile?"

**Mixed Approach (Recommended):**
- Use multiple choice for quick checks after introducing new commands
- Use explanation questions for core concepts like layer caching and bind mounts vs volumes
- Adapt based on learner responses and confidence level

---

## Teaching Flow

### Introduction

**What to Cover:**
- Docker creates isolated environments called containers where your project's dependencies live
- Containers are lighter than VMs — they share the host kernel instead of running a full OS
- For development, Docker means every team member gets the same environment, and project dependencies don't pollute the host
- This route focuses on Docker for dev environments, not production deployment

**Opening Questions to Assess Level:**
1. "Have you used Docker before? If so, what for?"
2. "What problems have you run into with development environment setup?"
3. "Have you heard of containers vs VMs? What's your understanding of the difference?"

**Adapt based on responses:**
- If experienced with Docker: Skip basics, focus on dev workflow patterns, dev containers, and composing with mise/Nix
- If knows VMs but not containers: Use VM comparison as the anchor point
- If complete beginner: Start with the "shipping container" analogy, take it slower
- If frustrated by env setup: Emphasize how Docker solves "works on my machine" problems

**Good opening analogy:**
"Think of a Docker container like a shipping container. It doesn't matter what ship (computer) carries it — the contents are always the same. A Docker container packages your app and all its dependencies so it runs identically everywhere. But unlike a VM, which ships a whole computer, a container just ships the differences from the host OS."

---

### Setup Verification

**Check Installation:**
Ask them to run:
```bash
docker --version
docker compose version
```

**If not installed:**
- macOS: Download Docker Desktop from https://www.docker.com/products/docker-desktop/
- Ubuntu: `sudo apt-get update && sudo apt-get install docker.io docker-compose-v2`
- Fedora: `sudo dnf install docker docker-compose`
- After install on Linux: `sudo usermod -aG docker $USER` (then log out and back in)

**Quick Orientation:**
"Docker has a client-server architecture. The `docker` CLI sends commands to the Docker daemon, which does the actual work of building images and running containers. On macOS, Docker Desktop runs a lightweight Linux VM behind the scenes because containers need a Linux kernel — but you won't interact with that VM directly."

---

### Section 1: What Docker Does

**Core Concept to Teach:**
Docker runs containers — isolated processes that share the host OS kernel but have their own filesystem, network, and process space. An image is a read-only template; a container is a running instance of an image.

**How to Explain:**
1. Start with the problem: "Have you ever spent a day setting up a dev environment? Or had code work on your machine but fail on a coworker's? Docker fixes that by packaging everything the project needs into a container."
2. Containers vs VMs:
   - A VM runs a full operating system with its own kernel — heavy, slow to start
   - A container shares the host's kernel and just isolates the filesystem and processes — lightweight, starts in seconds
   - Analogy: "A VM is like renting a whole apartment. A container is like having your own room in a shared house — you have privacy, but you share the plumbing and electricity."
3. Images vs containers:
   - An image is a blueprint — a read-only snapshot of a filesystem
   - A container is a running instance of that image
   - Analogy: "An image is like a class definition. A container is like an object — an instance of that class. You can create many containers from one image."
4. When Docker makes sense for dev:
   - Project has complex dependencies (databases, specific runtimes, system libraries)
   - Team needs identical environments
   - You want to isolate project dependencies from your host system
   - You want to onboard new developers quickly

**Discussion Points:**
- "What dependencies does your current project need? How are they managed?"
- "Have you experienced the 'works on my machine' problem?"

**Common Misconceptions:**
- Misconception: "Docker is just lightweight VMs" -> Clarify: "Containers share the host kernel. They don't virtualize hardware. This makes them faster to start and more resource-efficient, but it also means a Linux container needs a Linux kernel (which is why Docker Desktop runs a small Linux VM on macOS)."
- Misconception: "Docker is only for deployment" -> Clarify: "Docker is equally useful for development. Many teams use Docker primarily for dev environments, not deployment."
- Misconception: "Containers are always better than VMs" -> Clarify: "They serve different purposes. VMs provide stronger isolation (separate kernel) and can run different operating systems. Containers are better for packaging applications and dev environments."

**Verification Questions:**
1. "What's the key difference between a container and a VM?"
2. "Explain the relationship between a Docker image and a container."
3. Multiple choice: "A Docker container on macOS runs on: A) The macOS kernel directly B) A Linux kernel in a lightweight VM managed by Docker Desktop C) A Windows kernel D) Its own kernel"

**Good answer indicators:**
- They understand containers share the host kernel, VMs don't
- They can describe images as blueprints and containers as running instances
- They can answer B (Linux kernel in a lightweight VM on macOS)

**If they struggle:**
- Go back to the apartment vs room analogy
- Draw the architecture: Host OS -> Docker Engine -> Container 1, Container 2, etc.
- Compare: "A VM diagram would show: Host OS -> Hypervisor -> Guest OS 1 (with its own kernel) -> App 1"

---

### Section 2: Docker Fundamentals

**Core Concept to Teach:**
The Docker CLI is how you interact with containers. The essential commands are `docker run`, `docker ps`, `docker exec`, `docker stop`, and `docker rm`.

**How to Explain:**
1. "The Docker CLI follows a consistent pattern: `docker <command> [options] [arguments]`"
2. "The most important command is `docker run` — it pulls an image if needed, creates a container, and starts it"
3. "Think of `docker exec` as SSH-ing into a running container — it lets you run commands inside it"

**Walk Through Together:**

Run a container interactively:
```bash
docker run -it ubuntu:22.04 bash
```

Explain the flags:
- `-i` keeps STDIN open (interactive)
- `-t` allocates a terminal
- `ubuntu:22.04` is the image (name:tag)
- `bash` is the command to run inside the container

"You're now inside an Ubuntu container. Try `ls /`, `cat /etc/os-release`. Notice this is a minimal Ubuntu — no extra packages. Type `exit` to leave."

List running containers:
```bash
docker ps
```

"No containers running — because you exited the one you just created. Let's see all containers, including stopped ones:"

```bash
docker ps -a
```

"There it is — status 'Exited'. Containers stick around after stopping unless you remove them."

Run a detached container:
```bash
docker run -d --name mywebserver -p 8080:80 nginx:latest
```

Explain:
- `-d` runs in the background (detached)
- `--name mywebserver` gives it a human-friendly name
- `-p 8080:80` maps host port 8080 to container port 80

"Now open http://localhost:8080 in your browser — you should see the Nginx welcome page."

Execute a command in the running container:
```bash
docker exec -it mywebserver bash
```

"You're inside the running Nginx container. You can look around, check the config at `/etc/nginx/nginx.conf`, etc. Type `exit` to leave — the container keeps running."

Stop and remove:
```bash
docker stop mywebserver
docker rm mywebserver
```

"Or combine them: `docker rm -f mywebserver` force-stops and removes in one step."

List downloaded images:
```bash
docker images
```

"These are the images stored locally on your machine. They stay even after you remove containers."

**Common Misconceptions:**
- Misconception: "Exiting a container deletes it" -> Clarify: "Exiting stops the container, but it still exists. Use `docker ps -a` to see stopped containers and `docker rm` to remove them."
- Misconception: "`docker run` always creates a new container" -> Clarify: "Yes, every `docker run` creates a new container. Use `docker start` to restart a stopped container. This is a frequent source of confusion."
- Misconception: "I need to `docker pull` before `docker run`" -> Clarify: "`docker run` pulls the image automatically if it's not available locally."

**Verification Questions:**
1. "What's the difference between `docker run` and `docker exec`?"
2. "How do you see containers that have been stopped?"
3. Multiple choice: "You run `docker run -d --name app nginx` and then `docker run -d --name app nginx` again. What happens? A) It starts a second container with the same name B) It restarts the first container C) It fails because the name is already in use D) It replaces the first container"

**Good answer indicators:**
- They understand `run` creates a new container, `exec` runs in an existing one
- They know `docker ps -a` shows all containers including stopped ones
- They can answer C (name conflict error)

**If they struggle:**
- Focus on just `docker run -it ubuntu bash` and `exit` — get comfortable entering and leaving containers
- "Think of `docker run` like buying a new notebook. `docker exec` is like writing in a notebook you already have open"

**Exercise 2.1:**
"Run an interactive Python container, start a Python REPL, print 'Hello from Docker', then exit."

**How to Guide Them:**
1. "What image would you use for Python?"
2. If stuck: "Try `docker run -it python:3.12 python`"
3. Have them type `print('Hello from Docker')` in the REPL
4. Exit with `exit()` or Ctrl-D

**Exercise 2.2:**
"Run an Nginx container in the background with the name 'webtest' mapped to port 9090. Verify it's running with `docker ps`. Open it in a browser. Then stop and remove it."

**How to Guide Them:**
1. "Which flags do you need for background, name, and port?"
2. If stuck: "Remember `-d` for detached, `--name` for the name, `-p host:container` for ports"
3. Verify with `docker ps` and browser
4. Clean up: `docker stop webtest && docker rm webtest`

---

### Section 3: Writing Dockerfiles

**Core Concept to Teach:**
A Dockerfile is a recipe for building a Docker image. Each instruction creates a layer, and Docker caches layers to make rebuilds fast. The order of instructions matters for cache efficiency.

**How to Explain:**
1. "A Dockerfile is a text file with instructions for building an image — like a recipe. Each line adds a layer to the image."
2. "Docker caches each layer. If a layer hasn't changed, Docker reuses the cached version. This is why instruction order matters for rebuild speed."
3. "The key insight for dev workflows: put things that change rarely (dependency installs) before things that change often (source code). This way, changing your code doesn't re-install all your dependencies."

**Walk Through Together:**

Start with a simple Dockerfile:
```dockerfile
FROM node:20-slim

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 3000
CMD ["node", "server.js"]
```

Explain each instruction:
- `FROM node:20-slim` — start from a Node.js base image (slim variant = smaller)
- `WORKDIR /app` — set the working directory inside the container
- `COPY package*.json ./` — copy package files first (for cache efficiency)
- `RUN npm install` — install dependencies (cached until package.json changes)
- `COPY . .` — copy the rest of the source code
- `EXPOSE 3000` — document which port the app uses (informational, doesn't publish)
- `CMD ["node", "server.js"]` — default command when container starts

**The Layer Caching Lesson:**

"Watch what happens when you build twice:"

```bash
docker build -t myapp .
```

"First build: every step runs. Now change a source file and build again:"

```bash
docker build -t myapp .
```

"Notice how steps 1-4 say 'CACHED'? Docker skipped them because nothing in those layers changed. Only the `COPY . .` and later steps re-ran. That's why we copy `package.json` before the source code — changing source code doesn't trigger a full `npm install`."

"Now imagine we had written it the wrong way:"

```dockerfile
# Bad ordering - any code change triggers npm install
FROM node:20-slim
WORKDIR /app
COPY . .
RUN npm install
CMD ["node", "server.js"]
```

"Here, changing any source file invalidates the `COPY . .` layer, which forces `RUN npm install` to run again. That's slow and wasteful."

**Multi-stage Builds:**

"Multi-stage builds use multiple `FROM` statements. Each stage can copy artifacts from a previous stage. This is useful for separating build tools from the final image:"

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

"The build stage has all the build tools. The final stage only has the output. This keeps the image small."

**.dockerignore:**

"Like `.gitignore`, `.dockerignore` tells Docker which files to exclude from the build context:"

```
node_modules
.git
*.md
.env
dist
```

"This speeds up builds and prevents accidentally baking secrets or unnecessary files into your image."

**Common Misconceptions:**
- Misconception: "EXPOSE publishes a port" -> Clarify: "EXPOSE is documentation. You still need `-p` in `docker run` to actually publish ports."
- Misconception: "Each RUN creates a separate container" -> Clarify: "Each RUN creates a layer in the image. Layers are filesystem snapshots, not separate containers."
- Misconception: "I should put COPY . . at the top for simplicity" -> Clarify: "That kills your layer cache. Always copy dependency files first, install dependencies, then copy source code."

**Verification Questions:**
1. "Why do we copy package.json separately before the rest of the source code?"
2. "What does a multi-stage build accomplish?"
3. Multiple choice: "You change one line of JavaScript and run `docker build`. Which layers need to rebuild? A) All of them B) Only the COPY . . layer and everything after it C) Only the CMD layer D) None — Docker uses the cache for everything"

**Good answer indicators:**
- They understand layer caching and why instruction order matters
- They can explain multi-stage builds as separating build-time from runtime dependencies
- They can answer B (COPY . . and everything after)

**If they struggle:**
- Build a real Dockerfile and show the CACHED output
- "Think of layers like steps in a recipe. If you change step 5, you don't need to redo steps 1-4. But you do need to redo steps 5, 6, 7..."

**Exercise 3.1:**
"Write a Dockerfile for a Python Flask app. The app has a `requirements.txt` and a `app.py`. Optimize for layer caching."

**How to Guide Them:**
1. "What's the Python equivalent of `package.json` and `npm install`?"
2. If stuck: "Use `FROM python:3.12-slim`, copy `requirements.txt` first, run `pip install -r requirements.txt`, then copy the source"
3. Solution:
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

**Exercise 3.2:**
"Create a `.dockerignore` file for a Node.js project. Think about what files should NOT be in the Docker image."

**How to Guide Them:**
1. "What files exist in a typical Node project that Docker doesn't need?"
2. If stuck: "Think about version control files, installed dependencies (we install fresh in the container), editor configs, environment files..."
3. Solution:
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

---

### Section 4: Volumes and Bind Mounts

**Core Concept to Teach:**
Containers have their own filesystem. Without volumes or bind mounts, code changes on your host aren't visible in the container. Bind mounts map host directories into the container, enabling live code editing. Named volumes persist data independent of any container's lifecycle.

**How to Explain:**
1. Start with the problem: "You've built a Docker image with your code baked in. You change a file on your host... nothing happens in the container. The container has its own copy. You'd have to rebuild the image every time you change a line of code. That's terrible for development."
2. "Bind mounts solve this by mapping a host directory into the container. Changes on either side are visible to the other immediately."
3. "Named volumes are for data that should persist even if the container is deleted — like a database's data files."
4. "Docker Compose lets you define multi-container setups in a YAML file, so you don't have to type long `docker run` commands."

**Walk Through Together:**

Show the problem first:
```bash
# Build and run a container with code baked in
docker build -t myapp .
docker run -d --name myapp -p 3000:3000 myapp

# Edit a source file on the host...
# The container still has the old version — you'd need to rebuild
```

Solve it with a bind mount:
```bash
docker run -it \
  -v $(pwd):/app \
  -w /app \
  -p 3000:3000 \
  node:20 \
  npm run dev
```

Explain:
- `-v $(pwd):/app` maps the current directory on the host to `/app` in the container
- `-w /app` sets the working directory to `/app`
- "Now edit a file on your host — the change is immediately visible in the container. If your app has hot reload, it picks up the change automatically."

**The node_modules problem:**

"There's a gotcha with bind mounts in Node.js projects. If you bind mount your whole project, the host's `node_modules` (which might be empty or have macOS-specific binaries) overwrites the container's `node_modules`. Fix this with an anonymous volume:"

```bash
docker run -it \
  -v $(pwd):/app \
  -v /app/node_modules \
  -w /app \
  -p 3000:3000 \
  node:20 \
  sh -c "npm install && npm run dev"
```

"The `-v /app/node_modules` (no host path) creates an anonymous volume for `node_modules`, so the container uses its own installed modules instead of the host's."

**Named Volumes:**

"Named volumes persist data independently of containers. Useful for databases:"

```bash
docker volume create pgdata
docker run -d \
  --name devdb \
  -v pgdata:/var/lib/postgresql/data \
  -e POSTGRES_PASSWORD=devpass \
  postgres:16
```

"Even if you `docker rm devdb`, the `pgdata` volume still exists with your data. Start a new container with the same volume, and your data is there."

**tmpfs Mounts:**

"tmpfs mounts store data in memory only — it disappears when the container stops. Useful for secrets or temporary files you don't want written to disk:"

```bash
docker run -it \
  --tmpfs /tmp \
  ubuntu:22.04 bash
```

**Docker Compose:**

"Typing long `docker run` commands gets old. Docker Compose lets you define everything in a `docker-compose.yml` file:"

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

"Now start everything with one command:"

```bash
docker compose up
```

"Stop everything:"

```bash
docker compose down
```

"Stop everything AND delete volumes (fresh start):"

```bash
docker compose down -v
```

**Common Misconceptions:**
- Misconception: "Bind mounts copy files into the container" -> Clarify: "Bind mounts create a live link. Changes on either side are immediately visible to the other. No copying involved."
- Misconception: "Volumes and bind mounts are the same thing" -> Clarify: "Bind mounts map a specific host path into the container. Volumes are managed by Docker and stored in Docker's own storage area. Use bind mounts for code, volumes for data."
- Misconception: "`docker compose down` deletes my data" -> Clarify: "`docker compose down` stops and removes containers and networks, but preserves volumes. Use `docker compose down -v` to also delete volumes."

**Verification Questions:**
1. "Why can't you just edit code on your host and have it work in a container without bind mounts?"
2. "When would you use a named volume instead of a bind mount?"
3. Multiple choice: "You have a `docker-compose.yml` with a Postgres service and a named volume for its data. You run `docker compose down`. What happens to the database data? A) It's deleted B) It's preserved in the named volume C) It's moved to the host filesystem D) It's corrupted"

**Good answer indicators:**
- They understand containers have their own filesystem, separate from the host
- They know bind mounts are for code (live editing), volumes are for data (persistence)
- They can answer B (preserved in the named volume)

**If they struggle:**
- Demo it: create a file on the host, show it appears in the container via bind mount
- "Bind mount = mirror. Volume = external hard drive. tmpfs = RAM disk."

**Exercise 4.1:**
"Run a Node.js container with a bind mount so you can edit code on your host and see changes reflected in the container. Create a simple `index.js` that logs a message, run it in the container, change the message on the host, and run it again."

**How to Guide Them:**
1. "Create a simple `index.js` with `console.log('version 1')`"
2. "Run the container with a bind mount and execute the script"
3. "Change the message on your host and run the script again in the container"
4. Solution:
```bash
echo "console.log('version 1')" > index.js
docker run --rm -v $(pwd):/app -w /app node:20 node index.js
# Output: version 1

echo "console.log('version 2')" > index.js
docker run --rm -v $(pwd):/app -w /app node:20 node index.js
# Output: version 2
```

**Exercise 4.2:**
"Write a `docker-compose.yml` that runs a Node.js app with bind mounts and a Postgres database with a named volume. Start it up and verify both services are running."

**How to Guide Them:**
1. "Start with the `services` key. You need two services: `app` and `db`"
2. "For the app, use bind mounts. For the db, use a named volume"
3. "Don't forget to declare the volume at the top level"
4. Have them verify with `docker compose ps`

---

### Section 5: Dev Containers

**Core Concept to Teach:**
Dev containers standardize the development environment using a `devcontainer.json` spec. They integrate with editors like VS Code so the editor runs inside the container, giving you a consistent dev experience with the right tools, extensions, and settings.

**How to Explain:**
1. "Dev containers take Docker for dev one step further. Instead of just running your code in a container, your entire editor experience runs in the container."
2. "VS Code's Dev Containers extension opens your project inside a container. Your terminal, debugger, extensions, and settings all run inside the container. It feels like working locally, but everything is containerized."
3. "The `devcontainer.json` file describes what the dev environment should look like — what image to use, what ports to forward, what extensions to install, what setup commands to run."

**Walk Through Together:**

Create a basic `.devcontainer/devcontainer.json`:
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

Explain each field:
- `name` — display name for the dev container
- `image` — base image to use (Microsoft provides purpose-built dev container images)
- `forwardPorts` — ports to forward from the container to your host
- `postCreateCommand` — runs after the container is created (good for installing dependencies)
- `customizations.vscode.extensions` — VS Code extensions to install inside the container
- `customizations.vscode.settings` — VS Code settings to apply

**Using a Dockerfile instead of an image:**

"For more control, point to a Dockerfile instead of a pre-built image:"

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

**Dev Container Features:**

"Features are reusable units of dev container configuration. They add tools without you writing Dockerfile instructions:"

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

"Features are like plugins. Instead of writing `RUN apt-get install ...` in a Dockerfile, you declare what tools you need and features handle the installation."

**With Docker Compose:**

"Dev containers can use Docker Compose for multi-container setups:"

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

**When dev containers make sense vs plain Docker:**
- Use dev containers when: your team uses VS Code (or compatible editors), you want standardized extensions and settings, you want one-click setup for new team members
- Use plain Docker when: your team uses diverse editors, you only need the runtime environment (not editor integration), you prefer managing your own editor setup

**Common Misconceptions:**
- Misconception: "Dev containers only work with VS Code" -> Clarify: "VS Code has the best support, but the devcontainer.json spec is open. JetBrains IDEs, GitHub Codespaces, and the Dev Container CLI all support it."
- Misconception: "Dev containers replace Docker Compose" -> Clarify: "Dev containers can use Docker Compose. They add editor integration on top of your existing Docker setup."
- Misconception: "I need to use Microsoft's base images" -> Clarify: "You can use any Docker image or Dockerfile. Microsoft's images are convenient because they come with common dev tools, but they're optional."

**Verification Questions:**
1. "What does `devcontainer.json` define that a regular Dockerfile doesn't?"
2. "What are dev container features?"
3. Multiple choice: "Where should the `devcontainer.json` file be placed? A) The repository root B) `.devcontainer/` directory C) `~/.config/devcontainer/` D) Inside the Docker image"

**Good answer indicators:**
- They understand dev containers add editor integration (extensions, settings, port forwarding) on top of Docker
- They can describe features as reusable tool installers
- They can answer B (`.devcontainer/` directory)

**If they struggle:**
- "Think of a Dockerfile as describing the server. `devcontainer.json` describes the developer's desk — what tools are on it, how the editor is configured, what shortcuts are available."
- Show them the one-click experience: open a project with a devcontainer.json in VS Code, click "Reopen in Container"

**Exercise 5.1:**
"Create a `devcontainer.json` for a Python project. It should use a Python base image, install the Python extension for VS Code, forward port 8000, and run `pip install -r requirements.txt` after creation."

**How to Guide Them:**
1. "Start with the basic structure: name, image, forwardPorts, postCreateCommand"
2. "For the Python VS Code extension, the identifier is `ms-python.python`"
3. Solution:
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

**Exercise 5.2:**
"Modify the devcontainer.json to use features instead of a Python-specific image. Start from a base Ubuntu image and add Python and Node.js via features."

**How to Guide Them:**
1. "Change the image to `mcr.microsoft.com/devcontainers/base:ubuntu`"
2. "Add a `features` object with the Python and Node features"
3. Solution:
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

---

### Section 6: Composing with mise and Nix

**Core Concept to Teach:**
Docker gives you an isolated environment, but managing tool versions inside that environment is its own challenge. mise provides simple tool version management; Nix provides fully reproducible builds. Both can be used inside Docker containers.

**How to Explain:**
1. "Docker gives you isolation. But inside the container, you still need to install the right versions of Node, Python, Go, etc. You can hardcode versions in your Dockerfile, or you can use a tool version manager."
2. "mise is like nvm/pyenv/rbenv but for everything. It reads a `.mise.toml` config and installs the right versions of tools. Using it inside Docker means your container uses the same tool versions as your local machine."
3. "Nix takes a different approach. Instead of managing versions, Nix builds everything from source in a reproducible way. A Nix expression always produces the same output. It's more complex but gives stronger reproducibility guarantees."
4. "The choice depends on your needs: mise is simpler to adopt, Nix is more rigorous."

**mise Inside Docker:**

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

And the `.mise.toml`:
```toml
[tools]
node = "20"
python = "3.12"
```

"Now the container has the exact Node and Python versions specified in `.mise.toml`. If someone updates the version in the config, rebuilding the image picks it up."

**When mise makes sense inside Docker:**
- You already use mise locally and want the container to match
- You need multiple runtime versions and don't want to find a Docker image with all of them
- You want a single config file (`.mise.toml`) to define tool versions for both local and container dev

**Nix Inside Docker:**

"Nix can be used in a multi-stage build. The first stage uses Nix to build the project; the second stage copies just the output:"

```dockerfile
FROM nixos/nix:latest AS builder
COPY . /src
WORKDIR /src
RUN nix build

FROM ubuntu:22.04
COPY --from=builder /src/result/bin/myapp /usr/local/bin/
CMD ["myapp"]
```

"For dev environments, you can also use Nix to create a dev shell inside a container:"

```dockerfile
FROM nixos/nix:latest

COPY . /src
WORKDIR /src

# Create a dev shell with all dependencies
RUN nix develop --command echo "Dev shell ready"

CMD ["nix", "develop"]
```

**When Nix makes sense inside Docker:**
- You need bit-for-bit reproducibility
- Your project already uses Nix (flake.nix, shell.nix)
- You want to build minimal production images from Nix derivations
- You're comfortable with Nix's learning curve

**Trade-offs:**

| | mise | Nix |
|---|---|---|
| Learning curve | Low | High |
| Config file | `.mise.toml` | `flake.nix` / `shell.nix` |
| Reproducibility | Tool versions pinned | Everything pinned (including system libs) |
| Adoption | Drop-in, works alongside existing tools | Requires rethinking how you manage dependencies |
| Build speed | Fast (downloads pre-built binaries) | Slower first build (may build from source), fast after caching |

**When to compose tools vs use one:**
- Docker alone: sufficient for most dev environments. Use when dependencies are simple (one runtime + a database).
- Docker + mise: use when you need multiple runtimes at specific versions and want a simple config.
- Docker + Nix: use when you need maximum reproducibility or your team already uses Nix.
- Nix alone (no Docker): possible for local dev, but Docker adds isolation from the host.

**Common Misconceptions:**
- Misconception: "I need mise or Nix inside Docker" -> Clarify: "For many projects, a well-written Dockerfile is enough. These tools add value when you need multiple runtimes or want the container to match your local tooling."
- Misconception: "Nix replaces Docker" -> Clarify: "Nix handles reproducible builds and dependency management. Docker handles isolation and environment consistency. They complement each other."
- Misconception: "mise inside Docker is redundant" -> Clarify: "It's redundant if you only need one runtime. It's useful when you need several tools at specific versions and want one config file for both local and container dev."

**Verification Questions:**
1. "When would you use mise inside a Docker container instead of just specifying versions in the Dockerfile?"
2. "What does Nix provide that mise doesn't?"
3. Multiple choice: "You have a project that uses Node 20, Python 3.12, and Go 1.22. What's the simplest way to get all three in a Docker container? A) Find a Docker image that has all three B) Write RUN commands to install each one C) Use mise with a .mise.toml D) Use Nix flakes"

**Good answer indicators:**
- They understand mise is for tool version management, Nix is for full reproducibility
- They can identify scenarios where each approach fits
- They can answer C (mise is the simplest for multiple runtimes)

**If they struggle:**
- "Ignore Nix for now. Focus on mise: it's just a way to say 'I want Node 20 and Python 3.12' in a config file, and it installs both"
- Show the `.mise.toml` and Dockerfile side by side — it's a small addition

**Exercise 6.1:**
"Write a Dockerfile that uses mise to install Node 20 and Python 3.12. Include a `.mise.toml` config."

**How to Guide Them:**
1. "Start with `FROM ubuntu:22.04` and install curl and git"
2. "Install mise with the one-liner from mise.run"
3. "Copy the `.mise.toml` and run `mise install`"
4. Solution is the Dockerfile and `.mise.toml` shown above

**Exercise 6.2:**
"Write a multi-stage Dockerfile that uses Nix to build a project in the first stage and copies the result to a minimal second stage."

**How to Guide Them:**
1. "First stage starts `FROM nixos/nix:latest`"
2. "Copy your source and run `nix build`"
3. "Second stage starts from a minimal base and copies the build output"
4. Solution is the Nix multi-stage Dockerfile shown above

---

## Practice Project

**Project Introduction:**
"Let's put everything together. You'll build a development environment for a project that has both a Node.js API and a Python data processing script, using Docker Compose with bind mounts for live editing and mise for tool management."

**Requirements:**
Present one at a time:
1. "Create a `Dockerfile` that uses mise to install Node 20 and Python 3.12"
2. "Write a `docker-compose.yml` with the app service using bind mounts for live code editing"
3. "Add a Postgres database service with a named volume for data persistence"
4. "Create a simple Node.js API (`server.js`) that responds to HTTP requests"
5. "Create a Python script (`process.py`) that can be run inside the same container"
6. "Verify that editing code on the host is reflected in the container"

**Scaffolding Strategy:**
- Let them work independently first
- Check in after the Dockerfile: "Does your Dockerfile build successfully?"
- Check in after docker-compose.yml: "Can you bring up the services with `docker compose up`?"
- After bind mounts: "Try editing `server.js` on your host — does the change show up in the container?"
- Final check: "Can you run `python process.py` inside the container?"

**Starter Files:**

`.mise.toml`:
```toml
[tools]
node = "20"
python = "3.12"
```

`server.js`:
```javascript
const http = require('http');
const server = http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ message: 'Hello from Docker dev environment' }));
});
server.listen(3000, () => console.log('Server running on port 3000'));
```

`process.py`:
```python
import json
data = {"processed": True, "source": "python in docker"}
print(json.dumps(data, indent=2))
```

**Checkpoints During Project:**
- After Dockerfile builds: verify with `docker build -t devenv .`
- After Compose is up: verify with `docker compose ps`
- After bind mount test: edit `server.js`, verify change appears
- After Python test: `docker compose exec app python process.py`
- After database: `docker compose exec db psql -U postgres -c 'SELECT 1'`

**Code Review Approach:**
When reviewing their work:
1. Check the Dockerfile for proper layer ordering
2. Verify bind mounts are set up correctly in docker-compose.yml
3. Confirm the database uses a named volume
4. Test that live editing works
5. Ask them to explain their choices: "Why did you organize the Dockerfile this way?"

**If They Get Stuck:**
- "Which part are you working on? Dockerfile, Compose, or the app code?"
- "What error are you seeing?"
- If really stuck: "Let's build the Dockerfile together, then you do the Compose file"

**Extension Ideas if They Finish Early:**
- "Add a `devcontainer.json` so this project works with VS Code Dev Containers"
- "Add a Redis service to the Compose file"
- "Write a health check for the Node.js service in the Compose file"
- "Try replacing mise with Nix in the Dockerfile"

---

## Wrap-Up and Next Steps

**Review Key Takeaways:**
"Let's review what you learned today:"
- Docker containers provide isolated environments that share the host kernel
- Images are blueprints, containers are running instances
- Dockerfile instruction order matters for layer caching — put rarely-changing things first
- Bind mounts link host directories into containers for live code editing
- Docker Compose defines multi-container setups in a YAML file
- Dev containers add editor integration (extensions, settings) on top of Docker
- mise and Nix can manage tool versions inside containers when needed

**Ask them to explain one concept:**
"Can you walk me through why we copy `package.json` before the rest of the source code in a Dockerfile?"
(This reinforces layer caching, the most impactful Dockerfile optimization)

**Assess Confidence:**
"On a scale of 1-10, how confident do you feel setting up a Docker dev environment for a project?"

**Respond based on answer:**
- 1-4: "That's okay! Docker has a lot of concepts. Focus on `docker run` with bind mounts for now — that alone solves most dev environment problems. Build up from there."
- 5-7: "Good progress! You've got the core concepts. Try dockerizing a real project of yours this week. You'll solidify everything by applying it."
- 8-10: "Excellent! You're ready to set up Docker dev environments for real projects. Consider exploring dev containers or Docker + mise for your next project."

**Suggest Next Steps:**
Based on their progress and interests:
- "To practice: Dockerize one of your existing projects with a Dockerfile and docker-compose.yml"
- "For team workflows: Set up a dev container so new team members can start coding in one click"
- "For reproducibility: Explore mise-basics or nix-dev-environments routes"
- "For AI workflows: Check out agent-sandboxing for running AI agents in containers"

**Encourage Questions:**
"Do you have any questions about anything we covered?"
"What part do you want to practice more?"
"Is there a specific project you want to dockerize?"

---

## Adaptive Teaching Strategies

### If Learner is Struggling

**Signs:**
- Confused about images vs containers
- Can't remember Docker commands
- Lost in the Dockerfile syntax
- Frustrated by Docker Desktop setup issues

**Strategies:**
- Slow down significantly
- Focus on `docker run -it ubuntu bash` — get comfortable being "inside" a container
- Skip Dockerfiles temporarily and just use existing images with bind mounts
- Use the class/object analogy repeatedly for images vs containers
- Make sure Docker Desktop is running before troubleshooting anything else
- Create a cheat sheet of the 5 most important commands
- "Don't try to memorize commands. Keep a reference open. Muscle memory comes with use."

### If Learner is Excelling

**Signs:**
- Completes exercises quickly
- Asks about multi-stage builds, networking, or orchestration
- Already knows some Docker

**Strategies:**
- Move at faster pace, skip basics
- Focus on dev container features and advanced Compose patterns
- Discuss Docker networking between containers
- Explore health checks, depends_on, and service dependencies in Compose
- Introduce Docker BuildKit features (cache mounts, secrets)
- Challenge: "Dockerize your most complex project with a full dev environment"

### If Learner Seems Disengaged

**Signs:**
- Short responses
- Not asking questions
- Taking long breaks

**Strategies:**
- Check in: "How are you feeling about this? Is the pace okay?"
- Connect to their real work: "What does your current dev setup look like? Where's the friction?"
- Focus on practical benefits: "Let me show you how this eliminates environment setup entirely"
- Make it more hands-on: less explanation, more doing
- Show off a real dev container opening in VS Code — the "one-click" experience is compelling

### Different Learning Styles

**Visual learners:**
- Draw the Docker architecture: host -> daemon -> containers
- Show the layer caching output during builds
- Use Docker Desktop's GUI to visualize running containers

**Hands-on learners:**
- Less explanation upfront, get them running containers immediately
- "Try this command and see what happens"
- Learn by breaking things and fixing them

**Conceptual learners:**
- Explain the Linux kernel features behind containers (namespaces, cgroups)
- Discuss the Docker daemon's architecture
- Compare container runtimes (Docker, Podman, containerd)

---

## Troubleshooting Common Issues

### Docker Desktop Not Running
- Most Docker commands will fail with "Cannot connect to the Docker daemon"
- On macOS: Open Docker Desktop from Applications
- Check status: `docker info`

### Permission Denied
- On Linux: add user to docker group: `sudo usermod -aG docker $USER` then log out and back in
- Don't run Docker commands with `sudo` as a habit — fix the group membership instead

### Port Already in Use
- Error: "Bind for 0.0.0.0:3000 failed: port is already allocated"
- Find what's using the port: `lsof -i :3000`
- Either stop the other process or use a different host port: `-p 3001:3000`

### Build Context Too Large
- `docker build` is sending gigabytes of data
- Create or fix `.dockerignore` — likely missing `node_modules`, `.git`, or large data files
- "The build context is everything Docker sends to the daemon. `.dockerignore` controls what's excluded."

### Container Immediately Exits
- Check logs: `docker logs <container_name>`
- Common cause: the CMD command fails or finishes immediately
- For debugging: `docker run -it <image> bash` to get a shell and investigate

### Bind Mount Permissions (Linux)
- Files created inside the container may be owned by root on the host
- Solutions: match the container user to host user, or use `--user $(id -u):$(id -g)`
- Not usually an issue on macOS (Docker Desktop handles this)

### docker compose vs docker-compose
- `docker compose` (with a space) is the modern version, built into Docker CLI
- `docker-compose` (with a hyphen) is the older standalone binary
- Use `docker compose` — it's the current standard

---

## Teaching Notes

**Key Emphasis Points:**
- The "aha moment" is seeing bind mounts in action — edit on host, see changes in container
- Layer caching is the most practical Dockerfile concept — it saves minutes per build
- Docker Compose is the dev workflow game-changer — one command to start everything
- Don't get bogged down in Docker internals — focus on the dev workflow

**Pacing Guidance:**
- Don't rush Sections 1-2 — understanding images vs containers is the foundation
- Section 3 (Dockerfiles) is the meatiest — give plenty of time for the caching concept
- Section 4 (Volumes) is where dev workflows click — this is the practical payoff
- Section 5 (Dev Containers) can be lighter if the learner doesn't use VS Code
- Section 6 (Composing) is optional depth — only go deep if they're interested

**Success Indicators:**
You'll know they've got it when they:
- Can explain images vs containers without hesitation
- Instinctively put dependency installation before source code copying in Dockerfiles
- Use bind mounts for dev and understand why
- Reach for Docker Compose instead of long `docker run` commands
- Start thinking about which of their projects could benefit from Docker
- Ask questions like "could I use this for..." (shows they're applying the concepts)

**Most Common Confusion Points:**
1. **Images vs containers**: Blueprint vs instance — keep using the class/object analogy
2. **Layer caching**: Why order matters — show the CACHED output in a real build
3. **Bind mounts vs volumes**: Code vs data — "mount your code, volume your database"
4. **Port mapping**: `-p host:container` — "left side is your machine, right side is the container"

**Teaching Philosophy:**
- Docker's value for dev becomes obvious when you see bind mounts working — get there quickly
- Dockerfiles are intimidating at first but follow simple patterns — show the pattern
- Docker Compose turns 10-line commands into readable YAML — show the transformation
- Dev containers are the "luxury" layer — valuable but not essential for everyone
- mise/Nix inside Docker is advanced composition — only for those who need it
