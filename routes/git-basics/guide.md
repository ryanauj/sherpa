---
title: Git Basics - Version Control Fundamentals
difficulty: beginner
duration: 45 minutes
route_map: /routes/git-basics/map.md
paired_sherpa: /routes/git-basics/sherpa.md
prerequisites: 
  - Basic command line usage
  - Text editor familiarity
topics:
  - Version Control
  - Git
  - Repositories
  - Commits
---

# Git Basics - Guide (Human-Focused Content)

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/git-basics/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/git-basics/map.md` for the high-level overview.

## Overview

Git is a distributed version control system that helps you track changes in your code over time. This tutorial will teach you the fundamental concepts and commands needed to use Git for managing your projects. You'll learn how to create repositories, track changes, and understand the basic Git workflow.

## Learning Objectives

By the end of this tutorial, you will be able to:
- Explain what version control is and why it's important
- Create and initialize a Git repository
- Track and commit changes to files
- View the history of your project
- Understand the basic Git workflow

## Prerequisites

Before starting this tutorial, you should be familiar with:
- Using the command line/terminal
- Creating and editing text files
- Basic file system navigation (cd, ls, pwd)

## Setup

Install Git on your system:

**macOS:**
```bash
brew install git
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install git
```

**Windows:**
Download from [git-scm.com](https://git-scm.com/download/win)

**Verify installation:**
```bash
git --version
```

**Expected Output:**
```
git version 2.x.x
```

**Configure Git (replace with your info):**
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## Section 1: Understanding Version Control

### Introduction

Version control is a system that records changes to files over time. Think of it like a detailed "undo" history for your entire project, with the ability to see who made changes, when, and why.

### Why Version Control?

Without version control, you might have seen folders like:
- `project_final.zip`
- `project_final_v2.zip`
- `project_final_ACTUALLY_FINAL.zip`
- `project_final_USE_THIS_ONE.zip`

Version control solves this problem by:
- Tracking all changes systematically
- Allowing multiple people to work together
- Enabling you to revert to previous versions
- Providing a clear history of what changed and why

### Key Concepts

- **Repository (repo)**: A folder that Git tracks
- **Commit**: A snapshot of your project at a point in time
- **Working Directory**: Your current files
- **Staging Area**: Files prepared for the next commit

### Exercise 1.1: Identify Version Control Benefits

**Task:** Think about a project you've worked on. List three problems that version control could have solved.

<details>
<summary>Example Answers</summary>

Possible answers:
1. Accidentally deleting important code
2. Not knowing who made a specific change
3. Wanting to try an experiment without breaking working code
4. Collaborating with others on the same files
</details>

### Checkpoint 1

Before moving on, make sure you understand:
- [ ] What version control is
- [ ] Why version control is useful
- [ ] Basic Git terminology (repository, commit)

---

## Section 2: Creating Your First Repository

### Creating a New Repository

Let's create a new Git repository:

```bash
# Create a new directory for your project
mkdir my-first-repo
cd my-first-repo

# Initialize Git repository
git init
```

**Expected Output:**
```
Initialized empty Git repository in /path/to/my-first-repo/.git/
```

### What Just Happened?

Git created a hidden `.git` folder in your directory. This folder contains all the version control information. Your directory is now a Git repository!

```bash
# View the .git folder (optional)
ls -la
```

### Checking Repository Status

The most important Git command you'll use:

```bash
git status
```

**Expected Output:**
```
On branch main

No commits yet

nothing to commit (create/copy files and use "git add" to track)
```

### Key Points
- `git init` creates a new repository
- `.git` folder stores all version control data
- `git status` shows the current state of your repository

### Exercise 2.1: Create a Practice Repository

**Task:** Create a new directory called `practice-repo` and initialize it as a Git repository.

<details>
<summary>Solution</summary>

```bash
mkdir practice-repo
cd practice-repo
git init
git status
```

**Explanation:** These commands create a new directory, navigate into it, initialize Git, and check the status.
</details>

### Checkpoint 2

Before moving on, make sure you can:
- [ ] Create a new Git repository
- [ ] Check the status of a repository
- [ ] Understand what the `.git` folder does

---

## Section 3: Tracking Changes with Commits

### The Git Workflow

The basic Git workflow has three stages:
1. **Modify** files in your working directory
2. **Stage** changes you want to commit
3. **Commit** staged changes to the repository

### Creating and Tracking a File

```bash
# Create a new file
echo "# My Project" > README.md

# Check status
git status
```

**Expected Output:**
```
On branch main

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	README.md

nothing added to commit but untracked files present (use "git add" to track)
```

Git sees the file but isn't tracking it yet.

### Staging Changes

```bash
# Stage the file
git add README.md

# Check status
git status
```

**Expected Output:**
```
On branch main

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
	new file:   README.md
```

The file is now staged and ready to commit.

### Making a Commit

```bash
# Commit the staged changes
git commit -m "Add README file"
```

**Expected Output:**
```
[main (root-commit) abc1234] Add README file
 1 file changed, 1 insertion(+)
 create mode 100644 README.md
```

### Understanding Commit Messages

Good commit messages:
- Start with a verb ("Add", "Fix", "Update")
- Are concise but descriptive
- Explain *what* changed and *why*

Examples:
- ✅ "Add user authentication"
- ✅ "Fix typo in welcome message"
- ❌ "Update" (too vague)
- ❌ "Made some changes" (not descriptive)

### Exercise 3.1: Make Your First Commit

**Task:** 
1. Create a file called `hello.txt` with the text "Hello, Git!"
2. Stage the file
3. Commit it with the message "Add hello file"

<details>
<summary>Solution</summary>

```bash
echo "Hello, Git!" > hello.txt
git add hello.txt
git commit -m "Add hello file"
```

**Explanation:** We create a file, stage it with `git add`, and commit it with a descriptive message.
</details>

### Viewing History

```bash
git log
```

**Expected Output:**
```
commit abc1234567890... (HEAD -> main)
Author: Your Name <your.email@example.com>
Date:   Mon Jan 1 12:00:00 2024 -0500

    Add README file
```

For a more compact view:
```bash
git log --oneline
```

### Common Pitfalls

1. **Forgetting to stage files**: You must use `git add` before `git commit`
2. **Empty commit messages**: Always include a meaningful message with `-m`
3. **Committing too much**: Make small, focused commits rather than huge ones

### Checkpoint 3

Before moving on, make sure you can:
- [ ] Create and modify files
- [ ] Stage changes with `git add`
- [ ] Commit changes with `git commit -m`
- [ ] View commit history with `git log`

---

## Section 4: Making and Tracking Multiple Changes

### Modifying Existing Files

```bash
# Add more content to README
echo "This is a practice repository for learning Git." >> README.md

# Check what changed
git status
```

**Expected Output:**
```
On branch main
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
	modified:   README.md
```

### Viewing Differences

```bash
git diff README.md
```

This shows what changed in the file (lines added or removed).

### Staging and Committing Multiple Files

```bash
# Create another file
echo "print('Hello from Python')" > hello.py

# Stage all changes
git add .

# Commit
git commit -m "Update README and add Python script"
```

**Note:** `git add .` stages all changed and new files in the current directory.

### Exercise 4.1: Practice the Full Workflow

**Task:**
1. Modify `hello.txt` to add a second line: "Welcome to version control!"
2. Create a new file `notes.txt` with any content
3. Stage both changes
4. Commit with an appropriate message
5. View your commit history

<details>
<summary>Solution</summary>

```bash
echo "Welcome to version control!" >> hello.txt
echo "These are my Git notes." > notes.txt
git add .
git commit -m "Update hello.txt and add notes"
git log --oneline
```
</details>

---

## Practice Project

### Project Description

Create a simple personal journal using Git to track your entries over time.

### Requirements
- Create a `journal` repository
- Add at least 3 journal entries (as separate files or appended to one file)
- Make separate commits for each entry
- Each commit should have a descriptive message with the date

### Getting Started

```bash
mkdir journal
cd journal
git init
```

Now create your journal entries and commit them!

### Validation

Your commit history should show at least 3 commits:
```bash
git log --oneline
```

Each commit should have a meaningful message indicating when the entry was made.

---

## Summary

Congratulations! You've learned the fundamentals of Git:
- Version control tracks changes to your code over time
- `git init` creates a new repository
- `git status` shows what's changed
- `git add` stages files for commit
- `git commit -m "message"` saves a snapshot
- `git log` shows your history

## Next Steps

Now that you understand Git basics, explore:
- **Branching and Merging**: Work on features without affecting the main code
- **Remote Repositories**: Share your code on GitHub/GitLab
- **Collaboration**: Work with others on the same repository
- **Advanced Git**: Stashing, rebasing, and more

## Additional Resources

- [Git Documentation](https://git-scm.com/doc): Official Git docs
- [GitHub Git Handbook](https://guides.github.com/introduction/git-handbook/): Quick reference
- [Visualizing Git](https://git-school.github.io/visualizing-git/): Interactive visualization
