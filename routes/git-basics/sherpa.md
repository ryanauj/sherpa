---
title: Git Basics - Version Control Fundamentals
difficulty: beginner
duration: 45 minutes
route_map: /routes/git-basics/map.md
paired_guide: /routes/git-basics/guide.md
topics:
  - Version Control
  - Git
  - Repositories
  - Commits
---

# Git Basics - Sherpa (AI Teaching Guide)

**Purpose**: This sherpa guide helps AI assistants teach Git fundamentals effectively through structured interaction.

**Route Map**: See `/routes/git-basics/map.md` for the high-level overview of this route.
**Paired Guide**: The human-focused content is at `/routes/git-basics/guide.md`.

---

## Teaching Overview

### Learning Objectives
By the end of this session, the learner should be able to:
- Explain what version control is and why it's important
- Create and initialize a Git repository
- Track and commit changes to files
- View the history of their project
- Understand the basic Git workflow

### Prerequisites to Verify
Before starting, verify the learner has:
- Basic command line skills (cd, ls, pwd)
- A text editor they're comfortable with
- Git installed on their system

**If prerequisites are missing**: Help them install Git first. If command line skills are weak, provide a 5-minute primer on basic navigation.

### Estimated Time
- 10 minutes for introduction and setup
- 25 minutes for core Git concepts
- 10 minutes for practice project
- Total: ~45 minutes

---

## Teaching Flow

### Introduction (5 min)

**What to Cover:**
- Version control tracks changes to files over time
- Git is the most popular version control system
- They'll learn to save "snapshots" of their work
- Useful for solo projects and collaboration

**Opening Questions to Assess Level:**
1. "Have you ever used version control before, like Git, SVN, or anything similar?"
2. "What made you interested in learning Git?"
3. "Do you have a project in mind that you want to track with Git?"

**Adapt based on responses:**
- If experienced with other VCS: Draw comparisons, move faster through basics
- If complete beginner: Use more analogies, take it slower
- If has specific project: Use their project as example throughout

**Good opening analogy:**
"Think of Git like a save system in a video game. Instead of having just one save file that gets overwritten, Git lets you have unlimited save points. You can go back to any save point, see what changed between saves, and even try different paths without losing your progress."

---

### Setup Verification (5 min)

**Check Installation:**
Ask them to run: `git --version`

**If not installed:**
- macOS: `brew install git` or direct them to git-scm.com
- Linux: `sudo apt-get install git` or equivalent
- Windows: Direct to git-scm.com download

**Configure Git:**
Guide them through:
```bash
git config --global user.name "Their Name"
git config --global user.email "their.email@example.com"
```

Explain: "This tells Git who is making changes. It's like signing your name on your work."

---

### Section 1: Understanding Version Control (5 min)

**Core Concept to Teach:**
Version control systematically records changes to files over time.

**How to Explain:**
1. Start with the problem: "Have you ever had files named 'project_final', 'project_final2', 'project_ACTUALLY_FINAL'?" (Make it relatable and humorous)
2. Explain: "Version control solves this by keeping one file but remembering all its past versions"
3. Key benefits: Can undo mistakes, see history, collaborate without conflicts

**Discussion Points:**
- "What problems have you had managing file versions?"
- "Why might it be useful to see who changed what and when?"

**Common Misconceptions:**
- Misconception: "Version control is only for code" → Clarify: "Works for any text files - documentation, config files, even writing"
- Misconception: "It's too complex for personal projects" → Clarify: "Actually simplifies personal projects once you learn the basics"

**Verification Questions:**
1. "Can you explain in your own words what version control does?"
2. "Why would someone want to use version control instead of just saving files?"

**Good answer indicators:**
- They understand it tracks changes over time
- They can name at least one benefit (undo, history, collaboration)

**If they struggle:**
- Revisit the "video game save point" analogy
- Ask if they've ever lost work or couldn't remember what they changed
- Give a concrete example: "Imagine you write a paper. Git lets you save versions as you work, so you can always go back if you decide you liked yesterday's version better"

---

### Section 2: Creating Your First Repository (10 min)

**Core Concept to Teach:**
A repository (repo) is a folder that Git tracks. Initialize it with `git init`.

**How to Explain:**
1. "A repository is just a folder where Git keeps track of changes"
2. "The `git init` command tells Git to start tracking this folder"
3. "Git stores all its information in a hidden `.git` folder"

**Walk Through Together:**
```bash
mkdir my-first-repo
cd my-first-repo
git init
```

**After running `git init`:**
- Point out the success message
- Explain the `.git` folder (they can ignore it, Git manages it)
- Introduce `git status` as the most important command

**Demonstrate:**
```bash
git status
```

**Explain the output:**
- "On branch main" - don't worry about branches yet
- "No commits yet" - haven't saved any snapshots
- "nothing to commit" - no files to save yet

**Common Misconceptions:**
- Misconception: "I need to do something with the .git folder" → Clarify: "Never touch .git directly, Git manages it"
- Misconception: "`git init` creates a repo on GitHub" → Clarify: "This creates a local repo on your computer only"

**Verification Questions:**
1. "What does `git init` do?"
2. "What command shows you the current status of your repository?"

**If they struggle:**
- Have them run the commands again in a new folder
- Use analogy: "git init is like saying 'start keeping track of this notebook'"

**Exercise 2.1:**
"Create your own repository called 'practice-repo' and initialize it with Git. Then check its status."

**How to Guide Them:**
1. First ask: "What commands do you think you need?"
2. If stuck: "You'll need to make a folder, go into it, and initialize Git"
3. Let them try, then verify together

**Review Together:**
- Check they ran `mkdir practice-repo`
- Check they ran `cd practice-repo`
- Check they ran `git init`
- Check they ran `git status`

---

### Section 3: Tracking Changes with Commits (15 min)

**Core Concept to Teach:**
The Git workflow: Modify files → Stage changes → Commit (save snapshot)

**How to Explain:**
Use the "photography" analogy:
1. "Modify files is like arranging people for a photo"
2. "Staging (`git add`) is like deciding who to include in the photo"
3. "Committing (`git commit`) is like taking the photo - you've saved that moment"

**Walk Through Together:**
```bash
echo "# My Project" > README.md
git status
```

**Point out:**
- README.md appears as "Untracked"
- Git sees the file but isn't tracking it yet
- Red color typically means "not staged"

**Stage the file:**
```bash
git add README.md
git status
```

**Point out:**
- Now it says "Changes to be committed"
- Green color typically means "staged and ready"
- The file is "ready for the photo"

**Make the commit:**
```bash
git commit -m "Add README file"
```

**Explain the parts:**
- `-m` means "message"
- Message describes what this snapshot contains
- Good messages start with a verb: "Add", "Fix", "Update"

**Show the result:**
```bash
git status
```
- "nothing to commit, working tree clean"
- This means everything is saved, no pending changes

**Show history:**
```bash
git log
```
or
```bash
git log --oneline
```

**Common Misconceptions:**
- Misconception: "I can skip `git add` and just commit" → Clarify: "Must stage first, this lets you commit only some changes"
- Misconception: "Commit messages don't matter" → Clarify: "They're crucial for understanding history later"

**Verification Questions:**
1. "What are the three steps in the Git workflow?"
2. "What does `git add` do?"
3. "Why do we need a commit message?"

**Good answer indicators:**
- Can name: modify, stage, commit
- Understands staging selects what to save
- Knows messages describe changes

**If they struggle:**
- Draw out the workflow (or describe it step by step)
- Use the photography analogy again
- Do another example together

**Exercise 3.1:**
"Create a file called hello.txt with any text, stage it, and commit it with the message 'Add hello file'"

**How to Guide Them:**
1. Ask: "What's the first step? Creating the file or using Git?"
2. After they create the file: "How do you check if Git sees it?"
3. Lead them through: create → status → add → status → commit → status

**Progressive Hints if Stuck:**
- Hint 1: "Start by creating the file with echo or a text editor"
- Hint 2: "Remember to use `git add` before you can commit"
- Hint 3: "The command is `git commit -m 'Add hello file'`"

**Review Their Work:**
- Verify with `git log` that they have a commit
- Ask them to explain what they did
- Praise: "Great! You've now saved your first snapshot"

---

### Section 4: Making Multiple Changes (10 min)

**Core Concept to Teach:**
You can modify multiple files and stage/commit them together or separately.

**How to Explain:**
"Now that you know the basic workflow, let's practice it with multiple files and changes."

**Demonstrate:**
```bash
echo "Some documentation" >> README.md
echo "print('Hello')" > hello.py
git status
```

**Point out:**
- README.md shows as "modified" (not "untracked")
- hello.py shows as "untracked"
- Can stage all at once or individually

**Show `git diff`:**
```bash
git diff README.md
```

Explain: "This shows what changed in the file since the last commit. Lines with + are additions."

**Stage all changes:**
```bash
git add .
```

Explain: "The dot means 'add everything in this directory'"

**Commit:**
```bash
git commit -m "Update README and add Python script"
```

**View history:**
```bash
git log --oneline
```

**Common Misconceptions:**
- Misconception: "`git add .` adds files from my whole computer" → Clarify: "Only adds files in the current directory and subdirectories"
- Misconception: "I must commit after every tiny change" → Clarify: "Commit when you've completed a logical unit of work"

**Verification Questions:**
1. "How do you see what changed in a file before staging?"
2. "How do you stage multiple files at once?"
3. "When should you make a commit?"

**If they struggle:**
- Review the workflow diagram again
- Do another example together
- Emphasize `git status` as their friend - use it often

**Exercise 4.1:**
"Modify hello.txt by adding a new line, create a new file notes.txt, then stage and commit both changes with an appropriate message."

**How to Guide Them:**
1. Don't give away the answer immediately
2. If stuck: "Remember the workflow - modify, stage, commit"
3. Check their commit message: if it's vague like "update", suggest being more descriptive

**After Exercise:**
Ask: "How many commits do you have now?" 
Have them check with `git log --oneline`

---

## Practice Project (10 min)

**Project Introduction:**
"Now let's put it all together. Create a simple journal repository where you'll make several entries and track them with Git."

**Requirements:**
Present one at a time:
1. "Create a new repository called 'journal'"
2. "Make at least 3 journal entries (as separate files or entries in one file)"
3. "Make a separate commit for each entry with a descriptive message"

**Scaffolding Strategy:**
- Let them work independently first
- Check in after they create the repo: "Got the repo set up?"
- Check in after first entry: "Great! Remember to commit before making the next entry"
- Be available for questions

**Checkpoints During Project:**
- After repo creation: Verify they ran `git init`
- After first commit: Check their commit message quality
- Midway: "How's it going? Any questions?"
- At completion: Review their `git log`

**Code Review Approach:**
When reviewing their work:
1. Run `git log --oneline` to see their commits
2. Praise: "Nice commit messages!" or "Good job keeping commits focused"
3. If messages are vague: "What if we made these more descriptive? 'Add journal entry about learning Git' tells more than just 'update'"
4. Count commits: "Great! You have 3+ commits, exactly what we needed"

**If They Get Stuck:**
- "Where are you stuck? Repo creation or making entries?"
- "What have you tried so far?"
- If really stuck: "Let's do the first entry together, then you can do the others"

**Extension Ideas if They Finish Early:**
- "Try using `git log` with different options: `--oneline`, `--graph`, `--all`"
- "What happens if you use `git diff` before staging? After?"
- "Create a folder with multiple files and commit them together"

---

## Wrap-Up and Next Steps (5 min)

**Review Key Takeaways:**
"Let's review what you learned today:"
- Version control tracks changes systematically
- `git init` creates a repository
- `git status` shows what's changed
- `git add` stages files for commit
- `git commit -m "message"` saves a snapshot
- `git log` shows history

**Ask them to explain one concept:**
"Can you walk me through the basic Git workflow in your own words?"
(This reinforces learning and shows you what stuck)

**Assess Confidence:**
"On a scale of 1-10, how confident do you feel with basic Git?"

**Respond based on answer:**
- 1-4: "That's okay! Git takes practice. Would you like to review anything? Try using it for a real project this week"
- 5-7: "That's normal for just learning! The more you use it, the more natural it becomes. Start using it for your projects"
- 8-10: "Excellent! You're ready to use Git. Next step is learning about branches and remote repositories"

**Suggest Next Steps:**
Based on their progress and interests:
- "To practice: Use Git for your next project, even a small one"
- "When ready: Learn about branches (Git's killer feature)"
- "For collaboration: Learn about GitHub and remote repositories"
- "For going deeper: Learn about git diff, git restore, and the staging area"

**Encourage Questions:**
"Do you have any questions about anything we covered?"
"What part do you want to practice more?"
"Is there anything about Git you're curious about?"

---

## Adaptive Teaching Strategies

### If Learner is Struggling

**Signs:**
- Taking long time on exercises
- Confused about commands
- Can't answer verification questions

**Strategies:**
- Slow down significantly
- Do more examples together before having them try alone
- Use more analogies (save points, photography, taking notes)
- Have them repeat commands several times
- Check if command line itself is the issue (may need CLI basics first)
- Do exercises together rather than independently

### If Learner is Excelling

**Signs:**
- Completes exercises quickly
- Asks "what if" questions
- Wants to know more

**Strategies:**
- Move at faster pace, less explanation
- Introduce extra commands: `git diff`, `git restore`, `git rm`
- Discuss branches briefly as a preview
- Give more complex exercises (multiple files, selective staging)
- Ask deeper questions: "Why do you think Git uses a staging area?"
- Suggest they explore `git log` options

### If Learner Seems Disengaged

**Signs:**
- Short responses
- Not asking questions
- Taking long breaks

**Strategies:**
- Check in: "How are you feeling about this? Is the pace okay?"
- Connect to their goals: "What do you want to build? How will Git help?"
- Take a break if it's been intense
- Make it more interactive: less explaining, more doing
- Share a relatable story about why Git is useful

### Different Learning Styles

**Visual learners:**
- Describe the workflow as a diagram: "Imagine three boxes: Working Directory → Staging Area → Repository"
- Use ASCII art if possible
- Talk about colors in git status output

**Hands-on learners:**
- Less explanation upfront, get them doing quickly
- "Try this command and see what happens"
- Learn through experimentation

**Conceptual learners:**
- Explain why Git is designed this way
- Discuss the benefits of the staging area
- Talk about Git's distributed nature

---

## Troubleshooting Common Issues

### Git Not Installed
- Direct to installation for their OS
- On macOS, running git first time may trigger install
- Verify installation with `git --version`

### Permission Issues
- Usually on Windows
- Try running terminal as administrator
- Check folder permissions

### Wrong Directory
- Very common: they run `git init` in wrong place
- Show them `pwd` to see where they are
- Guide them to the right folder with `cd`

### Forgot to Configure User
- Git will complain on first commit
- Help them set user.name and user.email
- Explain it's one-time setup

### Unclear Git Status Output
- Walk through each line of the output
- Use colors as clues (if their terminal shows them)
- Run git status frequently so they get used to it

### Commit Message Mistakes
- Forgot the `-m`: Git opens editor, which confuses beginners
  - Help them exit editor (`:q` in vim, Ctrl+X in nano)
  - Show them how to set default editor if needed
  - Re-run commit with `-m`
- Typo in message: Show `git commit --amend` if just committed

---

## Additional Resources to Suggest

**If they want more practice:**
- "Try using Git for any project you're working on"
- "Create a repository for notes or documentation"
- "Practice with the journal project idea"

**If they want deeper understanding:**
- "Read about the Git staging area and why it exists"
- "Learn about Git branches - that's where Git really shines"
- "Check out the Pro Git book (free online)"

**If they want to see real applications:**
- "Explore repositories on GitHub to see how others use Git"
- "Learn about remote repositories and GitHub/GitLab"
- "Look into Git workflows used by teams"

---

## Teaching Notes

**Key Emphasis Points:**
- Really emphasize the workflow: modify → stage → commit
- Make sure they understand staging before moving on
- Stress that `git status` is their best friend
- Commit messages matter more than they think initially

**Pacing Guidance:**
- Don't rush Section 3 (commits) - this is the foundation
- Can move faster through Section 4 if they got Section 3
- Allow plenty of time for practice project
- Better to go slow and solid than fast and confused

**Success Indicators:**
You'll know they've got it when they:
- Use `git status` without prompting
- Can explain the workflow in their own words
- Complete exercises with minimal hints
- Start writing good commit messages
- Ask questions like "what if I want to..." (shows they're thinking ahead)

**Most Common Confusion Points:**
1. **Why staging exists**: Explain it lets you commit only part of your changes
2. **Git vs GitHub**: Git is local version control, GitHub is for hosting/sharing
3. **When to commit**: After any logical unit of work is complete
4. **Commit message quality**: They'll write "update" - guide them to be more specific

**Teaching Philosophy:**
- It's okay to not cover everything - focus on core workflow
- They'll learn more by using Git than by hearing about it
- Mistakes are learning opportunities (emphasize Git makes it hard to lose work)
- Getting them comfortable with basic commands is more important than comprehensive knowledge
