---
title: How to Write Effective Commit Messages
type: best-practice
topics:
  - Git
  - Version Control
  - Documentation
related:
  - /tutorials/tools/git-basics.md
---

# How to Write Effective Commit Messages

## Purpose

This guide helps you write clear, informative commit messages that make your project history easier to understand and navigate.

---

## Quick Reference

Good commit message format:

```
Add user authentication feature

- Implement login form validation
- Add password encryption
- Create session management
```

**Pattern:**
1. One-line summary (50 characters or less)
2. Blank line
3. Detailed description (if needed)

---

## Detailed Explanation

### When to Use This

Use these guidelines every time you make a commit. Good commit messages help:
- Future you understand what changed and why
- Team members review your work
- Anyone investigating bugs or features in the history
- AI assistants provide better context-aware help

### The Seven Rules of Great Commit Messages

#### 1. Separate Subject from Body with a Blank Line

```
Fix bug in user registration

The email validation was allowing invalid formats.
Added regex pattern to check for proper email structure.
```

#### 2. Limit the Subject Line to 50 Characters

Keep it concise but descriptive. Think of it as a headline.

**Good:** "Add user profile page"
**Too long:** "Add a new user profile page with all the user information and settings"

#### 3. Capitalize the Subject Line

**Good:** "Add password reset feature"
**Bad:** "add password reset feature"

#### 4. Do Not End the Subject Line with a Period

**Good:** "Update documentation for API endpoints"
**Bad:** "Update documentation for API endpoints."

#### 5. Use the Imperative Mood in the Subject Line

Write as if giving a command:

**Good:** "Add", "Fix", "Update", "Remove", "Refactor"
**Bad:** "Added", "Fixing", "Updates"

Think: "If applied, this commit will... [your subject line]"

#### 6. Wrap the Body at 72 Characters

Makes messages readable in various tools and terminals.

#### 7. Use the Body to Explain What and Why (Not How)

The code shows how; the message explains what and why:

```
Refactor authentication to use JWT tokens

The old session-based auth was causing issues with
mobile clients. JWT tokens provide better support
for stateless authentication and work well across
different platforms.
```

### Examples

**Bad Commit Messages:**
```
fix bug
update code
WIP
asdf
more changes
```

**Good Commit Messages:**
```
Fix null pointer exception in user search

Add input validation to prevent XSS attacks

Update README with installation instructions

Remove deprecated API endpoints
```

**Excellent Commit Messages (with body):**
```
Optimize database queries for user dashboard

The dashboard was loading slowly for users with many
transactions. Added indexing on transaction_date and
user_id columns, reducing load time from 3s to 200ms.

Closes #142
```

---

## Common Issues

### Issue 1: Not Sure What to Write

**Symptoms:** You type `git commit -m "update"`

**Cause:** The change is too large or unfocused

**Solution:**
- Break your changes into smaller, focused commits
- Each commit should represent one logical change
- Ask yourself: "What does this commit do?"

### Issue 2: Commit Message is Too Long

**Symptoms:** Subject line is multiple sentences

**Cause:** Trying to explain too much in the subject

**Solution:**
Use the commit message body:
```bash
git commit
# This opens your editor for a multi-line message
```

### Issue 3: Inconsistent Style Across Team

**Symptoms:** Everyone writes commits differently

**Solution:**
- Create a CONTRIBUTING.md with commit guidelines
- Use commit message templates
- Set up commit message linters (like commitlint)

---

## Tips and Best Practices

- **Reference issues:** Include issue numbers like "Fixes #123" or "Closes #456"
- **Use consistent prefixes:** Some teams use "feat:", "fix:", "docs:" (conventional commits)
- **Commit often:** Many small commits are better than one large one
- **Test before committing:** Ensure code works before committing
- **Review before pushing:** Use `git log` to review your commits
- **Amend if needed:** Fix your last commit with `git commit --amend` (before pushing)

---

## Related Resources

- [Git Basics Tutorial](/tutorials/tools/git-basics.md): Learn fundamental Git commands
- [Conventional Commits](https://www.conventionalcommits.org/): Standardized commit format
- [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/): Detailed article

---

## For AI Assistants

### Key Context
- Good commit messages are about communication, not just recording changes
- The audience includes future developers (including the original author)
- Clarity and consistency are more important than following rules rigidly

### Common User Needs
- Users often struggle with finding the right level of detail
- They may not understand why good messages matter (explain benefits)
- Help them break down large changes into logical commits
- Suggest templates or examples based on their specific changes
