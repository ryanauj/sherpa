---
title: tmux Basics - Terminal Multiplexing Fundamentals
route_map: /routes/tmux-basics/map.md
paired_guide: /routes/tmux-basics/guide.md
topics:
  - Terminal Multiplexing
  - tmux
  - Sessions
  - Windows
  - Panes
---

# tmux Basics - Sherpa (AI Teaching Guide)

**Purpose**: This sherpa guide helps AI assistants teach tmux fundamentals effectively through structured interaction.

**Route Map**: See `/routes/tmux-basics/map.md` for the high-level overview of this route.
**Paired Guide**: The human-focused content is at `/routes/tmux-basics/guide.md`.

---

## Teaching Overview

### Learning Objectives
By the end of this session, the learner should be able to:
- Explain what a terminal multiplexer does and why it's useful
- Create, detach from, and reattach to tmux sessions
- Organize work with windows and panes
- Navigate efficiently with tmux key bindings
- Write a basic `.tmux.conf` configuration
- Use copy mode to scroll back through terminal output

### Prerequisites to Verify
Before starting, verify the learner has:
- Basic command line skills (cd, ls, pwd, running programs)
- A terminal emulator they're comfortable with
- tmux installed on their system

**If prerequisites are missing**: Help them install tmux first. If command line skills are weak, provide a quick primer on basic navigation.

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
- Example: "Which key combination is the default tmux prefix? A) Ctrl-a B) Ctrl-b C) Ctrl-t D) Ctrl-x"

**Explanation Questions:**
- Ask learner to explain concepts in their own words
- Assess deeper understanding and ability to apply knowledge
- Example: "What's the difference between detaching from a session and killing it?"

**Mixed Approach (Recommended):**
- Use multiple choice for quick checks after introducing new key bindings
- Use explanation questions for core concepts like the session lifecycle
- Adapt based on learner responses and confidence level

---

## Teaching Flow

### Introduction

**What to Cover:**
- tmux is a terminal multiplexer — it lets you run multiple terminal sessions inside one window
- Sessions persist even if you disconnect or close the terminal
- You can organize your workspace with windows (tabs) and panes (splits)
- Essential for remote work, long-running processes, and workspace organization

**Opening Questions to Assess Level:**
1. "Have you used tmux or a similar tool (screen, byobu) before?"
2. "What made you interested in learning tmux?"
3. "Do you work with remote servers or long-running terminal processes?"

**Adapt based on responses:**
- If experienced with screen: Draw comparisons, highlight tmux advantages, move faster
- If complete beginner: Use more analogies, take it slower
- If works with remote servers: Emphasize session persistence as the key motivator

**Good opening analogy:**
"Think of tmux like a virtual desktop manager for your terminal. Right now, closing your terminal kills everything running in it. tmux lets you create persistent workspaces that keep running whether you're watching them or not — like locking your computer versus shutting it down."

---

### Setup Verification

**Check Installation:**
Ask them to run: `tmux -V`

**If not installed:**
- macOS: `brew install tmux`
- Ubuntu/Debian: `sudo apt-get install tmux`
- Fedora: `sudo dnf install tmux`

**Quick Orientation:**
Before diving in, explain the prefix key concept:
"tmux uses a special key combination called a 'prefix' to know when you're talking to it versus typing in the terminal. The default prefix is `Ctrl-b`. You press `Ctrl-b`, release both keys, then press the next key. It's sequential, not simultaneous — think of it as saying 'Hey tmux, listen up' before giving a command."

**Common Misconception — Address Immediately:**
Many beginners try to hold all keys at once. Emphasize: press `Ctrl-b`, release, then press the command key. Two separate actions.

---

### Section 1: Understanding Terminal Multiplexing

**Core Concept to Teach:**
tmux manages terminal sessions that persist independently of your terminal window. It uses a hierarchy: sessions contain windows, windows contain panes.

**How to Explain:**
1. Start with the problem: "Ever had a terminal running a long process, then accidentally closed the window? Or needed to SSH into a server and run multiple things?"
2. Explain the hierarchy with analogies:
   - Sessions = virtual desktops (separate workspaces for different projects)
   - Windows = browser tabs (different views within a workspace)
   - Panes = split screen (seeing multiple things at once within a tab)
3. The prefix key: "tmux needs a way to tell your keystrokes apart from commands meant for tmux itself. That's what the prefix key does."

**Discussion Points:**
- "Can you think of situations where you'd want a terminal session to survive closing the window?"
- "When might you want to see two terminal outputs side by side?"

**Common Misconceptions:**
- Misconception: "tmux is only useful on remote servers" → Clarify: "It's valuable for local development too — persistent sessions, workspace organization, and split views"
- Misconception: "The prefix key is held simultaneously with the command key" → Clarify: "It's sequential: Ctrl-b, release, then the command key"
- Misconception: "Closing the terminal kills tmux" → Clarify: "tmux sessions keep running in the background. That's the whole point"

**Verification Questions:**
1. "Can you describe the tmux hierarchy — sessions, windows, and panes?"
2. "How does the prefix key work?"

**Good answer indicators:**
- They understand sessions persist beyond the terminal window
- They can describe the hierarchy using their own words
- They know the prefix key is sequential

**If they struggle:**
- Revisit the virtual desktop / browser tab / split screen analogy
- Have them think about their own workflow and where each level fits

---

### Section 2: Sessions

**Core Concept to Teach:**
Sessions are the top-level container in tmux. You can create, name, detach from, and reattach to sessions. The key insight: detaching leaves everything running.

**How to Explain:**
1. "A session is a complete tmux workspace. Think of it as a virtual desktop"
2. "When you detach, it's like locking your computer — everything keeps running, you've just walked away"
3. "When you reattach, you pick up exactly where you left off"

**Walk Through Together:**

Start tmux:
```bash
tmux
```

Point out: "You're now inside a tmux session. Notice the green status bar at the bottom — that's how you know you're in tmux."

Detach from the session:
"Press `Ctrl-b`, release, then `d`"

Point out: "You're back in your regular terminal. But that tmux session is still running. Let's prove it."

List sessions:
```bash
tmux ls
```

Reattach:
```bash
tmux attach
```

**The "Aha" Moment:**
Now do the real demo. Have them:
1. Start a named session: `tmux new -s demo`
2. Run something visible: `top` or `htop` or even `ping localhost`
3. Detach: `Ctrl-b d`
4. Reattach: `tmux attach -t demo`
5. See that the process is still running

"This is the core value of tmux. That process never stopped."

**Common Misconceptions:**
- Misconception: "Detaching kills the session" → Clarify: "Detaching just disconnects your view. Everything keeps running"
- Misconception: "`exit` and detach are the same" → Clarify: "`exit` kills the shell (and the session if it's the last window). `Ctrl-b d` detaches safely"
- Misconception: "I can only have one session" → Clarify: "You can have as many as you want, each with its own name"

**Verification Questions:**
1. "What's the difference between detaching from a session and exiting it?"
2. "How do you reattach to a specific named session?"
3. Multiple choice: "After you detach from a tmux session running a compile job, what happens to the compile? A) It pauses B) It stops C) It keeps running D) It restarts when you reattach"

**Good answer indicators:**
- They understand detach preserves the session
- They know `tmux attach -t name` for named sessions
- They can answer C (keeps running)

**If they struggle:**
- Do the demo again — seeing the process survive is the best teacher
- Analogy: "When you lock your phone, your music keeps playing. Detaching is like that"

**Exercise 2.1:**
"Create two named sessions: one called 'work' and one called 'play'. In 'work', run `top`. Detach, then attach to 'play'. Then list all sessions to see both."

**How to Guide Them:**
1. First ask: "What command creates a named session?"
2. If stuck: "Remember, `tmux new -s name` creates a named session"
3. Let them try, then verify with `tmux ls`

**Exercise 2.2:**
"Start a named session, run `ping localhost` in it, detach, close your terminal entirely, open a new terminal, and reattach. Is the ping still running?"

**How to Guide Them:**
1. This exercises the core concept — persistence across terminal closures
2. If nervous about closing the terminal: "Don't worry, that's the whole point. tmux will keep it safe"

**Cleaning Up:**
Show them how to kill sessions when done:
```bash
tmux kill-session -t sessionname
```
Or from inside tmux: type `exit` in each window.

---

### Section 3: Windows and Panes

**Core Concept to Teach:**
Windows are like tabs, panes are like split screen. Together they let you organize multiple terminal views within a single session.

**How to Explain:**
1. "Windows are tabs. You can have many in a session, but you see one at a time"
2. "Panes split a window so you can see multiple terminals at once"
3. "You'll use both all the time — windows for separate tasks, panes for related tasks you want side by side"

**Walk Through Together — Windows:**

Create a new session:
```bash
tmux new -s workspace
```

Create a new window:
"Press `Ctrl-b c`"

Point out the status bar: "See the window list at the bottom? You now have two windows."

Navigate between windows:
- `Ctrl-b n` — next window
- `Ctrl-b p` — previous window
- `Ctrl-b 0`, `Ctrl-b 1` — jump by number

Rename the current window:
"`Ctrl-b ,` then type a name"

"Naming windows helps you remember what's where — like naming browser tabs."

**Walk Through Together — Panes:**

Split horizontally (top/bottom):
"`Ctrl-b \"` (that's Ctrl-b then double-quote)"

Split vertically (left/right):
"`Ctrl-b %`"

Navigate between panes:
"`Ctrl-b` then arrow key (up/down/left/right)"

Zoom a pane to full screen:
"`Ctrl-b z` (press again to un-zoom)"

"Zooming is great when you need to focus on one pane temporarily but don't want to lose your layout."

Close a pane:
"Type `exit` or press `Ctrl-d` in the pane"

**Common Misconceptions:**
- Misconception: "Windows and panes are the same thing" → Clarify: "Windows are tabs (one visible at a time), panes are splits (all visible at once within a window)"
- Misconception: "I need to memorize all the key bindings" → Clarify: "Start with just a few: `c` for new window, `\"` and `%` for splits, arrow keys for navigation. You'll learn more as needed"

**Verification Questions:**
1. "What's the difference between a window and a pane?"
2. "How do you create a new window? How do you split a pane?"
3. Multiple choice: "You have 3 panes visible and want to temporarily make one full-screen. What do you press? A) Ctrl-b f B) Ctrl-b z C) Ctrl-b m D) Ctrl-b x"

**Good answer indicators:**
- They understand windows = tabs, panes = splits
- They know the key bindings for creating and navigating
- They can answer B (Ctrl-b z)

**If they struggle:**
- Have them practice creating and destroying panes repeatedly — muscle memory matters
- Draw the layout: "Imagine your terminal as a rectangle. `\"` cuts it horizontally, `%` cuts it vertically"

**Exercise 3.1:**
"Create a session with 3 named windows: 'editor', 'server', and 'logs'. Navigate between them."

**How to Guide Them:**
1. "Start a new session, then create and rename windows"
2. If stuck on renaming: "Remember `Ctrl-b ,` renames the current window"
3. Verify they can jump between windows by number

**Exercise 3.2:**
"In one window, create a 3-pane layout: one large pane on the left, two smaller panes stacked on the right. Navigate between all three panes."

**How to Guide Them:**
1. "Start with a full window. Split it vertically first, then split the right pane horizontally"
2. If the order confuses them: "Don't worry about getting it perfect. Just practice splitting and navigating"
3. Show them zoom (`Ctrl-b z`) as a way to focus on one pane

---

### Section 4: Customizing tmux

**Core Concept to Teach:**
tmux is configured via `~/.tmux.conf`. Common customizations include changing the prefix key, enabling mouse support, and adjusting window numbering.

**How to Explain:**
1. "tmux's defaults are usable but not everyone's preference. The config file lets you make it yours"
2. "The most popular changes: swap prefix to Ctrl-a (easier to reach), turn on mouse support, start window numbering at 1"

**Walk Through Together:**

Create the config file:
```bash
touch ~/.tmux.conf
```

Add common settings (guide them through adding each line and explain it):

Change prefix to Ctrl-a:
```
# Remap prefix from Ctrl-b to Ctrl-a
unbind C-b
set -g prefix C-a
bind C-a send-prefix
```

Enable mouse support:
```
# Enable mouse support (scrolling, pane selection, resizing)
set -g mouse on
```

Start window numbering at 1:
```
# Start window and pane numbering at 1 (0 is far from the other keys)
set -g base-index 1
setw -g pane-base-index 1
```

Reload config without restarting tmux:
"`Ctrl-b :` then type `source-file ~/.tmux.conf`"

Or add a reload binding to the config:
```
# Reload config with prefix-r
bind r source-file ~/.tmux.conf \; display "Config reloaded"
```

**Common Misconceptions:**
- Misconception: "I need to restart tmux to apply config changes" → Clarify: "You can reload the config from inside tmux"
- Misconception: "The old prefix stops working immediately after changing it" → Clarify: "It stops working after you reload the config. Then use the new prefix"

**Verification Questions:**
1. "Where does the tmux configuration file live?"
2. "How do you apply config changes without restarting tmux?"
3. "Why might someone change the prefix from Ctrl-b to Ctrl-a?"

**Good answer indicators:**
- They know `~/.tmux.conf`
- They can reload config from the tmux command prompt
- They understand Ctrl-a is easier to reach on most keyboards

**If they struggle:**
- Walk through adding one setting at a time, reloading, and verifying the change
- "Config files can feel abstract. Let's see each change take effect one at a time"

**Exercise 4.1:**
"Create a `.tmux.conf` with at least the prefix change and mouse support. Reload it inside tmux and verify both work."

**How to Guide Them:**
1. "Create the file, add the lines we discussed"
2. "Now reload: `Ctrl-b :` then `source-file ~/.tmux.conf`"
3. "Test the new prefix — try `Ctrl-a c` to create a window"
4. "Test mouse support — try clicking on different panes or scrolling"

---

### Section 5: Copy Mode

**Core Concept to Teach:**
Copy mode lets you scroll back through terminal output, search for text, and copy content. It turns your pane into a scrollable, searchable buffer.

**How to Explain:**
1. "Ever wanted to scroll up in your terminal to see output that scrolled past? In tmux, that's copy mode"
2. "Enter it with `prefix [`, scroll around, then press `q` to exit"
3. "By default you navigate with arrow keys and Page Up/Down. If you add `mode-keys vi`, you can use h/j/k/l instead"

**Walk Through Together:**

Enter copy mode:
"Press `prefix [` — the status bar will change to indicate you're in copy mode"

Navigate:
- Arrow keys or Page Up/Down to scroll (default)
- With `mode-keys vi`: h/j/k/l to move, Ctrl-u/Ctrl-d for half-page scrolling

Search:
- Press `/` to search forward (in vi mode)
- Press `?` to search backward (in vi mode)

Exit:
"Press `q` to leave copy mode"

**Vi vs Default Mode:**
"Adding `setw -g mode-keys vi` to your config gives you vim-style navigation in copy mode. If you're comfortable with vim keybindings, this is more efficient. If not, the default arrow-key navigation works fine."

**Common Misconceptions:**
- Misconception: "I can scroll with my mouse wheel without copy mode" → Clarify: "With mouse mode enabled, scrolling does enter copy mode automatically. Without it, you need `prefix [`"
- Misconception: "Copy mode is complicated" → Clarify: "For basic scrolling, it's just `prefix [`, scroll, `q`. That's it"

**Verification Questions:**
1. "How do you enter and exit copy mode?"
2. "What does `mode-keys vi` change?"

**Good answer indicators:**
- They know `prefix [` to enter and `q` to exit
- They understand vi mode changes navigation keys in copy mode

**If they struggle:**
- Focus on just entering, scrolling with arrow keys, and exiting
- "You don't need to learn everything about copy mode now — just scrolling back is useful enough"

**Exercise 5.1:**
"Run a command that produces lots of output (like `ls -la /usr/bin`), then use copy mode to scroll back and find a specific file."

**How to Guide Them:**
1. "Run the command, then press `prefix [` to enter copy mode"
2. "Use Page Up or arrow keys to scroll back through the output"
3. "Press `q` to exit when you've found it"

---

## Practice Project

**Project Introduction:**
"Let's put everything together. Build a development workspace that you'd actually use day-to-day."

**Requirements:**
Present one at a time:
1. "Create a named session called 'dev'"
2. "Set up 3 windows: 'editor', 'server', 'monitor'"
3. "In the 'monitor' window, create a split layout with at least 2 panes"
4. "Run a visible process in at least one pane (top, htop, watch ls, etc.)"
5. "Detach, then reattach to verify everything is still running"

**Scaffolding Strategy:**
- Let them work independently first
- Check in after session creation: "Got the session set up?"
- Check in after windows: "How are the windows looking? Named them yet?"
- After panes: "Nice! Can you navigate between panes?"
- Final check: "Detach and reattach. Everything still there?"

**Checkpoints During Project:**
- After session creation: Verify named session with `tmux ls`
- After window setup: Check the status bar shows 3 named windows
- After pane splits: Verify layout in the monitor window
- After detach/reattach: Confirm running processes survived

**Code Review Approach:**
When reviewing their work:
1. Have them show `tmux ls` output
2. Navigate through each window together
3. Check the pane layout in the monitor window
4. Praise good organization: "This is a setup you could use every day"
5. Suggest improvements if relevant: "You could add this to a shell script to recreate it automatically"

**If They Get Stuck:**
- "Which step are you on? Session, windows, or panes?"
- "What have you tried so far?"
- If really stuck: "Let's build the first window together, then you do the rest"

**Extension Ideas if They Finish Early:**
- "Write a shell script that creates this workspace automatically with `tmux new-session`, `send-keys`, etc."
- "Explore copy mode: `Ctrl-b [` to scroll and search through terminal output"
- "Try `Ctrl-b s` to get a session/window tree view"
- "Add more customizations to your `.tmux.conf`"

---

## Wrap-Up and Next Steps

**Review Key Takeaways:**
"Let's review what you learned today:"
- tmux is a terminal multiplexer that manages persistent sessions
- Sessions contain windows, windows contain panes
- The prefix key (`Ctrl-b` by default) signals tmux commands
- Detaching leaves sessions running in the background
- `.tmux.conf` lets you customize tmux to your liking

**Ask them to explain one concept:**
"Can you walk me through what happens when you detach from a session and reattach later?"
(This reinforces the core value proposition of tmux)

**Assess Confidence:**
"On a scale of 1-10, how confident do you feel using tmux for your daily work?"

**Respond based on answer:**
- 1-4: "That's okay! The key bindings take muscle memory. Keep tmux open and use it daily — it'll become second nature within a week"
- 5-7: "Good progress! You've got the concepts. Now it's about building habits. Try using tmux for every terminal session this week"
- 8-10: "Excellent! You're ready to make tmux part of your daily workflow. Next up: copy mode, scripted sessions, and plugins"

**Suggest Next Steps:**
Based on their progress and interests:
- "To practice: Use tmux for all your terminal work this week"
- "For productivity: Write a script that sets up your ideal workspace"
- "For power features: Learn copy mode for searching and copying terminal output"
- "For customization: Explore tmux plugin managers like tpm"

**Encourage Questions:**
"Do you have any questions about anything we covered?"
"What part do you want to practice more?"
"Is there a specific workflow you want to set up with tmux?"

---

## Adaptive Teaching Strategies

### If Learner is Struggling

**Signs:**
- Confused about the prefix key sequence
- Mixing up sessions, windows, and panes
- Can't remember key bindings

**Strategies:**
- Slow down significantly
- Focus on sessions only until they're comfortable, then add windows, then panes
- Use the hierarchy analogy repeatedly: virtual desktops → tabs → split screen
- Have them practice the prefix key sequence many times
- Enable mouse support early so they have a fallback
- Create a cheat sheet of the 5 most important bindings
- Check if they're trying to hold Ctrl-b and the next key simultaneously

### If Learner is Excelling

**Signs:**
- Completes exercises quickly
- Asks about advanced features
- Already customizing on their own

**Strategies:**
- Move at faster pace, less explanation
- Introduce copy mode (`Ctrl-b [`)
- Show session/window tree (`Ctrl-b s` and `Ctrl-b w`)
- Discuss scripted session creation
- Introduce `tmux send-keys` for automation
- Suggest exploring tpm (tmux plugin manager)
- Challenge: "Set up a tmux layout that matches your actual development workflow"

### If Learner Seems Disengaged

**Signs:**
- Short responses
- Not asking questions
- Taking long breaks

**Strategies:**
- Check in: "How are you feeling about this? Is the pace okay?"
- Connect to their real work: "What would your ideal terminal setup look like?"
- Focus on practical benefits: "Let me show you how this saves you from losing work"
- Make it more hands-on: less explanation, more doing
- Show off something impressive: a complex pane layout or session switching

### Different Learning Styles

**Visual learners:**
- Describe the session/window/pane hierarchy as nested boxes
- Focus on the status bar as visual feedback
- Point out the pane borders and window list

**Hands-on learners:**
- Less explanation upfront, get them creating sessions immediately
- "Try this and see what happens"
- Learn through experimentation and discovery

**Conceptual learners:**
- Explain the client-server architecture of tmux
- Discuss why sessions persist (the tmux server keeps running)
- Compare tmux to other multiplexers and explain design decisions

---

## Troubleshooting Common Issues

### tmux Not Installed
- macOS: `brew install tmux`
- Ubuntu/Debian: `sudo apt-get install tmux`
- Fedora: `sudo dnf install tmux`
- Verify with `tmux -V`

### "sessions should be nested" Error
- They're trying to start tmux inside tmux
- Explain: "You're already in a tmux session. Use windows and panes instead, or detach first"
- If they intentionally want nesting: set `TMUX` env variable (advanced, usually not needed)

### Prefix Key Not Working
- Make sure they're releasing Ctrl-b before pressing the next key
- Check if they've changed the prefix in config but haven't reloaded
- Verify they're actually inside tmux (check for status bar)

### Pane Navigation Confusion
- Common when they have many panes
- Show `Ctrl-b q` to display pane numbers briefly
- Suggest enabling mouse mode for easier navigation
- Remind them of zoom (`Ctrl-b z`) to focus on one pane

### Config Not Taking Effect
- Did they save the file? Check `cat ~/.tmux.conf`
- Did they reload? `Ctrl-b :` then `source-file ~/.tmux.conf`
- Is there a syntax error? tmux will show an error message on reload
- New config only applies to new sessions/windows unless explicitly reloaded

### Terminal Looks Weird or Garbled
- Try `Ctrl-b :` then `clear-history`
- Resize the terminal window
- Worst case: detach and reattach

---

## Teaching Notes

**Key Emphasis Points:**
- The prefix key concept is the biggest hurdle — spend time on it
- Session persistence is the "aha moment" — make sure they experience it
- Start simple (sessions only), layer on complexity (windows, then panes)
- Muscle memory matters — have them repeat key bindings multiple times

**Pacing Guidance:**
- Don't rush Section 2 (sessions) — this is the foundation and the biggest payoff
- Windows and panes can be covered faster once sessions click
- Give plenty of time for the practice project
- Better to master sessions and basic window/pane usage than to rush through everything

**Success Indicators:**
You'll know they've got it when they:
- Use the prefix key smoothly without thinking
- Can detach and reattach without hesitation
- Start creating named sessions and windows on their own
- Ask questions like "can I script this?" (shows they're thinking ahead)
- Use tmux without prompting during the rest of the session

**Most Common Confusion Points:**
1. **Prefix key timing**: Sequential, not simultaneous
2. **Detach vs exit**: Detach preserves, exit kills
3. **Windows vs panes**: Tabs vs splits
4. **Config reload**: Must explicitly reload after changing `.tmux.conf`

**Teaching Philosophy:**
- tmux's value becomes obvious through experience, not explanation
- Get them into tmux early and doing things, don't over-explain upfront
- The detach/reattach demo is the most important moment in the lesson
- Customization makes it feel personal — don't skip `.tmux.conf`
- They don't need to memorize every key binding — just the 8-10 they'll use daily
