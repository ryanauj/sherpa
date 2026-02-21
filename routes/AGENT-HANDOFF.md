# iOS Routes — Agent Handoff Instructions

This document tells follow-up agents what work remains and how to complete it consistently.

## Current State

### Completed (full sherpa.md + guide.md content):
1. **xcode-essentials** — Full sherpa and guide
2. **swift-for-developers** — Full sherpa and guide
3. **swiftui-fundamentals** — Full sherpa and guide

### Maps only (need sherpa.md + guide.md written):
4. **uikit-essentials**
5. **ios-data-persistence**
6. **cloudkit-integration**
7. **ios-app-patterns**
8. **app-store-publishing**
9. **ios-ci-cd-with-github-actions**

### Ascent:
- **ascents/my-first-ios-app/ascent.md** — Complete

## How to Write a Route's sherpa.md

Read these files first to understand the pattern:
- `techniques/sherpa-template-v1.md` — The template
- `routes/xcode-essentials/sherpa.md` — A completed example
- `routes/tmux-basics/sherpa.md` — Another completed example

### Required structure:

```yaml
---
title: [Same as map.md title]
route_map: /routes/[route-name]/map.md
paired_guide: /routes/[route-name]/guide.md
topics:
  - [Same topics as map.md]
---
```

1. **Teaching Overview** — Learning objectives (match the map), prerequisites to verify, learner preferences configuration section
2. **Assessment Strategies** — Multiple choice and explanation question formats
3. **Teaching Flow** — Introduction with opening questions to assess level
4. **Sections** — One per section in the map. Each needs:
   - Core concept to teach
   - How to explain (with analogies to backend/web dev concepts the learner knows)
   - Real code examples (not placeholders)
   - Common misconceptions with corrections
   - Verification questions (multiple choice + explanation)
   - "If they struggle" guidance with progressive hints
   - Exercise with guided approach
5. **Practice Project** — Scaffolding strategy, checkpoints, code review approach
6. **Wrap-Up** — Review takeaways, assess confidence, suggest next steps
7. **Adaptive Teaching Strategies** — Strategies for struggling, excelling, disengaged learners
8. **Troubleshooting Common Issues** — Technical setup problems, concept-specific confusion

### Style:
- Imperative voice ("Ask...", "Explain...", "Present this example...")
- Compare to tools/patterns the learner already knows
- No time estimates or difficulty ratings
- No ABOUTME comments (those are for code files only)

## How to Write a Route's guide.md

Read these files first:
- `techniques/guide-template-v1.md` — The template
- `routes/xcode-essentials/guide.md` — A completed example
- `routes/tmux-basics/guide.md` — Another completed example

### Required structure:

```yaml
---
title: [Same as map.md title]
route_map: /routes/[route-name]/map.md
paired_sherpa: /routes/[route-name]/sherpa.md
prerequisites:
  - [List from map.md]
topics:
  - [Same topics as map.md]
---
```

1. **Overview** — What the route covers, why it matters
2. **Learning Objectives** — Match the map
3. **Prerequisites** — What you need, with links to prerequisite routes
4. **Setup** — What to install/configure, with verification steps
5. **Sections** — One per section in the map. Each needs:
   - Clear explanation of the concept
   - Real, complete code examples with expected output
   - Step-by-step walkthrough of the code
   - Key points to remember
   - Exercise(s) with progressive hints in `<details>` blocks and solutions
   - Self-check checkpoint
6. **Practice Project** — Requirements, getting started steps, hints, example solution
7. **Summary** — Key takeaways, skills gained, self-assessment
8. **Next Steps** — Next routes, related routes, external resources
9. **Appendix** — Quick reference/cheat sheet, glossary, troubleshooting

### Style:
- Second person, conversational ("you")
- Complete, runnable code examples
- Show expected output
- Compare to tools/concepts from backend/web dev (React, Node, Python, etc.)
- Exercises use collapsible `<details>` blocks for hints and solutions
- No time estimates or difficulty ratings
- No ABOUTME comments

## Route-Specific Guidance

### uikit-essentials
- Keep this concise — it's an optional side route
- Focus on: UIViewRepresentable, UIViewControllerRepresentable, Coordinator pattern
- The practice project should wrap a UIKit component in SwiftUI
- Key comparison: imperative (UIKit) vs declarative (SwiftUI), like DOM manipulation vs React

### ios-data-persistence
- SwiftData is the primary focus (it's the modern approach)
- Core Data gets an overview section only (when you'll encounter it)
- UserDefaults and Keychain are quick practical sections
- Compare to ORMs the learner knows (ActiveRecord, SQLAlchemy, Prisma)
- Practice project: note-taking or task-tracking app with SwiftData

### cloudkit-integration
- CKSyncEngine is the recommended sync approach (iOS 17+)
- Cover the CloudKit Dashboard thoroughly — it's how you manage schema
- Emphasize: CloudKit is NOT a backend. No server-side logic, no joins, no aggregation.
- Error handling section is critical — partial failures, rate limiting, conflicts
- Compare to Firebase/Supabase where helpful
- Practice project: add CloudKit sync to the data persistence practice project

### ios-app-patterns
- MVVM + @Observable is the recommended architecture
- Don't over-prescribe — note that simple screens don't need ViewModels
- Repository pattern for data access abstraction
- Dependency injection via @Environment (SwiftUI-native DI)
- Mention TCA (The Composable Architecture) as a popular alternative but don't teach it
- Practice project: refactor a monolithic app into clean MVVM

### app-store-publishing
- Code signing section is critical — this is where everyone gets stuck
- Include common App Review rejection reasons with how to avoid them
- TestFlight internal vs external testing differences
- Privacy nutrition labels are required
- No hotfixes — every release goes through review
- Practice project: take an app through the full pipeline to TestFlight

### ios-ci-cd-with-github-actions
- Fastlane Match is the recommended code signing approach for CI
- App Store Connect API keys (`.p8`) over Apple ID + password
- macOS runners are 10x the cost of Linux — mention self-hosted as alternative
- Cache SPM packages aggressively
- Practice project: complete CI/CD pipeline with tests on PR + deploy on merge

## Research Reference

Detailed research on all 9 topics was gathered and is summarized below for accuracy:

### Key facts to get right:
- **@Observable** (Observation framework, Swift 5.9+) replaces ObservableObject/@Published. Use for all new code.
- **SwiftData** (iOS 17+) replaces Core Data for new projects. Uses @Model macro.
- **Swift 6** enforces strict concurrency (Sendable, @MainActor, actors are required).
- **CKSyncEngine** (iOS 17+) is Apple's first-party sync engine for CloudKit.
- **Swift Testing** (@Test, @Suite macros) is the modern test framework, replacing XCTest.
- **NavigationStack** replaced NavigationView (deprecated iOS 16).
- **#Preview macro** replaced PreviewProvider protocol.
- **SPM** is the standard dependency manager. CocoaPods is in maintenance mode.
- **Xcode 16** is current (Swift 6, iOS 18 SDK).
- **`any` vs `some`**: `some` for opaque types (SwiftUI), `any` for existentials.
- **Typed throws** (`throws(MyError)`) added in Swift 6.

## Quality Checklist

Before committing a route, verify:
- [ ] sherpa.md and guide.md are aligned with map.md (same sections, same objectives)
- [ ] YAML frontmatter cross-references are correct
- [ ] Code examples are real and accurate (not template placeholders)
- [ ] No time estimates or difficulty ratings anywhere
- [ ] Exercises have hints in `<details>` blocks and solutions
- [ ] Practice project is described with requirements and guidance
- [ ] Comparisons to backend/web dev concepts are included where helpful
- [ ] No ABOUTME comments (those are for code files, not content files)
