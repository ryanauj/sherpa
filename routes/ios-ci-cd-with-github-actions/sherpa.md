---
title: iOS CI/CD with GitHub Actions
route_map: /routes/ios-ci-cd-with-github-actions/map.md
paired_guide: /routes/ios-ci-cd-with-github-actions/guide.md
topics:
  - GitHub Actions
  - Fastlane
  - Code Signing in CI
  - TestFlight Automation
  - Continuous Deployment
---

# iOS CI/CD with GitHub Actions - Sherpa (AI Teaching Guide)

**Purpose**: This sherpa guide helps AI assistants teach iOS CI/CD automation to developers who understand App Store publishing manually and want to automate the build-test-deploy pipeline with GitHub Actions and Fastlane. It covers workflows, code signing in CI, and automated TestFlight deployment — with comparisons to web CI/CD where helpful.

**Route Map**: See `/routes/ios-ci-cd-with-github-actions/map.md` for the high-level overview of this route.
**Paired Guide**: The human-focused content is at `/routes/ios-ci-cd-with-github-actions/guide.md`.

---

## Teaching Overview

### Learning Objectives
By the end of this session, the learner should be able to:
- Explain why CI/CD for iOS is more complex than for web apps (macOS requirement, code signing, provisioning)
- Write a GitHub Actions workflow that builds and tests an iOS project on every push or PR
- Set up Fastlane for an iOS project with lanes for testing, building, and uploading
- Solve the code signing problem in CI using Fastlane Match or manual certificate installation
- Automate TestFlight uploads on merge to main
- Manage version and build numbers in CI
- Cache dependencies (SPM, Ruby gems) to speed up builds
- Choose between GitHub-hosted and self-hosted macOS runners

### Prior Sessions
Before starting, check `.sessions/index.md` and `.sessions/ios-ci-cd-with-github-actions/` for prior session history. If the learner has completed previous sessions on this route, review the summaries to understand what they've covered and pick up where they left off.

### Prerequisites to Verify
Before starting, verify the learner has:
- App Store publishing basics — certificates, provisioning profiles, App Store Connect, TestFlight (see app-store-publishing route)
- Familiarity with Git and GitHub (branches, PRs, merging)
- A working iOS app with tests
- Basic YAML syntax understanding (helpful but teachable inline)

**If prerequisites are missing**: If App Store publishing is weak, suggest the app-store-publishing route first — understanding manual publishing is essential before automating it. If YAML is unfamiliar, explain it briefly when writing the first workflow.

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

If no configuration file exists, use defaults (objective tone, mixed assessments, balanced pacing).

### Assessment Strategies

Use a combination of assessment types:

**Multiple Choice Questions:**
- Good for checking understanding of workflow concepts and trigger types
- Example: "Which GitHub Actions trigger runs a workflow when a PR is opened? A) push B) pull_request C) workflow_dispatch D) schedule"

**Configuration Questions:**
- Show a workflow YAML snippet and ask the learner to identify issues or extend it
- Example: "This workflow runs on every push to every branch. How would you limit it to only main and PRs?"

**Debugging Questions:**
- Present a failing CI build log and ask the learner to identify the root cause
- These assess real-world troubleshooting ability

**Mixed Approach (Recommended):**
- Use multiple choice for workflow concepts
- Use configuration questions for YAML fluency
- Use debugging questions for signing and build failures

---

## Web-to-iOS CI/CD Reference

Use this table for learners with web CI/CD experience:

| Web CI/CD Concept | iOS CI/CD Equivalent |
|-------------------|----------------------|
| GitHub Actions workflow | Same — GitHub Actions workflow |
| `ubuntu-latest` runner | `macos-latest` runner (macOS required) |
| `npm install && npm test` | `xcodebuild test` or `fastlane scan` |
| `npm run build` | `xcodebuild archive` or `fastlane gym` |
| `npm publish` / deploy to CDN | Upload to TestFlight / App Store Connect |
| `.env` secrets | GitHub Secrets (certificates, API keys) |
| No signing needed | Code signing (certificates + provisioning profiles) |
| Docker for reproducible builds | Not applicable — macOS runners are VMs |
| `package-lock.json` cache | SPM cache + Ruby gems cache |
| Netlify / Vercel deploy preview | TestFlight build from PR branch |
| Semantic Release | Fastlane + build number management |

---

## Teaching Flow

### Introduction

**What to Cover:**
- Manual publishing works, but it doesn't scale. Every release means: open Xcode, archive, wait, upload, wait. CI/CD automates this.
- iOS CI/CD is harder than web CI/CD for three reasons: requires macOS (can't use cheap Linux runners), code signing needs certificates not in the repo, and Apple's toolchain is large and slow.
- We'll solve each of these problems: GitHub Actions provides macOS runners, Fastlane Match handles signing, and caching makes builds tolerable.
- By the end, merging to main will automatically run tests and deploy to TestFlight.

**Opening Questions to Assess Level:**
1. "Have you used CI/CD before? GitHub Actions, Jenkins, CircleCI, or anything else?"
2. "What part of your iOS build process is most painful to do manually?"
3. "Have you used Fastlane or any iOS build automation tools?"

**Adapt based on responses:**
- If they've used CI/CD for web: Move quickly through GitHub Actions basics. Focus on iOS-specific challenges (signing, macOS runners).
- If new to CI/CD: Take time explaining the concept and value. Start with a simple "build and test" workflow before adding complexity.
- If they've used Fastlane: Skip Fastlane basics, focus on CI integration and Match.
- If they've used Jenkins or other CI for iOS: Focus on GitHub Actions syntax differences and Fastlane Match as the modern signing solution.

**Opening framing:**
"Right now, releasing your app means opening Xcode, waiting for an archive, clicking through the upload wizard, and hoping you remembered to increment the build number. That works for a solo developer releasing once a month. But if you're releasing weekly, or working with a team, or you want tests to run on every PR — you need automation. We're going to build a pipeline where pushing code triggers tests, and merging to main automatically uploads to TestFlight. The pieces: GitHub Actions runs the workflow, Fastlane handles the iOS-specific build and upload commands, and Fastlane Match solves the hardest part — code signing without a human clicking buttons in Xcode."

---

### Section 1: CI/CD Concepts for iOS

**Core Concept to Teach:**
CI/CD automates the build-test-deploy cycle. For iOS, this means: on every push, run tests; on merge to main, build, sign, and upload to TestFlight. iOS CI/CD is uniquely challenging because it requires macOS, code signing, and large toolchains.

**How to Explain:**
1. Define CI and CD in the iOS context
2. Show the pipeline stages
3. Explain what makes iOS harder than web
4. Introduce the tools (GitHub Actions + Fastlane)

**The iOS CI/CD Pipeline:**

```
Push/PR → Build → Test → Sign → Upload → TestFlight
  │         │       │      │       │          │
  │     Compile   Unit   Code   Upload    Distribute
  │     project   and    sign   to App    to testers
  │               UI     with   Store
  │              tests   certs  Connect
  │
  └── GitHub Actions triggers the workflow
```

**iOS-Specific Challenges:**

"Three things make iOS CI/CD harder than web CI/CD:"

1. **macOS requirement**: iOS builds require Xcode, which only runs on macOS. You can't use cheap Linux runners. GitHub-hosted macOS runners cost about 10x more per minute than Linux runners.

2. **Code signing**: To build for distribution, you need certificates and provisioning profiles. These are normally in your Keychain and managed by Xcode. In CI, there's no Keychain set up — you have to get certificates onto the runner somehow.

3. **Build times**: Xcode projects are large. A clean build + tests can take 10-30 minutes. Caching helps but doesn't eliminate the wait.

**GitHub-Hosted vs Self-Hosted Runners:**

| | GitHub-Hosted | Self-Hosted |
|---|---|---|
| Cost | ~$0.08/min (macOS) | Your hardware cost |
| Maintenance | None | You maintain the machine |
| Clean environment | Yes (fresh VM each run) | Persistent (can accumulate state) |
| Xcode versions | Pre-installed (several) | You install what you need |
| Availability | On-demand | Must be online |
| Best for | Most teams | Large teams with high build volume |

"Start with GitHub-hosted runners. They're more expensive per minute but zero maintenance. Consider self-hosted when CI costs become significant or you need faster builds."

**Common Misconceptions:**
- Misconception: "I can build iOS on Linux" → Clarify: "No. Xcode and the iOS SDK only run on macOS. This is a hard requirement."
- Misconception: "CI should do exactly what I do locally" → Clarify: "CI should be more strict — run all tests, enforce code signing, validate the archive. It catches things you might skip locally."
- Misconception: "GitHub Actions is the only option" → Clarify: "Other CI services work too — Bitrise, CircleCI, Xcode Cloud, and others. We focus on GitHub Actions because it integrates with GitHub and is widely used, but the concepts transfer."

**Verification Questions:**
1. "Why can't iOS CI/CD use Linux runners like web CI/CD does?"
2. "What are the three main challenges specific to iOS CI/CD?"
3. Multiple choice: "Which is the most expensive part of iOS CI/CD? A) Running tests B) Uploading to TestFlight C) macOS runner compute time D) Fastlane setup"

---

### Section 2: Your First GitHub Actions Workflow

**Core Concept to Teach:**
A GitHub Actions workflow is a YAML file in `.github/workflows/` that defines automated steps triggered by events (push, PR, manual). For iOS, the key elements are: macOS runner, Xcode version selection, building with xcodebuild, and running tests.

**How to Explain:**
1. Show the simplest possible iOS workflow
2. Explain each part
3. Build up to a practical test workflow
4. Cover trigger configuration

**The Simplest Workflow:**

```yaml
name: iOS Tests

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  test:
    runs-on: macos-15
    steps:
      - uses: actions/checkout@v4

      - name: Select Xcode version
        run: sudo xcode-select -s /Applications/Xcode_16.2.app

      - name: Build and test
        run: |
          xcodebuild test \
            -project YourApp.xcodeproj \
            -scheme YourApp \
            -destination 'platform=iOS Simulator,name=iPhone 16,OS=18.2' \
            -resultBundlePath TestResults
```

**Walk Through:**
- `on:` — When the workflow triggers. This runs on PRs targeting main and pushes to main.
- `runs-on: macos-15` — Use a macOS runner. GitHub provides several macOS versions with Xcode pre-installed.
- `actions/checkout@v4` — Checks out your code. Same as any CI.
- `xcode-select` — Selects which Xcode version to use. Multiple versions are pre-installed on GitHub runners.
- `xcodebuild test` — Builds and runs tests. The `-destination` flag specifies which simulator to use.

"In web terms: this is like your GitHub Actions workflow that runs `npm install && npm test`, except `xcodebuild` replaces npm and you need a macOS runner instead of Ubuntu."

**Trigger Configuration:**

Show common trigger patterns:
```yaml
# Only on PRs (most common for tests)
on:
  pull_request:
    branches: [main]

# On push to main (for deployment)
on:
  push:
    branches: [main]

# Manual trigger (for ad-hoc builds)
on:
  workflow_dispatch:

# All three
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]
  workflow_dispatch:
```

**Workspace vs Project:**

"If your app uses Swift Package Manager or CocoaPods, you have a `.xcworkspace` instead of (or in addition to) `.xcodeproj`:"

```yaml
# For workspace-based projects
xcodebuild test \
  -workspace YourApp.xcworkspace \
  -scheme YourApp \
  -destination 'platform=iOS Simulator,name=iPhone 16,OS=18.2'
```

**Common Misconceptions:**
- Misconception: "I need Docker for reproducible builds" → Clarify: "GitHub-hosted macOS runners are VMs that start fresh every time. They're inherently reproducible without Docker (which doesn't support macOS anyway)."
- Misconception: "The simulator destination string doesn't matter much" → Clarify: "It must match exactly — the simulator name and OS version must be available on the runner. Check GitHub's runner documentation for what's pre-installed."
- Misconception: "I should test on every iOS version" → Clarify: "Test on your deployment target and the latest version. Matrix builds for multiple versions are possible but expensive (each run is a separate macOS runner)."

**Verification Questions:**
1. "What would you change in this workflow to also run on PRs targeting a `develop` branch?"
2. "Your build fails with 'Unable to find a destination matching the provided destination specifier.' What's wrong?"
3. "Why does the workflow use `sudo xcode-select` instead of just running xcodebuild?"

**Good answer indicators:**
- They can modify the trigger configuration
- They understand the destination specifier must match available simulators
- They know multiple Xcode versions are installed and xcode-select chooses which one to use

---

### Section 3: Fastlane Setup

**Core Concept to Teach:**
Fastlane is a Ruby-based automation tool for iOS and Android build/release tasks. It wraps xcodebuild and Apple APIs into higher-level commands (called "actions") and organizes them into "lanes" (workflows). While you can use xcodebuild directly in CI, Fastlane handles edge cases and provides convenient actions for signing, building, uploading, and more.

**How to Explain:**
1. Explain what Fastlane provides over raw xcodebuild
2. Walk through installation and initialization
3. Show the key files (Fastfile, Appfile)
4. Cover the essential actions: scan, gym, pilot
5. Demonstrate running Fastlane locally before CI

**Why Fastlane:**

"You can do everything with raw xcodebuild and xcrun commands. Fastlane's value is:"
- Handles xcodebuild's verbose output and exit codes
- Provides actions for App Store Connect API (no browser needed)
- Manages code signing with Match (we'll cover this next)
- Has a large community with solutions for common problems
- Encapsulates complex multi-step processes in a single lane

"Think of Fastlane like a task runner (Gulp, Grunt) or build tool (Make) — it wraps lower-level commands into reusable, configurable tasks."

**Installation and Initialization:**

```bash
# Install Fastlane via Homebrew (recommended for macOS)
brew install fastlane

# Or via RubyGems
gem install fastlane

# Initialize in your project directory
cd /path/to/your/ios/project
fastlane init
```

"During `fastlane init`, choose option 4 (Manual setup) for the most control. This creates:"
- `fastlane/Fastfile` — Lane definitions (your automated workflows)
- `fastlane/Appfile` — App-level configuration (bundle ID, Apple ID, team ID)

**The Appfile:**

```ruby
app_identifier("com.yourcompany.yourapp")
apple_id("your@email.com")
team_id("ABCDE12345")  # Your Apple Developer Team ID

itc_team_id("123456789")  # App Store Connect Team ID (if different)
```

"The Appfile stores configuration so you don't repeat it in every lane."

**The Fastfile — Essential Lanes:**

```ruby
default_platform(:ios)

platform :ios do
  desc "Run tests"
  lane :tests do
    scan(
      project: "YourApp.xcodeproj",
      scheme: "YourApp",
      devices: ["iPhone 16"],
      clean: true
    )
  end

  desc "Build for App Store"
  lane :build do
    gym(
      project: "YourApp.xcodeproj",
      scheme: "YourApp",
      export_method: "app-store",
      output_directory: "build"
    )
  end

  desc "Upload to TestFlight"
  lane :beta do
    build
    pilot(
      skip_waiting_for_build_processing: true
    )
  end
end
```

**Walk Through:**
- `scan` = run tests. Wraps `xcodebuild test` with better output formatting and exit code handling.
- `gym` = build and archive. Wraps `xcodebuild archive` and `xcodebuild -exportArchive`. Produces an `.ipa` file.
- `pilot` = upload to TestFlight. Wraps the App Store Connect API. Handles authentication.
- `lane :beta` calls `build` first (lanes can call other lanes), then uploads.

**Running Locally:**

"Always test Fastlane locally before putting it in CI:"

```bash
# Run tests
fastlane tests

# Build (will fail without proper signing — that's expected if you haven't set up Match yet)
fastlane build
```

"If `fastlane tests` works locally, it should work in CI (with the right Xcode version and simulator)."

**Common Misconceptions:**
- Misconception: "Fastlane replaces GitHub Actions" → Clarify: "They work together. GitHub Actions is the CI orchestrator (when to run, what environment). Fastlane is the build tool (how to build, test, upload). GitHub Actions calls Fastlane."
- Misconception: "I need to learn Ruby to use Fastlane" → Clarify: "The Fastfile syntax is Ruby, but you don't need Ruby knowledge. It's declarative — you call actions with parameters. Copy-paste from documentation works for most setups."
- Misconception: "Fastlane is required for iOS CI/CD" → Clarify: "You can use raw xcodebuild commands. Fastlane makes it easier, especially for signing and uploading. For a simple 'run tests' workflow, xcodebuild is fine."

**Verification Questions:**
1. "What are the three Fastlane actions that correspond to testing, building, and uploading?"
2. "Your Fastlane tests pass locally but fail in CI. What's the first thing you'd check?"
3. "What goes in the Appfile vs the Fastfile?"

---

### Section 4: Code Signing in CI

**Core Concept to Teach:**
Code signing is the hardest part of iOS CI/CD. In local development, Xcode manages signing automatically using certificates in your Keychain. In CI, there's no Keychain pre-configured and no Xcode UI to click. You need to get certificates and provisioning profiles onto the CI runner programmatically. Fastlane Match is the recommended solution.

**How to Explain:**
1. Explain why signing is hard in CI
2. Present the two approaches: Match (recommended) and manual
3. Walk through Match setup in detail
4. Cover App Store Connect API keys

**The Problem:**

"When you archive locally, Xcode uses certificates from your Keychain and provisioning profiles it manages. In CI:"
- There's no Keychain with your certificates
- There's no Xcode UI to select signing options
- You can't run Xcode's automatic signing (it requires interactive login)
- You need the signing materials to be available programmatically

**Approach 1: Fastlane Match (Recommended)**

"Match stores your certificates and provisioning profiles in an encrypted, private Git repository (or cloud storage). In CI, Match clones that repo, decrypts the files, and installs them into the runner's Keychain."

```
Developer Portal                Private Git Repo
    │                                │
    │  certificates                  │  encrypted certs
    │  profiles      ──Match sync──→ │  encrypted profiles
    │                                │
                                     │
CI Runner                            │
    │                                │
    │  ←──── Match decrypt & install ┘
    │
    │  Certificates in Keychain ✓
    │  Profiles installed ✓
    │  Ready to sign ✓
```

**Setting Up Match:**

"Step 1: Create a private Git repo for certificates (e.g., `yourorg/ios-certificates`)."

"Step 2: Initialize Match:"
```bash
fastlane match init
# Choose 'git' storage
# Enter the URL of your private certificates repo
```

"Step 3: Generate certificates and profiles:"
```bash
# Generate development certificates and profiles
fastlane match development

# Generate App Store distribution certificates and profiles
fastlane match appstore
```

"Match generates the certificates, uploads them to the private repo (encrypted), and installs them locally. Every team member runs the same commands to get identical certificates."

"Step 4: Add Match to your Fastfile:"
```ruby
lane :beta do
  match(type: "appstore", readonly: true)
  gym(
    project: "YourApp.xcodeproj",
    scheme: "YourApp",
    export_method: "app-store"
  )
  pilot(skip_waiting_for_build_processing: true)
end
```

"`readonly: true` is important in CI — it means Match won't try to create new certificates, only download existing ones. This prevents CI from accidentally revoking and regenerating certificates."

**Approach 2: Manual Certificate Installation**

"If you don't want to use Match, you can install certificates manually using GitHub Secrets:"

1. Export your distribution certificate as a `.p12` file (from Keychain Access)
2. Base64-encode it: `base64 -i certificate.p12 | pbcopy`
3. Store the base64 string and password as GitHub Secrets
4. In the workflow, decode and install into a temporary Keychain

```yaml
- name: Install certificates
  env:
    CERTIFICATE_BASE64: ${{ secrets.DISTRIBUTION_CERTIFICATE_BASE64 }}
    CERTIFICATE_PASSWORD: ${{ secrets.CERTIFICATE_PASSWORD }}
  run: |
    # Create a temporary keychain
    security create-keychain -p "" build.keychain
    security default-keychain -s build.keychain
    security unlock-keychain -p "" build.keychain

    # Decode and import the certificate
    echo $CERTIFICATE_BASE64 | base64 --decode > certificate.p12
    security import certificate.p12 -k build.keychain \
      -P $CERTIFICATE_PASSWORD -T /usr/bin/codesign
    security set-key-partition-list -S apple-tool:,apple: \
      -s -k "" build.keychain

    # Clean up
    rm certificate.p12
```

"This works but is brittle. If certificates expire, you need to update secrets manually. Match automates all of this."

**App Store Connect API Keys:**

"To upload to TestFlight from CI, you need to authenticate with App Store Connect. Instead of storing your Apple ID password, use an API key:"

1. In App Store Connect > Users and Access > Integrations > App Store Connect API
2. Generate a key with "App Manager" role
3. Download the `.p8` file (you can only download it once)
4. Store the key ID, issuer ID, and key contents as GitHub Secrets

```ruby
# In Fastfile — authenticate with API key
lane :beta do
  api_key = app_store_connect_api_key(
    key_id: ENV["APP_STORE_CONNECT_KEY_ID"],
    issuer_id: ENV["APP_STORE_CONNECT_ISSUER_ID"],
    key_content: ENV["APP_STORE_CONNECT_KEY_CONTENT"]
  )

  match(type: "appstore", readonly: true, api_key: api_key)
  gym(
    project: "YourApp.xcodeproj",
    scheme: "YourApp",
    export_method: "app-store"
  )
  pilot(api_key: api_key, skip_waiting_for_build_processing: true)
end
```

**Common Misconceptions:**
- Misconception: "I can use Xcode's automatic signing in CI" → Clarify: "Automatic signing requires interactive Xcode sessions with Apple ID login. It doesn't work in headless CI environments."
- Misconception: "I should commit certificates to the repo" → Clarify: "Never commit unencrypted certificates to a repository. Match encrypts them in a separate private repo. Manual approach uses GitHub Secrets."
- Misconception: "Match is a Fastlane requirement" → Clarify: "Match is one Fastlane action. You can use Fastlane without Match (manual signing) or Match without the rest of Fastlane."
- Misconception: "Every developer needs their own certificate" → Clarify: "Match uses shared certificates. The whole team uses the same distribution certificate, stored encrypted in the Match repo. This is actually more reliable than each developer managing their own."

**Verification Questions:**
1. "Why can't you use Xcode's automatic signing in CI?"
2. "What does `readonly: true` do in the Match action, and why is it important in CI?"
3. "What's the advantage of an App Store Connect API key over username/password for CI authentication?"
4. Debugging: "Your CI build fails with 'No signing certificate found.' You've set up Match. What would you check?"

**Good answer indicators:**
- They understand CI has no interactive login or pre-configured Keychain
- `readonly: true` prevents CI from modifying certificates
- API keys don't require 2FA and can be scoped to specific roles
- They would check: Match secrets are configured, the certificates repo is accessible, the bundle ID matches

---

### Section 5: Automated TestFlight Deployment

**Core Concept to Teach:**
With signing solved, the deployment workflow is straightforward: on merge to main, build the app, sign it, upload to TestFlight, and manage version numbers. The complete pipeline ties together everything from the previous sections.

**How to Explain:**
1. Show the complete deployment workflow
2. Cover version and build number management
3. Explain the workflow structure
4. Discuss changelog generation

**The Complete Deployment Workflow:**

```yaml
name: Deploy to TestFlight

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: macos-15
    steps:
      - uses: actions/checkout@v4

      - name: Select Xcode
        run: sudo xcode-select -s /Applications/Xcode_16.2.app

      - name: Install Fastlane
        run: brew install fastlane

      - name: Deploy to TestFlight
        env:
          MATCH_PASSWORD: ${{ secrets.MATCH_PASSWORD }}
          MATCH_GIT_URL: ${{ secrets.MATCH_GIT_URL }}
          APP_STORE_CONNECT_KEY_ID: ${{ secrets.APP_STORE_CONNECT_KEY_ID }}
          APP_STORE_CONNECT_ISSUER_ID: ${{ secrets.APP_STORE_CONNECT_ISSUER_ID }}
          APP_STORE_CONNECT_KEY_CONTENT: ${{ secrets.APP_STORE_CONNECT_KEY_CONTENT }}
        run: fastlane beta
```

"That's it. When code is pushed to main, GitHub Actions spins up a macOS runner, installs Fastlane, and runs the `beta` lane — which uses Match for signing, gym for building, and pilot for uploading."

**Build Number Management:**

"The build number must increment with every upload to App Store Connect. In CI, you can automate this:"

```ruby
lane :beta do
  api_key = app_store_connect_api_key(
    key_id: ENV["APP_STORE_CONNECT_KEY_ID"],
    issuer_id: ENV["APP_STORE_CONNECT_ISSUER_ID"],
    key_content: ENV["APP_STORE_CONNECT_KEY_CONTENT"]
  )

  # Fetch the latest build number from App Store Connect and increment
  latest_build = latest_testflight_build_number(api_key: api_key)
  increment_build_number(build_number: latest_build + 1)

  match(type: "appstore", readonly: true, api_key: api_key)
  gym(
    project: "YourApp.xcodeproj",
    scheme: "YourApp",
    export_method: "app-store"
  )
  pilot(api_key: api_key, skip_waiting_for_build_processing: true)
end
```

"The `latest_testflight_build_number` action queries App Store Connect for the highest build number, then `increment_build_number` sets the next one. No manual version bumping needed."

**Alternative: Use the Git commit count as the build number:**
```ruby
build_number = sh("git rev-list --count HEAD").strip
increment_build_number(build_number: build_number)
```

**Changelog Generation:**

"You can auto-generate a changelog from git commits:"

```ruby
lane :beta do
  # ... signing and build steps ...

  changelog = changelog_from_git_commits(
    merge_commit_filtering: "exclude_merges"
  )

  pilot(
    api_key: api_key,
    changelog: changelog,
    skip_waiting_for_build_processing: true
  )
end
```

"This sets the 'What to Test' field in TestFlight to your recent commit messages. Not always user-friendly, but useful for internal testing."

**Common Misconceptions:**
- Misconception: "I need to manually bump the build number before merging" → Clarify: "CI can auto-increment the build number by querying the latest from App Store Connect. No manual step needed."
- Misconception: "TestFlight deploys are instant" → Clarify: "After upload, Apple processes the build (a few minutes to an hour). `skip_waiting_for_build_processing: true` lets CI finish without waiting for processing."
- Misconception: "I need separate workflows for testing and deployment" → Clarify: "You can use one workflow with conditional steps, or separate workflows for clarity. Separate workflows are usually cleaner."

**Verification Questions:**
1. "What GitHub Actions trigger would you use for deployment vs testing?"
2. "How does the CI pipeline know what build number to use?"
3. "Your deployment workflow succeeds but the build doesn't appear in TestFlight. What do you check?"

---

### Section 6: Advanced Workflow Patterns

**Core Concept to Teach:**
Once the basic pipeline works, optimizations and enhancements make it faster and more robust. Caching reduces build times, conditional steps prevent unnecessary work, and matrix builds test across configurations.

**How to Explain:**
1. Show dependency caching (SPM, Ruby gems)
2. Cover conditional steps
3. Introduce matrix builds
4. Discuss notifications

**Caching Dependencies:**

"SPM resolution and Fastlane gem installation can add minutes to every build. Cache them:"

```yaml
- name: Cache SPM packages
  uses: actions/cache@v4
  with:
    path: |
      ~/Library/Developer/Xcode/DerivedData/**/SourcePackages
    key: spm-${{ hashFiles('**/Package.resolved') }}
    restore-keys: spm-

- name: Cache Ruby gems (Fastlane)
  uses: actions/cache@v4
  with:
    path: vendor/bundle
    key: gems-${{ hashFiles('Gemfile.lock') }}
    restore-keys: gems-
```

"The cache key includes a hash of the lock file — when dependencies change, the cache invalidates. When they don't, the cached packages are restored, saving several minutes."

**Conditional Steps:**

```yaml
jobs:
  test:
    runs-on: macos-15
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: fastlane tests

  deploy:
    needs: test
    runs-on: macos-15
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Deploy
        run: fastlane beta
```

"`needs: test` means deploy only runs after tests pass. `if: github.ref == 'refs/heads/main'` means deploy only runs on main — PR pushes only run tests."

**Matrix Builds:**

"Test on multiple iOS versions:"

```yaml
strategy:
  matrix:
    destination:
      - 'platform=iOS Simulator,name=iPhone 16,OS=18.2'
      - 'platform=iOS Simulator,name=iPhone 15,OS=17.5'
steps:
  - name: Test
    run: |
      xcodebuild test \
        -project YourApp.xcodeproj \
        -scheme YourApp \
        -destination '${{ matrix.destination }}'
```

"Each matrix entry runs as a separate job in parallel. Be mindful of costs — each job uses its own macOS runner."

**Notifications:**

```yaml
- name: Notify on failure
  if: failure()
  uses: slackapi/slack-github-action@v2.0.0
  with:
    webhook: ${{ secrets.SLACK_WEBHOOK }}
    payload: |
      {"text": "iOS build failed: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"}
```

**Common Misconceptions:**
- Misconception: "Caching always makes builds faster" → Clarify: "Caching SPM packages and gems helps. Caching DerivedData is unreliable — Xcode often ignores it. Don't spend time trying to cache the build itself."
- Misconception: "Matrix builds are free" → Clarify: "Each matrix entry is a separate macOS runner. Testing on 3 OS versions costs 3x the compute time."

**Verification Questions:**
1. "What would you cache to speed up an iOS CI build?"
2. "How do you ensure deployment only happens after tests pass?"
3. "Your team wants Slack notifications only when builds fail. How would you set that up?"

---

### Section 7: Practice Project

**Project Introduction:**
"Let's set up a complete CI/CD pipeline for your iOS project. You'll create two workflows: one that runs tests on every PR, and one that deploys to TestFlight on merge to main."

**Requirements:**
1. Create a `.github/workflows/test.yml` that runs tests on PRs to main
2. Set up Fastlane with test and beta lanes
3. Configure Fastlane Match for code signing in CI (or document the manual approach)
4. Create a `.github/workflows/deploy.yml` that deploys to TestFlight on merge to main
5. Add dependency caching to both workflows
6. Test the pipeline end-to-end

**Scaffolding Strategy:**
1. **If they want to try alone**: Give them the requirements and check in at milestones.
2. **If they want guidance**: Build up step by step — test workflow first, then Fastlane, then signing, then deployment.
3. **If they don't have a paid GitHub account with macOS runners**: Focus on writing the workflows and Fastlane configuration locally, testing Fastlane commands locally. The pipeline will work when pushed to a repo with macOS runner access.

**Checkpoints During Project:**
- After test workflow: "Does `fastlane tests` pass locally? Is the workflow YAML valid?"
- After Fastlane setup: "Can you run `fastlane tests` and `fastlane build` locally?"
- After Match: "Did Match generate certificates? Can you build with manual signing?"
- After deploy workflow: "Are all secrets configured in GitHub? Are environment variables mapped correctly?"
- After end-to-end test: "Did the test workflow trigger on a PR? Did deployment trigger on merge?"

**If They Get Stuck:**
- On YAML: "YAML indentation matters. Use 2 spaces, not tabs. If in doubt, use a YAML validator."
- On Fastlane: "Run `fastlane tests` locally first. If it works locally, the CI version should be the same."
- On Match: "Verify the Match repo URL is correct and the MATCH_PASSWORD secret is set. Run `fastlane match appstore --readonly` locally to test."
- On deployment: "Check the Actions tab in GitHub. Click the failed run to see logs. The error usually points to a specific step."

---

## Wrap-Up and Next Steps

**Review Key Takeaways:**
1. iOS CI/CD requires macOS runners and solving code signing
2. GitHub Actions provides the CI orchestration; Fastlane handles iOS-specific build and upload
3. Fastlane Match is the recommended solution for code signing in CI
4. App Store Connect API keys replace username/password for CI authentication
5. Caching SPM packages and Ruby gems reduces build times
6. The complete pipeline: tests on PRs, deploy to TestFlight on merge to main

**Assess Confidence:**
"On a scale of 1-10, how confident do you feel setting up iOS CI/CD?"
- 1-4: Walk through each step again with their specific project. Focus on the part that's most confusing.
- 5-7: Normal. Suggest setting up the basic test workflow first and adding deployment once that's solid.
- 8-10: Consider exploring: Xcode Cloud, release automation, multi-environment deployments.

**Suggest Next Steps:**
- "Set up the test workflow first — it provides immediate value with low complexity"
- "Add deployment only after tests are reliable in CI"
- "Consider the My First iOS App ascent to build a complete app using all the routes you've learned"

---

## Adaptive Teaching Strategies

### If Learner is Struggling
- Focus on the test workflow only — skip Fastlane and deployment until they're comfortable with basic GitHub Actions
- Use the simplest possible workflow (just `xcodebuild test`) without Fastlane
- If YAML syntax is confusing, compare it to JSON — it's just a different serialization format
- If code signing is overwhelming, skip Match and start with a tests-only workflow (no signing needed for simulator tests)

### If Learner is Excelling
- Discuss multi-environment deployments (staging, production)
- Cover branch protection rules and required status checks
- Introduce Xcode Cloud as an Apple-native alternative
- Discuss build matrices for testing across device sizes and OS versions
- Explore release automation with git tags and semantic versioning

### If Learner Seems Disengaged
- Focus on the pain point they mentioned — if manual builds are the issue, jump to deployment
- If they're worried about cost, discuss the free tier and self-hosted runners
- If they already have CI elsewhere, focus on migration strategies

### Different Learning Styles
- **Visual learners**: Draw pipeline diagrams showing the flow from push to TestFlight
- **Hands-on learners**: Start writing the workflow immediately, explain as you go
- **Conceptual learners**: Spend time on why each piece exists before configuring it
- **Example-driven learners**: Show complete working workflows and modify them

---

## Troubleshooting Common Issues

### GitHub Actions Problems
- **"No runner available"**: macOS runners may have longer queue times. Check GitHub Status for outages.
- **"Xcode not found"**: The Xcode version path in `xcode-select` doesn't exist on the runner. Check GitHub's runner documentation for available versions.
- **"Simulator not found"**: The destination specifier doesn't match available simulators. List available simulators with `xcrun simctl list devices` in a workflow step.

### Fastlane Problems
- **"Could not find Fastfile"**: Fastlane is running from the wrong directory. Add `working-directory: ./ios` to the workflow step if your project is in a subdirectory.
- **"Unable to locate Xcode"**: Xcode isn't selected. Add the `xcode-select` step before running Fastlane.
- **"Authentication error"**: API key secrets aren't configured or are incorrect. Verify each secret in GitHub Settings > Secrets.

### Code Signing Problems
- **"No signing certificate found"**: Match hasn't installed certificates. Check MATCH_PASSWORD and MATCH_GIT_URL secrets. Run `fastlane match appstore --readonly` in CI to debug.
- **"Provisioning profile doesn't match"**: The profile in Match doesn't include the right bundle ID or certificate. Run `fastlane match nuke appstore` locally to regenerate (careful — this revokes existing certificates).
- **"Code signing is required for product type 'Application'"**: You're trying to build for a device without distribution signing. Ensure Match runs before gym.

### Upload Problems
- **"App Store Connect credentials error"**: API key secrets are misconfigured. Verify KEY_ID, ISSUER_ID, and KEY_CONTENT are all correct and the key has the right role.
- **"Invalid binary"**: The archive was built incorrectly. Check the export method matches "app-store" and the signing identity is a distribution certificate.
- **"Redundant binary upload"**: Same build number as a previous upload. Ensure build number auto-incrementing is working.

---

## Teaching Notes

**Key Emphasis Points:**
- Start simple. A test workflow provides immediate value. Add complexity (Fastlane, signing, deployment) incrementally.
- Code signing in CI is the hard part. Everything else is configuration. If they understand Match, they can handle the rest.
- Always test locally first. `fastlane tests` should work on their machine before it goes into CI.
- Cost awareness matters. macOS runners are expensive. Caching and conditional runs save money.

**Pacing Guidance:**
- Section 1 (Concepts): Brief for anyone with CI experience. Take time for CI newcomers.
- Section 2 (First Workflow): Foundational. Make sure they have a working test workflow before moving on.
- Section 3 (Fastlane): Moderate pace. Ensure they can run Fastlane locally.
- Section 4 (Code Signing): The hardest section. Take extra time. This is where most people get stuck.
- Section 5 (Deployment): Quick if signing is solved — it's just connecting the pieces.
- Section 6 (Advanced): Brief. Cover caching and conditionals. Skip matrix builds unless they need them.
- Section 7 (Practice): Allow the most time. Building a real pipeline requires iterating on failures.

**Success Indicators:**
You'll know they've got it when they:
- Can explain the CI/CD pipeline from trigger to TestFlight
- Write workflow YAML without copying from templates
- Know where to look when a CI build fails (logs, secrets, signing)
- Can set up Match and explain why it's needed
- Have a working test workflow running on their repo
