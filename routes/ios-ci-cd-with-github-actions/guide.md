---
title: iOS CI/CD with GitHub Actions
route_map: /routes/ios-ci-cd-with-github-actions/map.md
paired_sherpa: /routes/ios-ci-cd-with-github-actions/sherpa.md
prerequisites:
  - App Store publishing basics (see app-store-publishing)
  - Git and GitHub familiarity
  - A working iOS app with tests
topics:
  - GitHub Actions
  - Fastlane
  - Code Signing in CI
  - TestFlight Automation
  - Continuous Deployment
---

# iOS CI/CD with GitHub Actions

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/ios-ci-cd-with-github-actions/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/ios-ci-cd-with-github-actions/map.md` for the high-level overview.

## Overview

Manually archiving and uploading builds gets old fast. Open Xcode, wait for the archive, click through the upload wizard, hope you remembered to bump the build number — and do it all again next week. CI/CD automates this: tests run on every PR, and merging to main automatically builds, signs, and uploads to TestFlight.

If you've set up CI/CD for web projects, the concepts are the same — but iOS adds unique challenges: macOS is required (no cheap Linux runners), code signing needs certificates that aren't in the repo, and Apple's toolchain is large and slow. This guide solves each of those problems with GitHub Actions and Fastlane.

## Learning Objectives

By the end of this guide, you will be able to:
- Write a GitHub Actions workflow that builds and tests an iOS project
- Set up Fastlane with lanes for testing, building, and uploading to TestFlight
- Solve code signing in CI using Fastlane Match
- Automate TestFlight deployments on merge to main
- Manage version and build numbers automatically in CI
- Cache dependencies to speed up builds
- Configure conditional steps and notifications

## Prerequisites

Before starting, you should have:
- App Store publishing basics — certificates, provisioning profiles, App Store Connect, TestFlight (see app-store-publishing route)
- Familiarity with Git and GitHub (branches, PRs, merging)
- A working iOS app with at least one test
- (Helpful) Basic YAML syntax — if you've never seen YAML, it's JSON with less punctuation

## Setup

You'll need:

1. **A GitHub repository** with your iOS project pushed to it
2. **Xcode** installed locally (for testing Fastlane before CI)
3. **Homebrew** (for installing Fastlane): check with `brew --version`
4. **Fastlane** installed:

```bash
brew install fastlane
```

Verify:
```bash
fastlane --version
```

You should see something like:
```
fastlane 2.x.x
```

5. **An Apple Developer account** with App Store Connect access (from app-store-publishing)

---

## Section 1: CI/CD Concepts for iOS

### The Pipeline

CI/CD stands for Continuous Integration (automatically build and test) and Continuous Deployment (automatically release). For iOS, the pipeline looks like this:

```
Push/PR → Build → Test → Sign → Upload → TestFlight
```

Every push or PR triggers the pipeline. Tests run first — if they fail, the pipeline stops. If they pass and the push is to main, the app is built, signed, uploaded to App Store Connect, and distributed via TestFlight. No manual steps.

### What Makes iOS Different

If you've done CI/CD for web apps, three things are different:

**1. macOS is required.** iOS builds need Xcode, which only runs on macOS. You can't use cheap Linux runners. GitHub-hosted macOS runners cost about 10x more per minute than Linux runners (~$0.08/min vs ~$0.008/min).

**2. Code signing needs certificates.** To build for distribution, you need certificates and provisioning profiles. Locally, Xcode manages these through your Keychain. In CI, there's no Keychain — you need to get certificates onto the runner programmatically.

**3. Builds are slow.** Xcode projects are large. A clean build plus tests can take 10-30 minutes. Caching helps, but iOS CI will never be as fast as a JavaScript test suite.

### GitHub-Hosted vs Self-Hosted Runners

| | GitHub-Hosted | Self-Hosted |
|---|---|---|
| Cost | ~$0.08/min | Your hardware cost |
| Maintenance | None | You maintain the machine |
| Environment | Fresh VM every run | Persistent state |
| Xcode versions | Several pre-installed | You install what you need |
| Best for | Most teams | High build volume |

Start with GitHub-hosted. They're more expensive per minute but require zero maintenance. Consider self-hosted when CI costs become a line item in your budget.

### The Tools

Two tools work together:
- **GitHub Actions** — the CI orchestrator. Defines when to run (on push, PR, merge) and where (macOS runner).
- **Fastlane** — the iOS build tool. Handles building, testing, signing, and uploading. Wraps xcodebuild and Apple's APIs into simple commands.

You can use GitHub Actions without Fastlane (raw xcodebuild commands), but Fastlane handles signing, uploading, and edge cases that would otherwise require a lot of shell scripting.

### Checkpoint 1

Before moving on, make sure you understand:
- [ ] iOS CI/CD requires macOS runners (no Linux)
- [ ] Code signing in CI is the unique challenge — no Keychain or Xcode UI
- [ ] GitHub Actions orchestrates the pipeline; Fastlane handles iOS-specific tasks
- [ ] GitHub-hosted runners are easiest to start with

---

## Section 2: Your First GitHub Actions Workflow

### Workflow File Structure

GitHub Actions workflows are YAML files in `.github/workflows/`. Create the directory structure:

```bash
mkdir -p .github/workflows
```

### A Test Workflow

Create `.github/workflows/test.yml`:

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
      - name: Checkout code
        uses: actions/checkout@v4

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

**What each part does:**

- `name: iOS Tests` — The workflow name, shown in the Actions tab on GitHub.
- `on:` — Triggers. This runs on PRs targeting main and direct pushes to main.
- `runs-on: macos-15` — Use a macOS 15 runner. GitHub provides several macOS versions with Xcode pre-installed.
- `actions/checkout@v4` — Checks out your repository code. Required in every workflow.
- `xcode-select` — Selects which Xcode version to use. Multiple versions are pre-installed; this picks the one you want.
- `xcodebuild test` — Builds the project and runs tests on the specified simulator.

The `-destination` flag specifies the simulator. It must match what's available on the runner — if the runner doesn't have "iPhone 16" with iOS 18.2, the build fails.

### For Workspace-Based Projects

If your app uses Swift Package Manager dependencies or CocoaPods, you likely have a `.xcworkspace`:

```yaml
- name: Build and test
  run: |
    xcodebuild test \
      -workspace YourApp.xcworkspace \
      -scheme YourApp \
      -destination 'platform=iOS Simulator,name=iPhone 16,OS=18.2'
```

### Trigger Patterns

Common configurations:

```yaml
# Tests on PRs only (most common)
on:
  pull_request:
    branches: [main]

# Deploy on push to main
on:
  push:
    branches: [main]

# Manual trigger (for ad-hoc builds)
on:
  workflow_dispatch:

# Combined: test on PRs, deploy on main push, allow manual
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]
  workflow_dispatch:
```

### Exercise 2.1: Create a Test Workflow

**Task:** Create a GitHub Actions workflow that runs your app's tests on every PR.

1. Create `.github/workflows/test.yml`
2. Configure it to trigger on PRs to main
3. Use the correct project/workspace file for your app
4. Select the appropriate Xcode version
5. Choose a simulator destination

<details>
<summary>Hint 1: Finding the right destination</summary>

Run this locally to list available simulators:
```bash
xcrun simctl list devices available
```

Use a simulator name and OS version that matches what's on the GitHub runner. Check [GitHub's runner images documentation](https://github.com/actions/runner-images) for what's pre-installed.
</details>

<details>
<summary>Hint 2: Finding your scheme name</summary>

Run this locally to list available schemes:
```bash
xcodebuild -list
```

The scheme name is case-sensitive and must match exactly.
</details>

<details>
<summary>If you want to test without pushing</summary>

You can validate your workflow YAML locally with:
```bash
# Install act (runs GitHub Actions locally)
brew install act

# Dry run (validates syntax)
act -n
```

Or simply check the YAML syntax is valid — indentation errors are the most common problem.
</details>

### Checkpoint 2

Before moving on, make sure you understand:
- [ ] Workflows are YAML files in `.github/workflows/`
- [ ] `runs-on: macos-15` selects a macOS runner with Xcode pre-installed
- [ ] `xcode-select` picks which Xcode version to use
- [ ] The simulator destination must match what's available on the runner

---

## Section 3: Fastlane Setup

### What Fastlane Does

Fastlane wraps xcodebuild and Apple's APIs into higher-level commands called "actions," organized into "lanes" (workflows). Three actions cover most needs:

| Action | What it does | Raw equivalent |
|--------|-------------|----------------|
| `scan` | Run tests | `xcodebuild test` |
| `gym` | Build and archive | `xcodebuild archive` + `-exportArchive` |
| `pilot` | Upload to TestFlight | App Store Connect API |

You can use raw xcodebuild in CI and it works fine for testing. Fastlane becomes valuable when you need signing (Match), uploading (Pilot), and the many edge cases these involve.

### Installing and Initializing

```bash
# Install
brew install fastlane

# Initialize in your project directory
cd /path/to/your/ios/project
fastlane init
```

During initialization, choose **option 4 (Manual setup)**. This creates:

```
your-project/
├── fastlane/
│   ├── Appfile    # App-level configuration
│   └── Fastfile   # Lane definitions
├── Gemfile        # Ruby dependencies
└── Gemfile.lock
```

### The Appfile

Configuration that's shared across all lanes:

```ruby
app_identifier("com.yourcompany.yourapp")  # Bundle ID
apple_id("your@email.com")                  # Your Apple ID
team_id("ABCDE12345")                       # Developer Portal Team ID
itc_team_id("123456789")                     # App Store Connect Team ID
```

Find your Team ID in the Apple Developer Portal under Membership. The App Store Connect Team ID is in App Store Connect > Users and Access.

### The Fastfile

Lane definitions — your automated workflows:

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

  desc "Build for App Store distribution"
  lane :build do
    gym(
      project: "YourApp.xcodeproj",
      scheme: "YourApp",
      export_method: "app-store",
      output_directory: "build"
    )
  end

  desc "Run tests, build, and upload to TestFlight"
  lane :beta do
    tests
    build
    pilot(
      skip_waiting_for_build_processing: true
    )
  end
end
```

**How it works:**
- `lane :tests` — Runs your test suite. `scan` wraps xcodebuild with better output formatting.
- `lane :build` — Archives and exports an `.ipa` file. `gym` handles the archive and export steps.
- `lane :beta` — Calls `tests` first, then `build`, then uploads to TestFlight with `pilot`. Lanes can call other lanes.

### Running Locally

Always test Fastlane locally before putting it in CI:

```bash
# Run tests
fastlane tests

# Build (may fail without distribution signing — that's expected)
fastlane build
```

If `fastlane tests` works locally, it should work in CI with the right Xcode version and simulator.

### Exercise 3.1: Set Up Fastlane

**Task:** Initialize Fastlane and create a test lane for your project.

1. Run `fastlane init` (choose manual setup)
2. Configure the Appfile with your app's details
3. Write a `tests` lane in the Fastfile
4. Run `fastlane tests` locally and verify it passes

<details>
<summary>Hint: If tests fail with "Unable to find a device"</summary>

The `devices` parameter in `scan` must match an available simulator. List them:
```bash
xcrun simctl list devices available
```

Use the exact name, e.g., `["iPhone 16"]` not `["iPhone16"]`.
</details>

<details>
<summary>Hint: If using a workspace</summary>

Replace `project:` with `workspace:` in both `scan` and `gym`:
```ruby
scan(
  workspace: "YourApp.xcworkspace",
  scheme: "YourApp",
  devices: ["iPhone 16"],
  clean: true
)
```
</details>

### Checkpoint 3

Before moving on, make sure you understand:
- [ ] Fastlane provides `scan` (test), `gym` (build), and `pilot` (upload) actions
- [ ] The Appfile stores app-level configuration; the Fastfile defines lanes
- [ ] Lanes can call other lanes (e.g., `beta` calls `tests` then `build`)
- [ ] Always test Fastlane locally before running in CI

---

## Section 4: Code Signing in CI

### The Problem

Locally, Xcode's automatic signing handles certificates and provisioning profiles through your Keychain. In CI, there's no pre-configured Keychain, no Xcode UI, and no interactive login. You need to get signing materials onto the runner programmatically.

This is the hardest part of iOS CI/CD. Once you solve it, everything else is straightforward.

### Fastlane Match (Recommended)

Match stores your certificates and provisioning profiles in an encrypted, private Git repository. In CI, Match clones that repo, decrypts the files, and installs them into the runner's Keychain.

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

**Why Match is better than manual:**
- Certificates are version-controlled (encrypted) and shareable
- Every developer and CI runner uses the same certificates
- Rotation and renewal are handled in one place
- No manually exporting .p12 files and storing them as secrets

### Setting Up Match

**Step 1: Create a private Git repository for certificates.**

Create a new, private repo (e.g., `yourorg/ios-certificates`). It should be empty — Match manages its contents.

**Step 2: Initialize Match.**

```bash
fastlane match init
```

Choose `git` for storage. Enter the URL of your certificates repo. This creates a `Matchfile`:

```ruby
git_url("https://github.com/yourorg/ios-certificates")
storage_mode("git")
type("appstore")
app_identifier(["com.yourcompany.yourapp"])
```

**Step 3: Generate certificates and profiles.**

```bash
# Generate App Store distribution certificates and profiles
fastlane match appstore
```

Match will:
1. Generate a distribution certificate (or use an existing one)
2. Create a provisioning profile linked to your certificate and app ID
3. Encrypt everything with a passphrase you choose
4. Push the encrypted files to your certificates repo
5. Install the certificate and profile on your machine

**Remember the passphrase** — you'll store it as `MATCH_PASSWORD` in GitHub Secrets.

**Step 4: Update the Fastfile to use Match.**

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

`readonly: true` is critical in CI — it prevents Match from trying to create or modify certificates. CI should only download and install existing ones.

### App Store Connect API Keys

To upload to TestFlight from CI, you need to authenticate with App Store Connect. Instead of storing your Apple ID and password (which requires 2FA), use an API key:

1. Go to App Store Connect > Users and Access > Integrations > App Store Connect API
2. Click + to generate a key
3. Choose the "App Manager" role
4. Download the `.p8` key file (you can only download it once — save it!)
5. Note the Key ID and Issuer ID shown on the page

Store these as GitHub Secrets:
- `APP_STORE_CONNECT_KEY_ID` — the Key ID
- `APP_STORE_CONNECT_ISSUER_ID` — the Issuer ID
- `APP_STORE_CONNECT_KEY_CONTENT` — the contents of the `.p8` file

Update the Fastfile to use the API key:

```ruby
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

### Manual Signing (Alternative)

If you don't want to use Match, you can install certificates manually:

1. Export your distribution certificate as `.p12` from Keychain Access
2. Base64-encode it: `base64 -i certificate.p12 | pbcopy`
3. Store as a GitHub Secret (`DISTRIBUTION_CERTIFICATE_BASE64`)
4. Store the export password as another secret (`CERTIFICATE_PASSWORD`)

In the workflow:

```yaml
- name: Install certificates
  env:
    CERTIFICATE_BASE64: ${{ secrets.DISTRIBUTION_CERTIFICATE_BASE64 }}
    CERTIFICATE_PASSWORD: ${{ secrets.CERTIFICATE_PASSWORD }}
  run: |
    security create-keychain -p "" build.keychain
    security default-keychain -s build.keychain
    security unlock-keychain -p "" build.keychain

    echo $CERTIFICATE_BASE64 | base64 --decode > certificate.p12
    security import certificate.p12 -k build.keychain \
      -P $CERTIFICATE_PASSWORD -T /usr/bin/codesign
    security set-key-partition-list -S apple-tool:,apple: \
      -s -k "" build.keychain

    rm certificate.p12
```

This works but is more fragile than Match — certificate renewal requires manual secret updates.

### GitHub Secrets Setup

For Match-based signing, add these secrets to your repo (Settings > Secrets and variables > Actions):

| Secret | Value |
|--------|-------|
| `MATCH_PASSWORD` | The passphrase you chose during `fastlane match init` |
| `MATCH_GIT_URL` | URL of your certificates repo |
| `APP_STORE_CONNECT_KEY_ID` | API key ID |
| `APP_STORE_CONNECT_ISSUER_ID` | API issuer ID |
| `APP_STORE_CONNECT_KEY_CONTENT` | Contents of the `.p8` file |

If your certificates repo is private (it should be), you'll also need a way for CI to access it. Options:
- **SSH key**: Add a deploy key to the certificates repo and the private key as a GitHub Secret
- **Personal access token**: Store as `MATCH_GIT_BASIC_AUTHORIZATION` (base64-encoded `username:token`)

### Exercise 4.1: Set Up Match

**Task:** Set up Fastlane Match for your project.

1. Create a private Git repository for certificates
2. Run `fastlane match init` and configure it
3. Generate distribution certificates with `fastlane match appstore`
4. Verify by building locally with manual signing (using the Match-generated profile)

<details>
<summary>Hint: If Match fails to generate certificates</summary>

Common issues:
- Your Apple ID doesn't have the right permissions. You need Admin or App Manager role.
- There are existing certificates that conflict. Match may suggest revoking them — be careful if others on your team use them.
- The certificates repo URL is wrong. Verify you can `git clone` it manually.
</details>

<details>
<summary>Hint: Testing Match locally</summary>

After running `fastlane match appstore`, verify the certificate is installed:
1. Open Keychain Access
2. Look for "Apple Distribution" certificate
3. In Xcode, switch to manual signing and select the Match-generated profile
4. Try building — it should succeed with the Match certificate
</details>

### Exercise 4.2: Create an API Key

**Task:** Generate an App Store Connect API key for CI authentication.

1. Go to App Store Connect > Users and Access > Integrations > App Store Connect API
2. Generate a key with "App Manager" role
3. Download the `.p8` file and save it securely
4. Note the Key ID and Issuer ID

<details>
<summary>Important: Save the .p8 file immediately</summary>

You can only download the `.p8` file once. If you lose it, you'll need to generate a new key. Save it somewhere secure (password manager, encrypted drive).
</details>

### Checkpoint 4

Before moving on, make sure you understand:
- [ ] Code signing in CI requires getting certificates onto the runner programmatically
- [ ] Fastlane Match stores encrypted certificates in a private Git repo
- [ ] `readonly: true` prevents CI from modifying certificates
- [ ] App Store Connect API keys replace Apple ID/password for CI authentication
- [ ] All secrets are stored in GitHub Settings > Secrets

---

## Section 5: Automated TestFlight Deployment

### The Complete Deploy Workflow

With signing solved, the deployment workflow ties everything together. Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to TestFlight

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: macos-15

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Select Xcode version
        run: sudo xcode-select -s /Applications/Xcode_16.2.app

      - name: Set up Ruby
        uses: ruby/setup-ruby@v1
        with:
          ruby-version: '3.2'
          bundler-cache: true

      - name: Deploy to TestFlight
        env:
          MATCH_PASSWORD: ${{ secrets.MATCH_PASSWORD }}
          MATCH_GIT_URL: ${{ secrets.MATCH_GIT_URL }}
          APP_STORE_CONNECT_KEY_ID: ${{ secrets.APP_STORE_CONNECT_KEY_ID }}
          APP_STORE_CONNECT_ISSUER_ID: ${{ secrets.APP_STORE_CONNECT_ISSUER_ID }}
          APP_STORE_CONNECT_KEY_CONTENT: ${{ secrets.APP_STORE_CONNECT_KEY_CONTENT }}
        run: bundle exec fastlane beta
```

When code is pushed to main:
1. GitHub spins up a macOS runner
2. Checks out the code and selects Xcode
3. Sets up Ruby and installs Fastlane from the Gemfile
4. Runs the `beta` lane — Match handles signing, Gym builds, Pilot uploads

### Build Number Management

The build number must increment with every upload. Hardcoding it means manual bumps before every merge. Automate it instead.

**Option A: Query App Store Connect for the latest build number:**

```ruby
lane :beta do
  api_key = app_store_connect_api_key(
    key_id: ENV["APP_STORE_CONNECT_KEY_ID"],
    issuer_id: ENV["APP_STORE_CONNECT_ISSUER_ID"],
    key_content: ENV["APP_STORE_CONNECT_KEY_CONTENT"]
  )

  # Get the latest build number from TestFlight and increment
  latest = latest_testflight_build_number(api_key: api_key)
  increment_build_number(build_number: latest + 1)

  match(type: "appstore", readonly: true, api_key: api_key)
  gym(
    project: "YourApp.xcodeproj",
    scheme: "YourApp",
    export_method: "app-store"
  )
  pilot(api_key: api_key, skip_waiting_for_build_processing: true)
end
```

This queries App Store Connect for the highest build number and uses the next one. No manual bumping needed.

**Option B: Use the Git commit count:**

```ruby
build_number = sh("git rev-list --count HEAD").strip
increment_build_number(build_number: build_number)
```

Simpler, but can collide if you rebase or force-push (the commit count can decrease).

### Changelog Generation

Auto-generate TestFlight "What to Test" notes from git commits:

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

This sets the "What to Test" field to your recent commit messages. Useful for internal testing; for external testers, you might want manually curated notes.

### The Complete Fastfile

Here's a complete Fastfile with all the pieces:

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

  desc "Deploy to TestFlight"
  lane :beta do
    api_key = app_store_connect_api_key(
      key_id: ENV["APP_STORE_CONNECT_KEY_ID"],
      issuer_id: ENV["APP_STORE_CONNECT_ISSUER_ID"],
      key_content: ENV["APP_STORE_CONNECT_KEY_CONTENT"]
    )

    latest = latest_testflight_build_number(api_key: api_key)
    increment_build_number(build_number: latest + 1)

    match(type: "appstore", readonly: true, api_key: api_key)

    gym(
      project: "YourApp.xcodeproj",
      scheme: "YourApp",
      export_method: "app-store"
    )

    changelog = changelog_from_git_commits(
      merge_commit_filtering: "exclude_merges"
    )

    pilot(
      api_key: api_key,
      changelog: changelog,
      skip_waiting_for_build_processing: true
    )
  end
end
```

### Exercise 5.1: Create the Deploy Workflow

**Task:** Create a GitHub Actions workflow that deploys to TestFlight on merge to main.

1. Create `.github/workflows/deploy.yml`
2. Configure it to trigger on pushes to main
3. Add all required secrets to your GitHub repository
4. Use the complete Fastfile from above (adapted to your project)

<details>
<summary>Hint: Verifying secrets are set correctly</summary>

In GitHub: Settings > Secrets and variables > Actions. You should see:
- `MATCH_PASSWORD`
- `MATCH_GIT_URL`
- `APP_STORE_CONNECT_KEY_ID`
- `APP_STORE_CONNECT_ISSUER_ID`
- `APP_STORE_CONNECT_KEY_CONTENT`

Secret values are hidden after creation — you can't read them back. If you're unsure about a value, delete and re-create the secret.
</details>

<details>
<summary>Hint: If the deploy fails on the first run</summary>

Common first-run failures:
- **Match can't access the certificates repo**: Add SSH key or token for repo access
- **API key authentication fails**: Verify all three values (key ID, issuer ID, key content)
- **Build number collision**: If you've uploaded manually before, the auto-incremented number might collide. Set it higher manually for the first CI run.
- **Xcode version mismatch**: Ensure the runner has the Xcode version you selected
</details>

### Checkpoint 5

Before moving on, make sure you understand:
- [ ] The deploy workflow triggers on push to main and runs the `beta` lane
- [ ] Build numbers are auto-incremented by querying App Store Connect
- [ ] All secrets (Match, API key) must be configured in GitHub
- [ ] `skip_waiting_for_build_processing: true` lets CI finish before Apple processes the build

---

## Section 6: Advanced Workflow Patterns

### Caching Dependencies

SPM package resolution and Fastlane gem installation add minutes to every build. Cache them:

```yaml
# In your workflow, before the build step:

- name: Cache SPM packages
  uses: actions/cache@v4
  with:
    path: |
      ~/Library/Developer/Xcode/DerivedData/**/SourcePackages
    key: spm-${{ hashFiles('**/Package.resolved') }}
    restore-keys: spm-

- name: Cache Ruby gems
  uses: actions/cache@v4
  with:
    path: vendor/bundle
    key: gems-${{ hashFiles('Gemfile.lock') }}
    restore-keys: gems-
```

The cache key includes a hash of the lock file. When dependencies change, the cache is invalidated and rebuilt. When they don't change, the cached packages are restored, typically saving 2-5 minutes per run.

**What's worth caching:**
- SPM packages (`Package.resolved`) — saves download time
- Ruby gems (`Gemfile.lock`) — saves Fastlane installation time
- CocoaPods (`Podfile.lock`) — saves pod installation time

**What's NOT worth caching:**
- DerivedData (Xcode's build cache) — Xcode often ignores cached DerivedData from a different environment, and the cache can be very large
- The Xcode toolchain — it's pre-installed on runners

### Separating Test and Deploy Jobs

Split your pipeline into jobs so tests must pass before deployment:

```yaml
name: CI/CD

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
      - name: Select Xcode
        run: sudo xcode-select -s /Applications/Xcode_16.2.app
      - name: Run tests
        run: |
          xcodebuild test \
            -project YourApp.xcodeproj \
            -scheme YourApp \
            -destination 'platform=iOS Simulator,name=iPhone 16,OS=18.2'

  deploy:
    needs: test
    runs-on: macos-15
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Select Xcode
        run: sudo xcode-select -s /Applications/Xcode_16.2.app
      - name: Set up Ruby
        uses: ruby/setup-ruby@v1
        with:
          ruby-version: '3.2'
          bundler-cache: true
      - name: Deploy
        env:
          MATCH_PASSWORD: ${{ secrets.MATCH_PASSWORD }}
          MATCH_GIT_URL: ${{ secrets.MATCH_GIT_URL }}
          APP_STORE_CONNECT_KEY_ID: ${{ secrets.APP_STORE_CONNECT_KEY_ID }}
          APP_STORE_CONNECT_ISSUER_ID: ${{ secrets.APP_STORE_CONNECT_ISSUER_ID }}
          APP_STORE_CONNECT_KEY_CONTENT: ${{ secrets.APP_STORE_CONNECT_KEY_CONTENT }}
        run: bundle exec fastlane beta
```

**Key points:**
- `needs: test` — deploy only runs after tests pass
- `if: github.ref == 'refs/heads/main'` — deploy only runs on main pushes, not on PRs
- PRs trigger only the test job; pushes to main trigger both

### Matrix Builds

Test on multiple iOS versions:

```yaml
test:
  runs-on: macos-15
  strategy:
    matrix:
      destination:
        - 'platform=iOS Simulator,name=iPhone 16,OS=18.2'
        - 'platform=iOS Simulator,name=iPhone 15,OS=17.5'
  steps:
    - uses: actions/checkout@v4
    - name: Select Xcode
      run: sudo xcode-select -s /Applications/Xcode_16.2.app
    - name: Test
      run: |
        xcodebuild test \
          -project YourApp.xcodeproj \
          -scheme YourApp \
          -destination '${{ matrix.destination }}'
```

Each matrix entry runs as a separate job in parallel. Be mindful of costs — two entries means two macOS runner charges.

### Failure Notifications

Get notified when builds fail:

```yaml
- name: Notify Slack on failure
  if: failure()
  uses: slackapi/slack-github-action@v2.0.0
  with:
    webhook: ${{ secrets.SLACK_WEBHOOK }}
    payload: |
      {"text": "iOS build failed on ${{ github.ref_name }}: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"}
```

`if: failure()` means this step only runs when a previous step failed. The Slack message includes a link to the failed run.

### Exercise 6.1: Add Caching

**Task:** Add dependency caching to your test workflow.

1. Add SPM package caching (if you use SPM)
2. Add Ruby gems caching (if you use Fastlane)
3. Push the changes and compare build times (first run fills the cache, second run uses it)

<details>
<summary>Hint: Cache placement matters</summary>

Place cache steps before the build step, but after checkout:
```yaml
steps:
  - uses: actions/checkout@v4       # 1. Get the code
  - name: Cache SPM                  # 2. Restore cached packages
    uses: actions/cache@v4
    with: ...
  - name: Build and test             # 3. Build (uses cache if available)
    run: xcodebuild test ...
```
</details>

### Checkpoint 6

Before moving on, make sure you understand:
- [ ] Caching SPM packages and Ruby gems saves build time
- [ ] `needs:` creates dependencies between jobs (test must pass before deploy)
- [ ] `if: github.ref == 'refs/heads/main'` limits a job to main-branch pushes only
- [ ] Matrix builds test across configurations but multiply runner costs

---

## Practice Project

### Project Description

Set up a complete CI/CD pipeline for your iOS project: tests on every PR, automatic TestFlight deployment on merge to main.

### Requirements

1. A `.github/workflows/test.yml` that runs tests on PRs
2. Fastlane initialized with `tests` and `beta` lanes
3. Fastlane Match configured for code signing
4. A `.github/workflows/deploy.yml` that deploys to TestFlight on merge to main
5. Dependency caching in both workflows
6. End-to-end verification: open a PR → tests run → merge → deploy triggers

### Getting Started

**Step 1: Get the test workflow running.**

Start with the simplest possible test workflow (Section 2). Push it to a branch, open a PR, and verify tests run. Don't add Fastlane or signing yet — just make xcodebuild work in CI.

**Step 2: Add Fastlane.**

Set up Fastlane locally (Section 3). Verify `fastlane tests` passes on your machine. Replace the raw xcodebuild step in your workflow with `fastlane tests`.

**Step 3: Set up signing.**

Follow Section 4 to configure Match and API keys. Add all secrets to GitHub. Verify locally that `fastlane match appstore --readonly` works.

**Step 4: Create the deploy workflow.**

Add `.github/workflows/deploy.yml` (Section 5). Merge a change to main and verify the deployment triggers and uploads to TestFlight.

**Step 5: Add caching and optimizations.**

Add dependency caching (Section 6). Verify faster build times on the second run.

### Hints and Tips

<details>
<summary>If tests pass locally but fail in CI</summary>

Common causes:
- **Different Xcode version**: Check which version CI uses (`sudo xcode-select -s ...`) vs your local version
- **Different simulator**: The simulator name/OS must match what's on the runner
- **Missing dependencies**: If you use CocoaPods, add `pod install` before the build step
- **Timeouts**: CI tests may run slower than local. Consider increasing test timeouts.
</details>

<details>
<summary>If Match fails in CI</summary>

Check in order:
1. Is `MATCH_GIT_URL` correct? Can the runner access the repo?
2. Is `MATCH_PASSWORD` set and correct?
3. Is the certificates repo private? If so, the runner needs credentials to clone it (SSH key or token)
4. Are the certificates still valid (not expired or revoked)?

Debug by adding `--verbose` to the Match command temporarily.
</details>

<details>
<summary>If you don't have access to macOS runners</summary>

You can still complete most of this project:
1. Write and validate all workflow YAML files locally
2. Test all Fastlane lanes locally (`fastlane tests`, `fastlane build`)
3. Set up Match and verify signing locally
4. The workflows will work when pushed to a repo with macOS runner access (GitHub Teams/Enterprise, or a public repo)
</details>

---

## Summary

### Key Takeaways

- **iOS CI/CD** requires macOS runners and solving code signing — the two unique challenges compared to web CI/CD.
- **GitHub Actions** provides the CI orchestration (when and where to run). Workflows are YAML files in `.github/workflows/`.
- **Fastlane** handles iOS-specific tasks: testing (`scan`), building (`gym`), uploading (`pilot`), and signing (`match`).
- **Fastlane Match** stores encrypted certificates in a private Git repo, solving the code signing problem in CI.
- **App Store Connect API keys** replace username/password authentication for CI uploads.
- **Caching** SPM packages and Ruby gems reduces build times. Conditional steps prevent unnecessary work.
- **The pipeline**: tests on PRs, deploy to TestFlight on merge to main.

### Skills You've Gained

You can now:
- Write GitHub Actions workflows for iOS projects
- Set up Fastlane with test, build, and deploy lanes
- Configure code signing for CI using Fastlane Match
- Automate TestFlight uploads with automatic build number management
- Optimize CI builds with caching and conditional steps
- Debug common CI/CD failures (signing, builds, uploads)

### Self-Assessment

Take a moment to reflect:
- Could you set up CI/CD for a new iOS project from scratch?
- Do you understand why Match exists and how it solves the signing problem?
- Can you read a failing CI log and identify the issue?
- Do you feel confident that merging to main will reliably deploy to TestFlight?

---

## Next Steps

### Continue Learning

**Build on this topic:**
- Add branch protection rules requiring CI to pass before merging
- Set up multiple environments (staging TestFlight group, production App Store)
- Explore Xcode Cloud as an Apple-native CI/CD alternative

**Apply what you've learned:**
- Consider the **My First iOS App** ascent to build a complete app using all the iOS routes

### Additional Resources

**Documentation:**
- GitHub Actions documentation (docs.github.com/en/actions)
- Fastlane documentation (docs.fastlane.tools)
- Fastlane Match documentation (docs.fastlane.tools/actions/match/)
- GitHub runner images (what's pre-installed): github.com/actions/runner-images

**Common References:**
- GitHub Actions macOS runner pricing
- Available Xcode versions on GitHub runners
- Fastlane action reference (all available actions)

---

## Quick Reference

### Essential Commands

```bash
# Fastlane
fastlane init                           # Initialize Fastlane
fastlane tests                          # Run test lane
fastlane beta                           # Run deploy lane
fastlane match init                     # Initialize Match
fastlane match appstore                 # Generate distribution certs
fastlane match appstore --readonly      # Download certs (CI mode)

# Xcode build (without Fastlane)
xcodebuild test -project App.xcodeproj -scheme App \
  -destination 'platform=iOS Simulator,name=iPhone 16,OS=18.2'
xcodebuild archive -project App.xcodeproj -scheme App \
  -archivePath build/App.xcarchive

# Simulators
xcrun simctl list devices available     # List available simulators
```

### GitHub Secrets Checklist

```
[ ] MATCH_PASSWORD         — Match encryption passphrase
[ ] MATCH_GIT_URL          — Certificates repo URL
[ ] APP_STORE_CONNECT_KEY_ID       — API key ID
[ ] APP_STORE_CONNECT_ISSUER_ID    — API issuer ID
[ ] APP_STORE_CONNECT_KEY_CONTENT  — .p8 key file contents
[ ] (Optional) SSH key or token for certificates repo access
```

### Workflow Triggers Quick Reference

```yaml
# PRs only
on:
  pull_request:
    branches: [main]

# Push to main only
on:
  push:
    branches: [main]

# Manual
on:
  workflow_dispatch:
```

### Key Terms

- **Runner**: The machine that executes your workflow (GitHub-hosted or self-hosted)
- **Lane**: A Fastlane workflow (sequence of actions)
- **Action (Fastlane)**: A single build step (scan, gym, pilot, match)
- **Action (GitHub)**: A reusable workflow step (actions/checkout, actions/cache)
- **Match**: Fastlane's code signing management — stores encrypted certs in a Git repo
- **Scan**: Fastlane action for running tests
- **Gym**: Fastlane action for building and archiving
- **Pilot**: Fastlane action for uploading to TestFlight
