---
title: App Store Publishing
route_map: /routes/app-store-publishing/map.md
paired_sherpa: /routes/app-store-publishing/sherpa.md
prerequisites:
  - A working iOS app
  - Apple Developer Program membership ($99/year)
  - Xcode basics
topics:
  - App Store Connect
  - TestFlight
  - Code Signing
  - Provisioning
  - App Review
---

# App Store Publishing

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/app-store-publishing/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/app-store-publishing/map.md` for the high-level overview.

## Overview

Getting an iOS app from "it runs on my phone" to "it's on the App Store" involves certificates, provisioning profiles, App Store Connect, TestFlight, and Apple's review process. If you've deployed web apps, imagine adding a certificate authority, a mandatory staging environment, and a human review gate between you and your users. That's iOS publishing.

This guide walks through every step of the pipeline. The process is mechanical once you understand it — the first time is the hardest.

## Learning Objectives

By the end of this guide, you will be able to:
- Explain Apple's code signing system and how certificates, app IDs, and provisioning profiles connect
- Configure an app identifier and capabilities in the Apple Developer Portal
- Create and manage an app listing in App Store Connect
- Archive builds in Xcode and upload them to App Store Connect
- Distribute beta builds through TestFlight (internal and external)
- Prepare app metadata, screenshots, and descriptions for submission
- Submit for review and handle common rejection reasons
- Manage post-launch: version updates, phased releases, analytics, and reviews

## Prerequisites

Before starting, you need:
- A working iOS app to publish (or a simple sample app — the process is the same)
- An Apple Developer Program membership ($99/year) — free accounts cannot distribute through TestFlight or the App Store
- Xcode installed and configured with your Apple ID (see xcode-essentials route)

## Setup

Ensure your Apple Developer account is active:

1. Open Xcode > Settings > Accounts
2. Verify your Apple ID is listed and shows your team
3. If not, click + to add your Apple ID

You'll also need access to two web portals:
- **Apple Developer Portal** (developer.apple.com) — manage certificates, app IDs, and provisioning profiles
- **App Store Connect** (appstoreconnect.apple.com) — manage your app's store listing, TestFlight, and analytics

Log in to both and verify you have access before continuing.

---

## Section 1: Code Signing Fundamentals

### Why Code Signing Exists

Every iOS app is cryptographically signed before it can run on a device. Code signing proves two things: the app was built by a known developer, and the app hasn't been tampered with since it was built.

In web terms, it's similar to HTTPS — a certificate verifies the server's identity and ensures data integrity. For iOS, the "server" is you (the developer) and the "data" is your app binary.

### The Three Pieces

Code signing relies on three interconnected components:

**1. Certificate — "Who built this?"**

A cryptographic identity proving you are a registered Apple developer. Certificates are stored in your Mac's Keychain. Two types:
- **Apple Development** — for running on test devices during development
- **Apple Distribution** — for App Store and TestFlight builds

When you enrolled in the Developer Program and added your account to Xcode, a development certificate was created for you automatically.

**2. App ID — "What app is this?"**

A unique identifier registered in the Apple Developer Portal that matches your app's bundle identifier in Xcode (e.g., `com.yourcompany.yourapp`). The app ID also declares which capabilities your app uses (push notifications, iCloud, HealthKit, etc.).

**3. Provisioning Profile — "Who can run this, and where?"**

A file that ties everything together: a certificate (who), an app ID (what), and — for development profiles — a list of specific devices (where). The profile authorizes your signed app to run in a specific context.

```
Certificate (who) ──┐
                     ├──→ Provisioning Profile ──→ Signed App
App ID (what) ──────┘          │
                         Device List (where)
                         (development only)
```

For distribution profiles, there's no device list — the App Store handles device authorization.

### Development vs Distribution

| | Development | Distribution |
|---|---|---|
| Purpose | Run on your test devices | App Store / TestFlight |
| Certificate | Apple Development | Apple Distribution |
| Devices in profile | Yes (specific UDIDs) | No |
| Who creates it | Xcode (automatic) | Xcode (automatic) or you (manual) |

During development, Xcode's automatic signing handles everything — you may not have noticed any of this machinery. Distribution is where you need to understand it.

### Automatic vs Manual Signing

Xcode offers two approaches:

**Automatic signing** (recommended): Xcode manages certificates, app IDs, and provisioning profiles for you. In your target's Signing & Capabilities tab, check "Automatically manage signing" and select your team. Xcode creates whatever is needed.

**Manual signing**: You create certificates and profiles yourself in the Developer Portal and select them explicitly in Xcode. This is needed for CI/CD pipelines (see the ios-ci-cd-with-github-actions route) or complex team setups.

**Recommendation**: Use automatic signing until you have a specific reason not to. It handles the common case well.

### The Keychain

Certificates live in your Mac's Keychain (open Keychain Access from Spotlight). If you get a new Mac, you'll need to either:
- Let Xcode re-create your certificates (easiest — just add your Apple ID in Xcode > Settings > Accounts)
- Export your certificates from the old Mac and import on the new one (only needed if you use manual signing and can't regenerate)

### Exercise 1.1: Verify Your Signing Setup

**Task:** Check your current code signing configuration for your app.

1. Open your project in Xcode
2. Select your app target
3. Go to Signing & Capabilities
4. Note the team, bundle identifier, and whether automatic signing is enabled

<details>
<summary>What you should see</summary>

- **Team**: Your developer account or team name
- **Bundle Identifier**: Something like `com.yourname.yourapp`
- **Signing Certificate**: "Apple Development" (for debug) — shown automatically
- **Provisioning Profile**: "Xcode Managed Profile" if using automatic signing
- No red error banners

If you see errors, the most common fixes are:
- Select the correct team from the dropdown
- Ensure your Apple ID is added in Xcode > Settings > Accounts
- If "Automatically manage signing" is checked, try unchecking and re-checking it
</details>

### Exercise 1.2: Find Your Certificates

**Task:** Open Keychain Access (Spotlight > "Keychain Access") and find your development certificate.

1. In Keychain Access, select "login" keychain on the left
2. Select "My Certificates" category
3. Look for "Apple Development: Your Name" or "Apple Development: your@email.com"

<details>
<summary>What you should see</summary>

You should see at least one "Apple Development" certificate with your name or email. If you expand it (click the triangle), you'll see a private key underneath — this is the key pair that makes the certificate work.

If you don't see any certificates, go back to Xcode > Settings > Accounts, select your team, and click "Manage Certificates" > + to create one.
</details>

### Checkpoint 1

Before moving on, make sure you understand:
- [ ] Code signing proves who built the app and that it hasn't been modified
- [ ] Three pieces: certificate (who), app ID (what), provisioning profile (ties them together)
- [ ] Development signing is for test devices; distribution signing is for App Store/TestFlight
- [ ] Automatic signing in Xcode handles most of this for you

---

## Section 2: App Store Connect Setup

### Creating an App Record

Before you can upload a build, you need an app record in App Store Connect. This is the listing that represents your app on the App Store.

1. Go to [appstoreconnect.apple.com](https://appstoreconnect.apple.com)
2. Click My Apps > + > New App
3. Fill in the required fields:

| Field | What to enter |
|-------|--------------|
| **Platform** | iOS |
| **Name** | Your app's display name on the App Store (up to 30 characters, must be unique) |
| **Primary Language** | The language of your default metadata |
| **Bundle ID** | Select from the dropdown — must match your Xcode project |
| **SKU** | An internal identifier (not user-visible). Your bundle ID works fine. |

The **bundle ID** connection is critical: the App Store Connect record and your Xcode project must use the same bundle identifier. If they don't match, uploads will fail.

### App Information

After creating the record, fill in the App Information tab:

- **Subtitle** (optional, up to 30 characters): A brief tagline shown under the app name
- **Category**: Primary and optional secondary category (Games, Productivity, Utilities, etc.)
- **Content Rating**: Answer Apple's questionnaire about content (violence, language, etc.) to get an age rating

### Pricing and Availability

- **Price**: Free or select a price tier. Apple takes 30% (15% for small businesses under $1M/year revenue).
- **Availability**: Which countries and regions your app is available in. Default is all territories.

### App Privacy Details

Apple requires privacy "nutrition labels" for every app. You'll need to declare:

1. **What data you collect** — Contact info, location, browsing history, identifiers, usage data, etc.
2. **How it's linked to identity** — Is the data associated with the user's account?
3. **Is it used for tracking?** — Do you share data with third parties for advertising?

If your app uses any third-party SDKs (analytics, crash reporting, ads), include their data collection too. Check each SDK's privacy documentation.

Even if your app "doesn't collect data," consider:
- Do you use any analytics (Firebase, Mixpanel)? → Usage data
- Do you use crash reporting (Crashlytics, Sentry)? → Diagnostics
- Does the user create an account? → Contact info, identifiers
- Do you use Apple's frameworks that collect device info? → Identifiers

### Exercise 2.1: Create an App Record

**Task:** Create an App Store Connect record for your app (or a test app).

<details>
<summary>Hint: If you don't have a bundle ID in the dropdown</summary>

Your app's bundle ID must be registered in the Developer Portal first. With automatic signing in Xcode, this happens when you build. If it's not showing up:
1. Open your project in Xcode
2. Build it once (Cmd+B) with automatic signing enabled
3. Go back to App Store Connect and refresh — the bundle ID should appear
</details>

<details>
<summary>What you should have after this exercise</summary>

An app record in App Store Connect with:
- A name (can be changed later, but must remain unique)
- Your bundle ID selected
- Platform set to iOS
- Status: "Prepare for Submission"
</details>

### Checkpoint 2

Before moving on, make sure you understand:
- [ ] App Store Connect is where you manage your app's listing, builds, and testers
- [ ] The bundle ID must match between Xcode and App Store Connect
- [ ] Privacy disclosures are required and must include third-party SDK data collection
- [ ] You can change most metadata after creation, but the bundle ID is permanent

---

## Section 3: Building for Distribution

### Archive Builds

An archive is a release-optimized, distribution-signed build of your app. It's different from the debug build you create when you press Run:

| | Debug Build (Cmd+R) | Archive (Product > Archive) |
|---|---|---|
| Optimization | Minimal (fast compilation) | Full (fast execution) |
| Signing | Development certificate | Distribution certificate |
| Destination | Simulator or connected device | App Store Connect |
| Debug symbols | Included in binary | Uploaded separately (dSYM) |

### Creating an Archive

1. **Select the build destination**: Choose "Any iOS Device (arm64)" or a connected physical device. Archives don't work with the simulator selected.

2. **Set version and build numbers** in your target's General tab:
   - **Version** (Marketing Version): What users see — `1.0.0`, `1.1`, `2.0`. Follow semantic versioning.
   - **Build** (Current Project Version): Must increment with each upload to App Store Connect. Use integers: `1`, `2`, `3`.

3. **Archive**: Product > Archive (menu bar). Xcode compiles a release build and signs it with your distribution identity.

4. **On success**: The Organizer window opens (Window > Organizer) showing your archive with its date, version, and build number.

### Version and Build Numbers

Two numbers, two purposes:

- **Version** (`CFBundleShortVersionString`): The marketing version visible to users. Change this when releasing a user-visible update. Example: `1.0.0` → `1.1.0` for a feature update, `1.0.0` → `1.0.1` for a bug fix.

- **Build** (`CFBundleVersion`): An internal number that must increase with every upload. If you upload version 1.0.0 build 1, find a bug, and fix it before releasing, you upload version 1.0.0 build 2. Users never see this number.

In web terms: the version is like your npm package version. The build number is like a CI build counter — it always goes up.

### Uploading to App Store Connect

From the Organizer:

1. Select your archive
2. Click "Distribute App"
3. Select distribution method: **App Store Connect**
4. Choose **Upload** (not Export)
5. Xcode validates the build — checking signing, entitlements, and basic requirements
6. If validation passes, the upload begins

After upload, Apple runs additional processing on their servers:
- Checking for private API usage
- Validating the binary structure
- Generating app thinning variants (device-specific binaries)

This processing takes a few minutes to an hour. You'll receive an email if there are issues. Once processing completes, the build appears in App Store Connect.

### Common Validation Errors

**"No accounts with App Store Connect access"**: Your Apple ID doesn't have the right role. In App Store Connect > Users and Access, verify your user has App Manager, Developer, or Admin role.

**"Invalid bundle identifier"**: The bundle ID in your build doesn't match any app in App Store Connect. Create the app record first (Section 2).

**"Missing compliance information"**: Your app uses encryption. Most apps that only use HTTPS qualify for an exemption. In your Info.plist, add `ITSAppUsesNonExemptEncryption` set to `NO` if you only use HTTPS.

**"Provisioning profile doesn't match"**: Your archive was signed with a profile that doesn't match the app's bundle ID or distribution certificate. Toggle automatic signing off and on to regenerate.

### Exercise 3.1: Create and Upload an Archive

**Task:** Archive your app and upload it to App Store Connect.

<details>
<summary>Hint 1: Build destination</summary>

Make sure you have "Any iOS Device (arm64)" selected as the build destination, not a simulator. The Archive option is greyed out when a simulator is selected.
</details>

<details>
<summary>Hint 2: If the archive fails</summary>

Archive uses release build settings, which are stricter than debug. Common issues:
- Compiler warnings treated as errors in release configuration
- Missing assets or resources referenced in release mode
- Code that relies on `DEBUG` preprocessor flags

Check the build log (View > Navigators > Report Navigator) for specific errors.
</details>

<details>
<summary>Hint 3: If upload fails</summary>

Read the error message carefully — it usually tells you exactly what's wrong:
- Signing issues → Check Signing & Capabilities
- Missing icons → Add all required app icon sizes in Assets.xcassets
- Invalid binary → Check your deployment target and supported architectures
</details>

### Checkpoint 3

Before moving on, make sure you understand:
- [ ] Archives are release-optimized, distribution-signed builds
- [ ] The build destination must be a device (not simulator) to archive
- [ ] Version is the marketing number; build number must increment with every upload
- [ ] After upload, Apple processes the build — check email for issues

---

## Section 4: TestFlight

### What is TestFlight?

TestFlight is Apple's beta testing platform. After you upload a build to App Store Connect, you can distribute it to testers through the TestFlight app. There are two kinds of testers.

### Internal vs External Testing

| | Internal | External |
|---|---|---|
| Who | App Store Connect team members | Anyone with an invite |
| Max testers | 100 | 10,000 |
| Review required | No — instant access | Yes (first build only) |
| Auto-distribution | Optional per tester | Manual |
| Build expiry | 90 days | 90 days |

### Setting Up Internal Testing

Internal testers are members of your App Store Connect team — developers, marketers, QA engineers. They get builds instantly, no review needed.

1. Go to App Store Connect > Your App > TestFlight
2. Under Internal Testing, click the + button
3. Create a group (e.g., "Development Team")
4. Add testers — they must have App Store Connect accounts on your team
5. Enable "Automatic Distribution" if you want every uploaded build sent automatically

Testers receive an email invitation. They install the TestFlight app from the App Store, accept the invitation, and install your beta build.

### Setting Up External Testing

External testers can be anyone — clients, beta users, friends. They don't need to be part of your team.

1. In TestFlight, go to External Testing
2. Click + to create a test group (e.g., "Beta Testers")
3. Add testers by email, or generate a public invite link
4. Select a build to distribute to the group
5. Fill in "What to Test" — instructions for testers on what to focus on
6. Submit the build for TestFlight review

The first build sent to an external group requires a brief review by Apple (usually under 24 hours). Subsequent builds to the same group typically auto-approve.

### Collecting Feedback

Testers can submit feedback directly from the TestFlight app:
- **Screenshots** with annotations (draw on the screenshot to highlight issues)
- **Text feedback** describing the problem
- **Crash reports** sent automatically when the app crashes

View all feedback in App Store Connect > TestFlight > Feedback. Crash details appear in Xcode's Organizer (Window > Organizer > Crashes) with symbolicated stack traces.

### Managing Builds

- Builds expire **90 days** after upload. After that, testers can no longer use them.
- You can have multiple builds available simultaneously (e.g., different versions for different test groups).
- You can disable a build to prevent testers from installing it without removing the group.
- "What to Test" can be updated per build — use it to guide testers toward areas you changed.

### Exercise 4.1: Set Up Internal Testing

**Task:** Set up internal testing and distribute a build to yourself.

1. Upload a build if you haven't already (Section 3)
2. In App Store Connect > TestFlight, create an internal test group
3. Add yourself as a tester
4. Install the TestFlight app on your iPhone
5. Accept the invitation and install the build

<details>
<summary>If the build doesn't appear in TestFlight</summary>

After uploading, the build needs processing time (a few minutes to an hour). Check:
1. App Store Connect > TestFlight > Builds — is your build listed?
2. If it shows "Processing," wait. If it shows a warning icon, click it for details.
3. Check your email — Apple sends notifications if processing finds issues.

If the build is listed but testers aren't seeing it:
- Did you enable "Automatic Distribution" on the test group?
- If not, manually distribute the build by selecting it and adding it to the group.
</details>

### Checkpoint 4

Before moving on, make sure you understand:
- [ ] Internal testers get builds instantly; external testers require a brief review (first build)
- [ ] TestFlight builds expire after 90 days
- [ ] Testers submit feedback (screenshots, text, crash reports) through the TestFlight app
- [ ] You manage builds, groups, and feedback in App Store Connect

---

## Section 5: App Store Submission

### Preparing Your Metadata

Before submitting, you need complete metadata — this is what users see on your App Store page. Every field matters for discoverability and conversion.

**Description** (up to 4,000 characters):
- First sentence is critical — it's visible in search results before the user taps
- Focus on what the app does and why it's useful
- Use short paragraphs and whitespace for readability
- Don't keyword-stuff — Apple penalizes it

**Keywords** (up to 100 characters, comma-separated):
- These drive App Store search. Choose carefully.
- Don't repeat words from your app name — they're already indexed
- Don't include generic words like "app" or "free"
- Use the full 100 characters — separate with commas, no spaces after commas
- Example: `task,todo,productivity,planner,organize,checklist,reminder`

**Screenshots**:
- Required for each device size you support
- At minimum, provide screenshots for:
  - 6.7" display (iPhone 15 Pro Max / iPhone 16 Plus)
  - 6.5" display (iPhone 11 Pro Max / iPhone XS Max)
- Up to 10 screenshots per device size
- First 3 screenshots are the most important — they show in search results
- You can use actual screenshots or designed marketing images that include screenshots

**App Preview** (optional, up to 30 seconds):
- A video showing the app in action
- Autoplays in search results on Wi-Fi — a strong differentiator
- Must show actual app functionality (not a cinematic ad)

**Other Required Fields**:
- **Support URL**: A webpage where users can get help
- **Privacy Policy URL**: Required for all apps
- **What's New** (for updates): Release notes. Be specific — "Bug fixes and improvements" tells users nothing

### App Review Guidelines

Apple reviews every app submission. Understanding the guidelines prevents rejection delays.

**Most common rejection reasons:**

**Guideline 2.1 — App Completeness:**
Your app must be fully functional. No placeholder content, no "coming soon" features, no broken links. The build you submit should be the build you're ready to ship.

If your app requires a login, provide a demo account with credentials in the review notes. If it requires special hardware or location, explain how to test it.

**Guideline 2.3 — Accurate Metadata:**
Screenshots must reflect actual app functionality. Description must be truthful. Don't claim features you haven't built.

**Guideline 3.1.1 — In-App Purchase:**
If you sell digital content or services consumed within the app, you must use Apple's in-app purchase system (Apple takes 30%/15%). Physical goods and real-world services (Uber rides, restaurant delivery) can use external payment.

This is Apple's most strictly enforced guideline. If your app charges for digital content without IAP, it will be rejected.

**Guideline 4.0 — Design:**
The app must provide meaningful functionality beyond what a website could do. Apps that simply wrap a web view around a website are rejected.

**Guideline 5.1.1 — Data Collection and Privacy:**
You must have a privacy policy. Your privacy disclosures must accurately reflect your data collection. Mismatch between your declarations and actual behavior leads to rejection.

### Submitting for Review

1. In App Store Connect, go to the App Store tab
2. Select the version you're preparing
3. Fill in all metadata fields (description, keywords, screenshots, etc.)
4. Select a build from your uploaded builds
5. Fill in App Review Information:
   - **Contact Information**: So reviewers can reach you with questions
   - **Notes**: Explain anything the reviewer needs to know (demo credentials, special setup)
   - **Demo Account**: Username and password if login is required
6. Click "Submit for Review"

Review typically takes 24-48 hours. You'll receive an email with the result.

### Handling Rejections

Rejections aren't failures — they're feedback. Common responses:

**If the rejection is valid** (you have a real bug or guideline violation):
1. Fix the issue
2. Increment the build number
3. Archive and upload the new build
4. In App Store Connect, select the new build for the same version
5. Respond in the Resolution Center explaining what you fixed
6. Resubmit

**If you believe the rejection is incorrect:**
1. Respond in the Resolution Center with a clear, factual explanation
2. If the reviewer misunderstood your app, explain what it does and how to test it
3. If you still disagree after the response, you can appeal to the App Review Board

**Tips for avoiding rejection:**
- Test thoroughly on a physical device before submitting
- Provide clear review notes and demo credentials
- Don't ship placeholder content or "under construction" sections
- Review the guidelines for your app's category before submission

### Exercise 5.1: Prepare Your Metadata

**Task:** Write complete App Store metadata for your app.

1. Write a description (aim for 500-1,000 characters for a first version)
2. List your keywords (use all 100 characters)
3. Take or design at least 3 screenshots for one device size
4. Write a support URL page (even a simple one-page site)
5. Write or link to a privacy policy

<details>
<summary>Hint: Writing an effective description</summary>

Structure:
1. First sentence: What the app does in one line
2. Key features (3-5 bullet points or short paragraphs)
3. Why users should choose your app

Don't:
- Start with "Welcome to..." or "This app..."
- Use ALL CAPS or excessive punctuation
- List every feature — focus on the ones that matter
</details>

<details>
<summary>Hint: Screenshot strategies</summary>

Two approaches:
1. **Raw screenshots**: Actual screenshots from the app. Quick but less polished.
2. **Designed screenshots**: Screenshots placed inside device frames with marketing text. More work, but much higher conversion.

Tools for designed screenshots: Figma, Sketch, or dedicated tools like AppMockUp or Screenshots Pro.

For the first submission, raw screenshots are fine — you can polish them later.
</details>

### Checkpoint 5

Before moving on, make sure you understand:
- [ ] App Store metadata includes description, keywords, screenshots, and required URLs
- [ ] The most common rejection reasons: bugs, inaccurate metadata, missing IAP, privacy issues
- [ ] Always provide demo credentials and review notes for apps that require login
- [ ] Rejections are responded to in the Resolution Center — fix, rebuild, resubmit

---

## Section 6: Post-Launch Management

### Phased Releases

When releasing an update, you can choose phased release — a gradual rollout over 7 days:

| Day | % of Users |
|-----|-----------|
| 1 | 1% |
| 2 | 2% |
| 3 | 5% |
| 4 | 10% |
| 5 | 20% |
| 6 | 50% |
| 7 | 100% |

This is similar to a gradual rollout in web deployment. If you discover a critical bug after release, you can **pause** the phased release before all users get the update. Users who manually check for updates can still get it early.

To enable: when submitting a version, select "Release this version over a 7-day period" instead of "Manually release" or "Immediately after approval."

**When to use phased releases:** Any non-trivial update. The ability to pause gives you a safety net.

**Important limitation:** You can pause a phased release, but you cannot roll back to a previous version. If a critical bug gets through, pause the release and submit a fix as quickly as possible.

### Releasing Updates

The update workflow:

1. **Increment version** in Xcode (e.g., `1.0.0` → `1.1.0`). Increment build number too.
2. **Archive and upload** the new build (same as Section 3).
3. In App Store Connect, click **+ Version** to create a new version.
4. Fill in **What's New** — your release notes. Be specific about what changed.
5. **Select the new build**.
6. **Submit for review** — updates go through the same review process.

You can have multiple versions in different states simultaneously:
- One live on the App Store
- One in review
- One in preparation

### App Analytics

App Store Connect provides analytics at Analytics > Overview:

- **Impressions**: How many times your app appeared in search results or featured sections
- **Product Page Views**: How many users viewed your app's page
- **Downloads**: First-time downloads and re-downloads
- **Sessions**: How often the app is opened
- **Crashes**: Crash counts (use Xcode Organizer for detailed stack traces)
- **Retention**: Percentage of users who return after 1, 7, and 28 days

For detailed crash analysis, use Xcode's Organizer (Window > Organizer > Crashes). App Store Connect shows aggregate crash counts; Organizer shows symbolicated crash logs with stack traces pointing to specific lines in your code.

### Responding to User Reviews

In App Store Connect > Ratings and Reviews, you can respond to user reviews:

- Responses are **public** — every App Store visitor can see them
- Be professional and helpful
- If a user reports a bug you've fixed, say so: "Thanks for reporting this. We fixed this in version 1.1 — please update and let us know if the issue persists."
- Don't be defensive or dismissive
- You can update your response, but only the latest version is shown

### Exercise 6.1: Review Monitoring Plan

**Task:** Create a post-launch checklist for the first week after release.

<details>
<summary>Suggested checklist</summary>

**Daily for the first week:**
- [ ] Check crash reports in Xcode Organizer — any new crash patterns?
- [ ] Review user feedback in App Store Connect — any blocking issues?
- [ ] Monitor phased release progress — any reason to pause?
- [ ] Check analytics — are downloads and retention in expected range?

**If issues arise:**
- [ ] Pause phased release if crash rate is high
- [ ] Respond to user reviews reporting bugs
- [ ] Start working on a fix immediately — don't wait to gather all reports
- [ ] Submit a patch release (e.g., 1.0.1) as soon as the fix is ready

**After the first week:**
- [ ] Review analytics trends
- [ ] Respond to any remaining user reviews
- [ ] Plan the next update based on feedback
</details>

### Checkpoint 6

Before moving on, make sure you understand:
- [ ] Phased releases distribute updates gradually and can be paused (but not rolled back)
- [ ] Updates go through the same archive → upload → review cycle
- [ ] App Store Connect provides analytics; Xcode Organizer provides crash details
- [ ] Review responses are public — be professional and helpful

---

## Practice Project

### Project Description

Walk through the complete publishing pipeline end-to-end. Even if you're not ready to ship to the App Store, you'll configure everything through TestFlight distribution — which exercises 90% of the publishing process.

### Requirements

Using your app (or a simple sample app):

1. Verify your code signing setup in Xcode
2. Create an App Store Connect record with complete app information
3. Configure privacy disclosures
4. Set version to `1.0.0` and build to `1`
5. Archive the app
6. Upload to App Store Connect
7. Set up an internal TestFlight group and distribute the build
8. Install the build on your device via TestFlight

### Getting Started

**Step 1: Verify signing**

Open Xcode > your target > Signing & Capabilities. Ensure:
- Automatic signing is enabled
- Your team is selected
- No signing errors

**Step 2: Set version and build**

In your target's General tab:
- Marketing Version: `1.0.0`
- Current Project Version: `1`

**Step 3: Create the App Store Connect record**

Follow Section 2. Use your app's actual bundle ID.

**Step 4: Archive and upload**

Follow Section 3. Remember: select a device target (not simulator) before archiving.

**Step 5: Set up TestFlight**

Follow Section 4. Create an internal group, add yourself, and install via TestFlight app.

### Hints and Tips

<details>
<summary>If you're using a sample/test app</summary>

You can use any simple app for this exercise. A single-screen "Hello World" app works fine — the publishing process is identical regardless of app complexity. The goal is to practice the pipeline, not the app.
</details>

<details>
<summary>If archive fails with signing errors</summary>

1. Open Xcode > Settings > Accounts
2. Select your team and click "Download Manual Profiles" (downloads updated profiles)
3. In Signing & Capabilities, try toggling automatic signing off, then on
4. If still failing, try Product > Clean Build Folder (Cmd+Shift+K), then archive again
</details>

<details>
<summary>If upload succeeds but build doesn't appear in TestFlight</summary>

Processing takes time. Check:
1. Your email — Apple sends notifications about processing issues
2. App Store Connect > Activity — shows build processing status
3. Wait at least 30 minutes before troubleshooting further
</details>

---

## Summary

### Key Takeaways

- **Code signing** uses three interconnected pieces (certificate, app ID, provisioning profile) to verify app authenticity. Automatic signing handles most of this for you.
- **App Store Connect** is the central hub for managing your app's presence — listings, builds, TestFlight, analytics, and reviews.
- **Archives** are release-optimized, distribution-signed builds created through Product > Archive in Xcode.
- **TestFlight** provides beta distribution with internal (instant) and external (reviewed) tester groups. Builds expire after 90 days.
- **App Review** checks for completeness, accuracy, and guideline compliance. Most reviews complete within 24-48 hours.
- **Post-launch** involves monitoring crashes and analytics, responding to reviews, and using phased releases for safe updates.

### Skills You've Gained

You can now:
- Explain how iOS code signing works and troubleshoot common signing errors
- Set up a complete App Store Connect listing with proper privacy disclosures
- Archive, upload, and distribute builds through TestFlight
- Prepare metadata and submit for App Store review
- Handle rejections and manage post-launch updates

### Self-Assessment

Take a moment to reflect:
- Could you walk a teammate through the publishing process?
- Do you know what to check when an upload fails?
- Can you list the top App Store rejection reasons?
- Do you feel confident enough to publish your own app?

---

## Next Steps

### Continue Learning

**Build on this topic:**
- Practice by publishing a simple utility app end-to-end — the first time is the learning experience

**Explore related routes:**
- [iOS CI/CD with GitHub Actions](/routes/ios-ci-cd-with-github-actions/map.md) — Automate the build and upload pipeline
- [iOS App Patterns](/routes/ios-app-patterns/map.md) — Structure your app for maintainability before shipping

### Additional Resources

**Documentation:**
- Apple's App Store Review Guidelines (developer.apple.com/app-store/review/guidelines/)
- App Store Connect Help (developer.apple.com/help/app-store-connect/)
- Human Interface Guidelines (developer.apple.com/design/human-interface-guidelines/)

**Common References:**
- App Store screenshot sizes and requirements
- Export compliance and encryption documentation
- Privacy manifest documentation for third-party SDKs

---

## Quick Reference

### Publishing Checklist

```
[ ] Code signing configured (automatic or manual)
[ ] Bundle ID matches between Xcode and App Store Connect
[ ] App record created in App Store Connect
[ ] Privacy disclosures completed
[ ] Version and build numbers set
[ ] Archive created successfully
[ ] Build uploaded and processed
[ ] TestFlight testing completed
[ ] Metadata complete (description, keywords, screenshots)
[ ] Privacy policy URL set
[ ] Support URL set
[ ] Review notes and demo credentials provided
[ ] Submitted for review
```

### Key Terms

- **Certificate**: Cryptographic identity proving you're a registered developer
- **App ID**: Unique identifier matching your bundle identifier
- **Provisioning Profile**: Ties certificate + app ID (+ devices for development)
- **Archive**: Release-optimized, distribution-signed build
- **Version**: Marketing number visible to users (e.g., 1.0.0)
- **Build Number**: Internal number that must increment with each upload
- **TestFlight**: Apple's beta testing platform
- **Phased Release**: Gradual rollout of an update over 7 days
- **Resolution Center**: Where you communicate with App Review about rejections
