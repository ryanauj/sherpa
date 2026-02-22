---
title: App Store Publishing
route_map: /routes/app-store-publishing/map.md
paired_guide: /routes/app-store-publishing/guide.md
topics:
  - App Store Connect
  - TestFlight
  - Code Signing
  - Provisioning
  - App Review
---

# App Store Publishing - Sherpa (AI Teaching Guide)

**Purpose**: This sherpa guide helps AI assistants teach iOS app publishing to developers who are comfortable building apps but haven't navigated Apple's distribution pipeline before. It covers code signing, App Store Connect, TestFlight, submission, and post-launch management — with comparisons to web deployment where helpful.

**Route Map**: See `/routes/app-store-publishing/map.md` for the high-level overview of this route.
**Paired Guide**: The human-focused content is at `/routes/app-store-publishing/guide.md`.

---

## Teaching Overview

### Learning Objectives
By the end of this session, the learner should be able to:
- Explain how Apple's code signing system works: certificates, app IDs, provisioning profiles
- Configure an app identifier and capabilities in the Apple Developer Portal
- Create and manage an app listing in App Store Connect
- Archive a build in Xcode and upload it to App Store Connect
- Distribute beta builds through TestFlight to internal and external testers
- Prepare complete app metadata (description, keywords, screenshots) for submission
- Submit an app for review and handle common rejection scenarios
- Manage post-launch concerns: version updates, phased releases, and responding to reviews

### Prior Sessions
Before starting, check `.sessions/index.md` and `.sessions/app-store-publishing/` for prior session history. If the learner has completed previous sessions on this route, review the summaries to understand what they've covered and pick up where they left off.

### Prerequisites to Verify
Before starting, verify the learner has:
- A working iOS app they want to publish (or are willing to use a sample app)
- An Apple Developer Program membership ($99/year) — free accounts cannot publish
- Xcode installed and configured with their Apple ID
- Basic familiarity with Xcode (building, running on a device)

**If prerequisites are missing**: If they don't have a Developer account, they can still learn the concepts but won't be able to follow along hands-on. If Xcode basics are weak, suggest the xcode-essentials route first.

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
- Good for checking understanding of code signing relationships (what connects to what)
- Example: "What does a provisioning profile contain? A) Only the certificate B) Only the app ID C) The certificate, app ID, and device list D) The app binary and certificate"

**Scenario Questions:**
- Present a publishing problem (rejection reason, signing error) and ask how to resolve it
- Example: "Your build uploads successfully but fails processing with 'Invalid Bundle' — what would you check?"

**Process Questions:**
- Ask the learner to describe a publishing workflow in order
- These assess whether they understand the end-to-end flow, not just individual steps

**Mixed Approach (Recommended):**
- Use multiple choice for code signing concepts
- Use scenario questions for troubleshooting
- Use process questions for the overall submission flow

---

## Web-to-iOS Publishing Reference

Use this table for learners with web development experience:

| Web Concept | iOS/App Store Equivalent |
|-------------|--------------------------|
| `npm publish` / deploy to production | Archive + Upload to App Store Connect |
| Staging environment | TestFlight |
| SSL certificate | Code signing certificate |
| Domain registration | App ID / Bundle Identifier |
| CI/CD deploy | Xcode Cloud / GitHub Actions + Fastlane |
| Package registry (npm) | App Store Connect |
| README / docs | App Store listing (description, screenshots) |
| Semantic versioning | Version (marketing) + Build number |
| Feature flags / gradual rollout | Phased release |
| Lighthouse / web vitals | App Store review |
| `robots.txt` / SEO | App Store Optimization (keywords, screenshots) |

---

## Teaching Flow

### Introduction

**What to Cover:**
- Publishing an iOS app involves more steps than deploying a web app. Apple controls the distribution pipeline and reviews every submission.
- Code signing is the piece that confuses everyone the first time. We'll demystify it.
- The process is: sign your app → upload to App Store Connect → test via TestFlight → submit for review → release.
- It's mechanical once you understand it. The first time is the hardest; after that, it's routine.

**Opening Questions to Assess Level:**
1. "Have you ever published an app to the App Store or any other app store?"
2. "Have you run your app on a physical device? Did Xcode handle signing automatically?"
3. "Are you familiar with what code signing is, even conceptually?"

**Adapt based on responses:**
- If they've published before: Focus on areas where they had trouble. Skip the conceptual basics, go straight to process.
- If they've only used automatic signing: They have certificates and profiles but may not understand what they are. Start with the conceptual model.
- If completely new: Take time with the mental model. Code signing is the foundation everything else builds on.
- If they have a web background: Draw parallels to SSL certificates, deployment pipelines, and staging environments.

**Opening framing:**
"Deploying a web app, you push to a server and it's live. iOS is different — Apple sits between you and your users. Every app is cryptographically signed to prove it came from you, reviewed by Apple to meet their guidelines, and distributed exclusively through the App Store (for most apps). The upside is trust and security for users. The downside is process. Let's walk through that process so it becomes second nature."

---

### Section 1: Code Signing Fundamentals

**Core Concept to Teach:**
Code signing is Apple's system for proving that an app comes from a known, trusted developer and hasn't been tampered with. It relies on three interconnected pieces: certificates, app IDs, and provisioning profiles.

**How to Explain:**
1. Start with why: Apple needs to verify who built the app and that it hasn't been modified
2. Explain the three pieces and how they connect
3. Show where these live (Keychain, Developer Portal, Xcode)
4. Clarify the difference between development and distribution signing

**The Three Pieces:**

"Think of it like a notarized document:"
- **Certificate** = "Who are you?" — A cryptographic identity proving you (or your team) are a registered developer. Stored in your Mac's Keychain. There are two types: Development (for running on test devices) and Distribution (for App Store and TestFlight).
- **App ID** = "What app is this?" — A unique identifier for your app, matching the bundle identifier in Xcode (e.g., `com.yourcompany.yourapp`). Registered in the Developer Portal.
- **Provisioning Profile** = "Who can run this app, and where?" — Ties together a certificate, an app ID, and (for development) a list of devices. It answers: "This developer is allowed to run this app on these devices."

"In web terms: the certificate is like an SSL certificate proving you own a domain. The app ID is like the domain name itself. The provisioning profile is like a deployment configuration that says which servers can host your site."

**Diagram to describe:**
```
Certificate (who) ──┐
                     ├──→ Provisioning Profile ──→ Signed App
App ID (what) ──────┘          │
                         Device List (where)
                         (development only)
```

**Development vs Distribution:**

| | Development | Distribution |
|---|---|---|
| Purpose | Run on test devices | App Store / TestFlight |
| Certificate type | Apple Development | Apple Distribution |
| Device list in profile | Yes (specific devices) | No (any device via App Store) |
| Created by | Xcode (automatic) or Developer Portal | Developer Portal or Xcode |

"For development, Xcode usually handles everything automatically — you may not have even noticed. Distribution is where you need to understand the system."

**Automatic vs Manual Signing:**

"Xcode offers two modes:"
- **Automatic signing** (recommended for most developers): Xcode creates and manages certificates and profiles for you. You select your team, and Xcode handles the rest.
- **Manual signing**: You create certificates and profiles yourself in the Developer Portal and select them in Xcode. Needed for advanced setups (CI/CD, complex team configurations).

"Start with automatic. Switch to manual only when automatic doesn't meet your needs (usually when setting up CI/CD)."

**Common Misconceptions:**
- Misconception: "I need to create certificates manually" → Clarify: "Xcode's automatic signing creates everything for you. Manual creation is for specific scenarios like CI/CD."
- Misconception: "My certificate is on my Mac, so only I can submit" → Clarify: "Distribution certificates can be shared across a team. In larger teams, a shared certificate (or Fastlane Match) ensures anyone can build for distribution."
- Misconception: "A provisioning profile is just a certificate" → Clarify: "A provisioning profile combines a certificate, an app ID, and (for development) device IDs. It's the link between all three."
- Misconception: "Code signing is Apple being difficult" → Clarify: "It serves a real purpose: users can trust that the app on their phone actually came from the developer and wasn't modified in transit. It's like HTTPS for apps."

**Verification Questions:**
1. "What are the three pieces involved in code signing, and what does each one represent?"
2. "What's the difference between a development and distribution provisioning profile?"
3. Multiple choice: "You're seeing 'No signing certificate found' in Xcode. What's the most likely issue? A) Your app ID is wrong B) Your certificate isn't in the Keychain C) Your provisioning profile expired D) Your device isn't registered"
4. "If you get a new Mac, what do you need to do to build your app for distribution?"

**Good answer indicators:**
- They can name all three pieces and explain how they relate
- They understand development is for testing, distribution is for release
- They know automatic signing handles most cases

**If they struggle:**
- Use the notarized document analogy more heavily
- Draw out the relationship: "Certificate + App ID + Devices = Profile"
- If the Keychain is confusing: "It's just your Mac's secure storage for certificates, like a password manager but for cryptographic keys"

---

### Section 2: App Store Connect Setup

**Core Concept to Teach:**
App Store Connect is Apple's portal for managing everything about your app's presence on the App Store — the listing, pricing, TestFlight, analytics, and user reviews. Before you can upload a build, you need to create an app record here.

**How to Explain:**
1. Walk through creating an app record
2. Explain the required information
3. Cover pricing and availability
4. Introduce app privacy details

**Creating an App Record:**

"In App Store Connect (appstoreconnect.apple.com), go to My Apps → + → New App."

Required fields:
- **Platform**: iOS
- **Name**: Your app's display name on the App Store (up to 30 characters). Must be unique across the entire App Store.
- **Primary Language**: The language of your default metadata
- **Bundle ID**: Must match the bundle identifier in Xcode. Select from the dropdown — it's populated from your Developer Portal app IDs.
- **SKU**: An internal identifier (not visible to users). Can be anything — use your bundle ID or a short code.

"The bundle ID connection is critical. The App Store Connect record and your Xcode project must use the same bundle identifier. If they don't match, uploads will fail."

**App Information:**

Walk through each tab:
- **App Information**: Name, subtitle (up to 30 chars), category, content rating
- **Pricing and Availability**: Free or paid, which countries/regions
- **App Privacy**: Data collection disclosures (the "nutrition labels" users see before downloading)

"The privacy section trips people up. Apple requires you to disclose what data your app collects, even if it's just analytics. Be thorough — incomplete privacy disclosures are a common rejection reason."

**Privacy Details:**

"Apple's privacy labels require you to answer:"
1. Do you collect data? (Almost every app does, even if just crash logs)
2. What types of data? (Contact info, usage data, identifiers, etc.)
3. Is it linked to the user's identity?
4. Is it used for tracking across other companies' apps?

"If you use any third-party SDKs (analytics, crash reporting, ads), you need to include their data collection too. Check each SDK's documentation for their privacy manifest."

**Common Misconceptions:**
- Misconception: "I can change the bundle ID later" → Clarify: "The bundle ID is permanent once set. Think of it like a primary key. Choose carefully."
- Misconception: "The app name on the App Store must match the name under the icon" → Clarify: "They can be different. The App Store name is in App Store Connect; the name under the icon is the 'Display Name' in Xcode's target settings."
- Misconception: "I don't collect data so I can skip privacy" → Clarify: "If you use any analytics, crash reporting, or even Apple's own frameworks that collect device data, you collect data. Review your dependencies."

**Verification Questions:**
1. "What connects your Xcode project to your App Store Connect listing?"
2. "What happens if your app's bundle ID in Xcode doesn't match the one in App Store Connect?"
3. "Your app uses Firebase Analytics and Crashlytics. What do you need to include in your privacy disclosures?"

---

### Section 3: Building for Distribution

**Core Concept to Teach:**
To upload to App Store Connect, you create an archive build — a special build type that packages your app with the distribution signing identity. This is done through Xcode's Product > Archive workflow.

**How to Explain:**
1. Explain the difference between a debug build and an archive
2. Walk through the archive process
3. Cover the upload flow
4. Address common validation errors

**Archive vs Debug Build:**

"When you hit Run in Xcode, it creates a debug build — optimized for fast compilation, includes debug symbols, signed for development. An archive is a release build — optimized for performance, stripped of debug info, signed for distribution."

| | Debug Build (Run) | Archive |
|---|---|---|
| Optimization | Debug (fast compile) | Release (fast execution) |
| Signing | Development | Distribution |
| Destination | Your device / simulator | App Store Connect |
| Created via | Cmd+R | Product > Archive |

**The Archive Process:**

1. Select a physical device (not a simulator) as the build destination — archives require a real device target
2. Product > Archive (or Cmd+Shift+Archive if you set up a shortcut)
3. Xcode compiles, links, and signs your app
4. On success, the Organizer window opens showing your archive

"The Organizer (Window > Organizer) is your archive library. Every archive you create lives here with its date, version, and build number."

**Uploading to App Store Connect:**

1. In the Organizer, select your archive
2. Click "Distribute App"
3. Choose "App Store Connect" as the distribution method
4. Choose "Upload" (not "Export")
5. Xcode validates the build — if there are issues, it reports them here
6. Upload completes → the build appears in App Store Connect within a few minutes

"After upload, Apple runs additional processing — checking for private API usage, validating the binary, generating app thinning variants. This takes a few minutes to an hour. You'll get an email if there are issues."

**Common Validation Errors:**

Present these as scenarios to troubleshoot:
- **"No accounts with App Store Connect access"**: Your Apple ID isn't added to the team with the right role. Check App Store Connect > Users and Access.
- **"Invalid bundle identifier"**: The bundle ID in your build doesn't match any app in App Store Connect. Create the app record first.
- **"Missing compliance"**: The app uses encryption. You'll need to provide export compliance information (most apps that only use HTTPS can declare an exemption).
- **"Invalid provisioning profile"**: The profile doesn't include the distribution certificate. Regenerate it or use automatic signing.

**Version and Build Numbers:**

"Two numbers matter:"
- **Version** (CFBundleShortVersionString): The marketing version users see — `1.0.0`, `1.1.0`, `2.0.0`. Follow semantic versioning.
- **Build number** (CFBundleVersion): An internal number that must increment with each upload. Multiple builds can share the same version. Think of it as: version 1.0.0, build 1; version 1.0.0, build 2 (fixed a bug before release).

"In web terms: the version is like your npm package version. The build number is like a CI build number — it always goes up."

**Common Misconceptions:**
- Misconception: "I can archive with the simulator selected" → Clarify: "Archives require a device target. Select 'Any iOS Device' or a connected physical device."
- Misconception: "I need to increment the version for every upload" → Clarify: "The build number must increase with each upload. The version only needs to change when you release a new version to users."
- Misconception: "Upload failed means my code is wrong" → Clarify: "Upload failures are usually signing, provisioning, or metadata issues — not code bugs. Read the error carefully."

**Verification Questions:**
1. "What's the difference between a debug build and an archive?"
2. "You uploaded version 1.0.0 build 1, but found a bug before releasing. What do you change for the next upload?"
3. "Your archive uploads successfully but you get an email saying 'Invalid Binary' 30 minutes later. Where do you look for details?"

---

### Section 4: TestFlight

**Core Concept to Teach:**
TestFlight is Apple's beta testing platform. It lets you distribute pre-release builds to testers before going to the App Store. There are two tester types — internal (your team, instant access) and external (anyone, requires a brief review).

**How to Explain:**
1. Explain internal vs external testing
2. Walk through setting up test groups
3. Show how testers install and provide feedback
4. Cover the TestFlight review process for external builds

**Internal vs External Testing:**

| | Internal | External |
|---|---|---|
| Who | Team members (App Store Connect users) | Anyone with an invite link |
| Max testers | 100 | 10,000 |
| Review required | No — available immediately | Yes — Apple reviews the first build |
| Automatic distribution | Can be enabled per tester | Manual distribution |
| Expires | 90 days after upload | 90 days after upload |

"Start with internal testing. Add yourself and your team as internal testers. Once a build is uploaded, internal testers get it immediately — no review needed."

**Setting Up Internal Testing:**

1. App Store Connect > Your App > TestFlight
2. Internal Testing > + to add testers (must be App Store Connect users on your team)
3. Enable "Automatic Distribution" if you want every new build sent automatically
4. Testers receive an email invitation or see the build in the TestFlight app

**Setting Up External Testing:**

1. TestFlight > External Testing > + to create a test group
2. Add testers by email or generate a public link
3. Select a build to distribute
4. Fill in "What to Test" — tells testers what to focus on
5. Submit for TestFlight review (first build only — subsequent builds to the same group usually auto-approve)

"TestFlight review is lighter than App Store review — it checks for crashes, basic functionality, and obvious guideline violations. It usually takes a day or less."

**Collecting Feedback:**

"Testers can submit feedback directly from the TestFlight app:"
- Screenshots with annotations
- Text feedback
- Crash reports (automatic)

"You see all feedback in App Store Connect > TestFlight > Feedback. Crash reports appear in Xcode's Organizer (Window > Organizer > Crashes)."

**Common Misconceptions:**
- Misconception: "TestFlight is only for beta testing" → Clarify: "Many teams use it for ongoing internal distribution — QA, stakeholder demos, dogfooding. It's not just for the week before launch."
- Misconception: "External testers need to be in my Developer account" → Clarify: "External testers just need an email. They don't need an Apple Developer account — they install the TestFlight app and accept the invite."
- Misconception: "Every build needs TestFlight review" → Clarify: "Only the first build sent to an external group is reviewed. Subsequent builds to the same group usually auto-approve unless the app changes significantly."

**Verification Questions:**
1. "What's the key difference between internal and external testers?"
2. "You want to send a build to your client for review. They're not part of your development team. Which type of testing do you use?"
3. "Builds in TestFlight expire after how many days?"

---

### Section 5: App Store Submission

**Core Concept to Teach:**
Submitting to the App Store requires complete metadata (description, screenshots, keywords), compliance with Apple's App Review Guidelines, and patience through the review process. Understanding common rejection reasons prevents frustrating back-and-forth.

**How to Explain:**
1. Walk through the metadata requirements
2. Cover App Review Guidelines highlights
3. Explain the submission and review flow
4. Address common rejection reasons and how to respond

**App Store Metadata:**

"Every field matters. This is your app's storefront — it's how users decide whether to download."

Required metadata:
- **Description** (up to 4,000 characters): What your app does. First sentence is critical — it's what shows in search results.
- **Keywords** (up to 100 characters, comma-separated): Search terms. Don't repeat words from your app name (they're already indexed).
- **Screenshots**: Required for each device size you support. At minimum: 6.7" (iPhone 15 Pro Max) and 6.5" (iPhone 11 Pro Max) cover most cases.
- **App Preview** (optional but recommended): Up to 30-second videos showing the app in action.
- **What's New** (for updates): Release notes visible to users.
- **Support URL**: Required. A webpage with support information.
- **Privacy Policy URL**: Required for all apps.

"Think of keywords like SEO — 100 characters isn't much, so use them strategically. Don't waste space on common words like 'app' or your app name."

**App Review Guidelines — Key Sections:**

"Apple's guidelines are long, but most rejections come from a few areas:"

1. **Guideline 2.1 — App Completeness**: The app must be fully functional. No placeholder content, broken links, or "coming soon" features. The review build should be production-ready.

2. **Guideline 2.3 — Accurate Metadata**: Screenshots must reflect actual app functionality. Description must be accurate. Don't claim features you don't have.

3. **Guideline 3.1.1 — In-App Purchase**: If you sell digital content or subscriptions, you must use Apple's in-app purchase system. Physical goods and services can use external payment.

4. **Guideline 4.0 — Design**: The app must provide enough functionality to be useful. "Wrapper" apps that just load a website will be rejected.

5. **Guideline 5.1.1 — Data Collection**: You must have a privacy policy and accurately declare data collection in App Privacy.

**The Submission Flow:**

1. In App Store Connect > App Store tab, select the version
2. Fill in all metadata fields
3. Select a build (from your uploaded builds)
4. Add app review information: notes for the reviewer, demo account if login is required
5. Click "Submit for Review"

"If your app requires login, provide a demo account with credentials. If it requires special hardware or location, explain how to test it in the review notes. Reviewers are humans — help them test your app."

**Common Rejection Reasons and Responses:**

- **"Guideline 2.1 — Bug/Crash"**: The reviewer found a crash or bug. Fix it, upload a new build, and resubmit. In your response, explain what you fixed.
- **"Guideline 2.3 — Inaccurate metadata"**: Your screenshots or description don't match the app. Update them and resubmit.
- **"Guideline 4.3 — Spam/Duplicate"**: Apple thinks your app is too similar to existing apps (including your own). You may need to explain what makes it different.
- **"Guideline 5.1.1 — Privacy"**: Missing or inaccurate privacy disclosures. Update your privacy policy and App Store privacy details.

"If you disagree with a rejection, you can respond in the Resolution Center (App Store Connect). Be factual and specific. If the reviewer misunderstood your app, explain clearly. You can also appeal to the App Review Board."

**Common Misconceptions:**
- Misconception: "Review takes weeks" → Clarify: "Most reviews complete within 24-48 hours. Plan for delays during holidays and major iOS releases."
- Misconception: "I need a privacy policy only if I collect personal data" → Clarify: "Every app submitted to the App Store requires a privacy policy URL. No exceptions."
- Misconception: "If rejected, I have to start over" → Clarify: "You respond in the Resolution Center, fix the issue, upload a new build to the same version, and resubmit. Your metadata is preserved."
- Misconception: "Expedited reviews are for anyone" → Clarify: "Expedited reviews are for critical bug fixes or time-sensitive events. Apple decides whether to grant them. Don't abuse it."

**Verification Questions:**
1. "Your app requires a login to access features. What should you include with your submission?"
2. "You receive a rejection for Guideline 2.1 (Bug). What's your process?"
3. Multiple choice: "Which of these would likely cause a rejection? A) Using a custom font B) Selling physical products with Stripe C) Selling digital stickers without In-App Purchase D) Having a 5-screen app"
4. "How would you handle a rejection you believe is incorrect?"

---

### Section 6: Post-Launch Management

**Core Concept to Teach:**
Launching is just the beginning. Post-launch involves monitoring your app's performance, releasing updates, responding to user feedback, and managing multiple versions simultaneously. Apple provides tools for phased releases, analytics, and review management.

**How to Explain:**
1. Cover phased releases and why you'd use them
2. Explain version updates and build management
3. Show analytics in App Store Connect
4. Discuss responding to user reviews

**Phased Releases:**

"Phased release distributes an update gradually over 7 days:"
- Day 1: 1% of users
- Day 2: 2%
- Day 3: 5%
- Day 4: 10%
- Day 5: 20%
- Day 6: 50%
- Day 7: 100%

"This is like a gradual rollout in web deployment. If the update has a critical bug, you can pause the rollout before all users get it. Users who manually check for updates can still get it early."

"Use phased releases for any non-trivial update. The safety net of being able to pause is worth the slower distribution."

**Version Updates:**

"To release an update:"
1. Increment the version number in Xcode (and build number)
2. Archive and upload
3. In App Store Connect, click + Version to create a new version
4. Fill in "What's New" (release notes)
5. Select the new build
6. Submit for review

"You can have multiple versions in different states — one live on the App Store, one in review, one in preparation. App Store Connect tracks them all."

**App Analytics:**

"App Store Connect > Analytics shows:"
- **Impressions**: How many times your app appeared in search results
- **Product Page Views**: How many users viewed your app's page
- **Downloads**: First-time downloads and re-downloads
- **Proceeds**: Revenue (if applicable)
- **Crashes**: Crash counts and details
- **Retention**: How many users return after 1 day, 7 days, 28 days

"For crash details, Xcode's Organizer (Window > Organizer > Crashes) gives you symbolicated crash logs — actual stack traces pointing to lines in your code."

**Responding to Reviews:**

"You can respond to user reviews in App Store Connect > Ratings and Reviews:"
- Responses are public — everyone sees them
- Be professional and helpful
- If the user reports a bug, acknowledge it and mention if it's fixed in an update
- Don't be defensive

**Common Misconceptions:**
- Misconception: "Phased release means I can roll back" → Clarify: "You can pause a phased release, but you can't roll back to a previous version. If there's a critical bug, pause the release and submit a fix as quickly as possible."
- Misconception: "I should wait for all reviews before releasing updates" → Clarify: "Ship fixes quickly. Users appreciate fast responses to bugs."
- Misconception: "Crash reports in App Store Connect are all I need" → Clarify: "App Store Connect shows aggregate data. Xcode Organizer shows symbolicated crash logs with actual stack traces. Use both."

**Verification Questions:**
1. "You released an update and are getting reports of a crash affecting some users. The update is in a phased release on day 2. What do you do?"
2. "What's the difference between crash data in App Store Connect and in Xcode Organizer?"
3. "A user leaves a 1-star review saying the app crashes on login. How would you respond publicly?"

---

### Section 7: Practice Project

**Project Introduction:**
"Let's walk through the complete publishing pipeline. Even if you're not ready to publish a real app, you'll configure everything as if you were — creating the App Store Connect listing, preparing metadata, and setting up TestFlight distribution."

**Requirements:**
1. Register an app ID in the Developer Portal (or verify your existing one)
2. Create an app record in App Store Connect with complete metadata
3. Configure code signing for distribution (automatic or manual)
4. Create an archive build
5. Upload to App Store Connect
6. Set up internal TestFlight testing and distribute a build
7. (Optional) Set up an external testing group

**Scaffolding Strategy:**
1. **If they want to try alone**: Give them the requirements and check in at milestones.
2. **If they want guidance**: Walk through each step together, explaining what each screen and option means.
3. **If they're using a sample app**: Use Apple's sample projects or a simple single-screen app — the publishing process is the same regardless of app complexity.

**Checkpoints During Project:**
- After app ID setup: "Does the bundle ID match between Xcode and the Developer Portal?"
- After App Store Connect record: "Is all the required metadata filled in?"
- After archive: "Did the archive succeed? Any warnings?"
- After upload: "Did the upload pass validation? Check email for any processing issues."
- After TestFlight setup: "Can you see the build in TestFlight? Did internal testers receive it?"

**If They Get Stuck:**
- On code signing: "Check Xcode > Signing & Capabilities. Is automatic signing enabled? Is the right team selected?"
- On archive failures: "Is a physical device (not simulator) selected as the build destination? Are there any build errors?"
- On upload failures: "Read the exact error message. Most upload issues are signing-related — verify the provisioning profile includes a distribution certificate."
- On TestFlight: "After upload, builds take a few minutes to process. Refresh App Store Connect if you don't see it yet."

---

## Wrap-Up and Next Steps

**Review Key Takeaways:**
1. Code signing uses three interconnected pieces: certificates, app IDs, and provisioning profiles
2. App Store Connect is the central hub for managing your app's listing, builds, testers, and analytics
3. Archive builds are distribution-signed builds uploaded from Xcode
4. TestFlight provides internal (instant) and external (reviewed) beta distribution
5. App Review checks for completeness, accuracy, and guideline compliance
6. Post-launch management includes phased releases, version updates, and monitoring

**Assess Confidence:**
"On a scale of 1-10, how confident do you feel about publishing an app?"
- 1-4: Walk through the process again with their specific app. Focus on whichever part is most confusing (usually code signing).
- 5-7: Normal for first-timers. Suggest actually going through the process — even uploading a simple test app to TestFlight builds confidence quickly.
- 8-10: Suggest ios-ci-cd-with-github-actions to automate the pipeline.

**Suggest Next Steps:**
- "iOS CI/CD with GitHub Actions automates the build and upload process so you're not doing it manually every time"
- "Try publishing a simple app end-to-end — a single-screen utility app is fine. The first time through is the learning experience."
- "Read Apple's App Review Guidelines at least once. It's long, but knowing the rules prevents rejections."

---

## Adaptive Teaching Strategies

### If Learner is Struggling
- Focus on the mental model of code signing first — everything else flows from understanding certificates, IDs, and profiles
- Use visual diagrams showing how the three pieces connect
- Walk through each App Store Connect screen together rather than describing it abstractly
- If overwhelmed by the number of steps, focus on just getting to TestFlight first — save App Store submission for a follow-up session

### If Learner is Excelling
- Discuss advanced signing scenarios: team certificates, enterprise distribution
- Cover App Store Optimization strategies (keywords, screenshots, A/B testing)
- Discuss the business side: pricing strategies, freemium vs paid, subscription models
- Introduce Xcode Cloud or CI/CD automation as the next step

### If Learner Seems Disengaged
- Focus on their specific app — work through the process with their actual project
- If the steps feel rote, skip ahead to TestFlight — seeing their app on a real device via TestFlight is motivating
- If code signing feels pointless, discuss why it matters from a user trust perspective

### Different Learning Styles
- **Visual learners**: Walk through App Store Connect and Xcode screens, describing layouts and where to click
- **Hands-on learners**: Start with the practice project immediately, explain concepts as they come up
- **Conceptual learners**: Spend more time on the "why" of code signing and Apple's motivations
- **Example-driven learners**: Show real App Store listings and discuss what makes them effective

---

## Troubleshooting Common Issues

### Code Signing Problems
- **"No signing certificate found"**: Open Keychain Access, check for expired or missing certificates. In Xcode, try Signing & Capabilities > enable/disable automatic signing to trigger re-download.
- **"Provisioning profile doesn't include the selected certificate"**: The profile and certificate are out of sync. With automatic signing, toggle it off and on. With manual, regenerate the profile in the Developer Portal.
- **"The certificate used to sign has been revoked"**: Someone on the team revoked it. Generate a new one in the Developer Portal (or let Xcode auto-create one).

### Upload Issues
- **"The app references non-public selectors"**: Your app uses private APIs. Remove them — Apple will reject the binary.
- **"Invalid binary"**: Often a signing issue or missing required capabilities (like app transport security exemptions without justification). Check the email from Apple for specific details.
- **"Redundant binary upload"**: You're uploading the same build number. Increment the build number and archive again.

### TestFlight Issues
- **Build not appearing**: Processing takes time (minutes to an hour). Check your email for processing errors.
- **Testers not receiving invites**: Verify their email address. For internal testers, they must be App Store Connect users with the Tester role.
- **"Beta App Review Rejected"**: Even TestFlight external builds get a light review. Check the rejection reason in App Store Connect — usually it's a crash or incomplete functionality.

---

## Teaching Notes

**Key Emphasis Points:**
- Code signing is the foundation. If they understand certificates, app IDs, and profiles, everything else is just clicking through screens.
- Help them develop a mental checklist for submission: signing? metadata? screenshots? privacy policy? demo account?
- The first submission is the hardest. After that, it's routine.
- Common rejections are preventable. Knowing the guidelines saves time.

**Pacing Guidance:**
- Section 1 (Code Signing): Foundation. Take time here — rushing this causes confusion in every later step.
- Section 2 (App Store Connect): Walkthrough pace. Lots of screens and fields, but conceptually simple.
- Section 3 (Building): Quick if they've used Xcode. Focus on version/build numbers.
- Section 4 (TestFlight): Practical and motivating. Let them set up real testing.
- Section 5 (Submission): Important for avoiding rejections. Cover the common reasons.
- Section 6 (Post-Launch): Brief unless they have a live app.
- Section 7 (Practice): Allow the most time here for hands-on experience.

**Success Indicators:**
You'll know they've got it when they:
- Can explain code signing without referring to notes
- Navigate App Store Connect confidently
- Know what to check when an upload fails
- Can list the top rejection reasons and how to avoid them
- Feel ready to publish their own app (even if nervous — that's normal for the first time)
