---
title: App Store Publishing
topics:
  - App Store Connect
  - TestFlight
  - Code Signing
  - Provisioning
  - App Review
related_routes:
  - xcode-essentials
  - ios-ci-cd-with-github-actions
---

# App Store Publishing - Route Map

## Overview

Getting an iOS app from "it runs on my phone" to "it's on the App Store" involves certificates, provisioning profiles, App Store Connect, TestFlight, and Apple's review process. This route demystifies the publishing pipeline — the parts that trip up every developer the first time through.

## What You'll Learn

By following this route, you will:
- Understand Apple's code signing system (certificates, identifiers, profiles)
- Configure app identifiers and capabilities in the Developer Portal
- Set up and manage builds in App Store Connect
- Distribute beta builds through TestFlight
- Prepare app metadata, screenshots, and descriptions for submission
- Submit for review and handle common rejection reasons
- Manage app versions, builds, and phased releases

## Prerequisites

Before starting this route:
- **Required**: A working iOS app to publish
- **Required**: Apple Developer Program membership ($99/year)
- **Required**: Xcode basics (see xcode-essentials)
- **Helpful**: Understanding of your app's capabilities and entitlements

## Route Structure

### 1. Code Signing Fundamentals
- Why Apple requires code signing
- Certificates: development vs distribution
- App IDs and bundle identifiers
- Provisioning profiles: what they are and how they connect certificates to devices
- Automatic vs manual signing in Xcode
- The Keychain and managing certificates

### 2. App Store Connect Setup
- Creating an app record
- App information: name, subtitle, category, privacy policy
- Pricing and availability
- App privacy details (the nutrition labels)

### 3. Building for Distribution
- Archive builds in Xcode
- Upload to App Store Connect
- Build processing and validation errors
- Symbols and crash reporting

### 4. TestFlight
- Internal testing (team members, up to 100 testers)
- External testing (public link, up to 10,000 testers)
- Test groups and build distribution
- Collecting feedback from testers
- TestFlight review for external testing

### 5. App Store Submission
- Version metadata: description, keywords, screenshots, previews
- App Review Guidelines — the common rejection reasons
- Submitting for review
- Responding to rejections and appeals
- Expedited reviews

### 6. Post-Launch Management
- Phased releases
- Version updates and build management
- App analytics in App Store Connect
- Responding to user reviews
- Managing multiple app versions

### 7. Practice Project
- Take an app through the full pipeline: configure signing, create App Store Connect listing, upload to TestFlight, distribute to testers

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- Apple Developer Portal (developer.apple.com)
- App Store Connect (appstoreconnect.apple.com)
- Xcode Organizer (archive and upload)
- TestFlight app

## Next Steps

After completing this route:
- **iOS CI/CD with GitHub Actions** - Automate the build and upload pipeline
