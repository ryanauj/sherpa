---
title: iOS CI/CD with GitHub Actions
topics:
  - GitHub Actions
  - Fastlane
  - Code Signing in CI
  - TestFlight Automation
  - Continuous Deployment
related_routes:
  - app-store-publishing
  - xcode-essentials
---

# iOS CI/CD with GitHub Actions - Route Map

## Overview

Manually archiving and uploading builds gets old fast. This route covers automating your iOS build, test, and deployment pipeline with GitHub Actions and Fastlane — from running tests on every PR to automatically uploading builds to TestFlight on merge to main.

## What You'll Learn

By following this route, you will:
- Set up GitHub Actions workflows for iOS projects
- Run tests and build your app in CI
- Handle code signing in a CI environment (the hardest part)
- Use Fastlane to automate build, sign, and upload steps
- Automatically deploy to TestFlight on merge
- Manage secrets (certificates, API keys) securely in CI
- Choose between GitHub-hosted and self-hosted macOS runners

## Prerequisites

Before starting this route:
- **Required**: App Store publishing basics (see app-store-publishing)
- **Required**: Familiarity with Git and GitHub
- **Required**: A working iOS app with tests
- **Helpful**: Basic YAML syntax
- **Helpful**: Experience with any CI/CD system (Jenkins, CircleCI, GitLab CI)

## Route Structure

### 1. CI/CD Concepts for iOS
- Why CI/CD matters for mobile development
- The iOS CI/CD pipeline: build → test → sign → upload
- Challenges specific to iOS: macOS requirement, code signing, provisioning
- GitHub-hosted macOS runners vs self-hosted runners (cost, performance, availability)

### 2. Your First GitHub Actions Workflow
- Workflow file structure (.github/workflows/)
- Triggering on push, PR, and manual dispatch
- The macOS runner environment (what's pre-installed)
- Checking out code and selecting Xcode version
- Building with xcodebuild
- Running tests and reporting results

### 3. Fastlane Setup
- What Fastlane does (and why it's worth the setup)
- Installing Fastlane and initializing for your project
- Fastfile, Appfile, and lanes
- Key actions: build_app, run_tests, upload_to_testflight
- Running Fastlane locally before CI

### 4. Code Signing in CI
- The challenge: signing needs certificates and profiles not in the repo
- Approach 1: Fastlane Match (recommended — stores encrypted certs in a private repo)
- Approach 2: Manual certificate installation from GitHub Secrets
- App Store Connect API keys for authentication (no Apple ID password in CI)
- Provisioning profile management

### 5. Automated TestFlight Deployment
- Building and signing for distribution in CI
- Uploading to TestFlight with Fastlane
- Setting up automatic deploys on merge to main
- Version and build number management in CI
- Changelog generation

### 6. Advanced Workflow Patterns
- Caching dependencies (SPM packages, Ruby gems) for faster builds
- Matrix builds for multiple iOS versions
- Conditional steps (deploy only on main, test on PRs)
- Build artifacts and retention
- Notifications (Slack, email) on build status

### 7. Practice Project
- Set up a complete CI/CD pipeline: run tests on PRs, deploy to TestFlight on merge to main, with proper code signing via Fastlane Match

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- GitHub Actions
- Fastlane
- xcodebuild
- App Store Connect API
- Fastlane Match

## Next Steps

After completing this route:
- This is the capstone of the iOS development routes
- Consider building and shipping a complete app using the **My First iOS App** ascent
