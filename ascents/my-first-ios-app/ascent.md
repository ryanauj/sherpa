---
title: "My First iOS App"
routes:
  - xcode-essentials
  - swift-for-developers
  - swiftui-fundamentals
  - ios-data-persistence
  - cloudkit-integration
  - ios-app-patterns
  - app-store-publishing
  - ios-ci-cd-with-github-actions
---

# My First iOS App

## Overview

This ascent guides you through building a complete iOS app from scratch — a personal reading list tracker. You'll start with an empty Xcode project and end with a polished app that persists data, syncs across devices via iCloud, and is published to TestFlight. Each checkpoint corresponds to a route: learn the skills, then apply them to the project.

## The Summit

A fully functional reading list app that:
- Displays books in a searchable, sortable list
- Lets you add books with title, author, and notes
- Tracks reading status (want to read, reading, finished)
- Persists data locally with SwiftData
- Syncs across your devices via CloudKit
- Has a clean, native-feeling iOS UI
- Is distributed to testers via TestFlight
- Builds and deploys automatically via GitHub Actions

## Prerequisites

No iOS experience required — this ascent starts from zero. You should have:
- A Mac with Xcode installed
- An Apple Developer Program membership (for CloudKit and TestFlight)
- A GitHub account (for CI/CD)
- Programming experience in any language

## Route Checkpoints

### Checkpoint 1: Set Up Your Workspace → Route: xcode-essentials

**What you'll learn**: Xcode navigation, project creation, simulators, debugging basics.

**Apply to the project**:
- Create a new iOS App project in Xcode called "ReadingList"
- Choose SwiftUI for the interface and Swift for the language
- Explore the generated project structure (understand what each file does)
- Run the app on the simulator — see the default "Hello, World!" screen
- Set a breakpoint in ContentView and inspect the view hierarchy

**Milestone**: You have a running (empty) iOS app and can navigate Xcode confidently.

---

### Checkpoint 2: Learn the Language → Route: swift-for-developers

**What you'll learn**: Swift's unique features — optionals, value types, protocols, closures, async/await.

**Apply to the project**:
- Define a `Book` struct with properties: `title` (String), `author` (String), `notes` (String?), `status` (an enum: `.wantToRead`, `.reading`, `.finished`), `dateAdded` (Date)
- Make `Book` conform to `Identifiable` (add an `id` property)
- Write a function that filters a `[Book]` array by status using closures
- Handle the optional `notes` property safely with optional binding

**Milestone**: You have a `Book` model with Swift best practices — value type, protocol conformance, safe optional handling.

---

### Checkpoint 3: Build the UI → Route: swiftui-fundamentals

**What you'll learn**: SwiftUI views, modifiers, state management, navigation, lists, forms.

**Apply to the project**:
- Build a `BookListView` that displays books in a `List` grouped by reading status
- Create a `BookDetailView` showing all book properties
- Add a `NavigationStack` connecting the list to detail views
- Build an `AddBookView` with a `Form` for entering book details
- Use `@State` for form fields and `@Binding` to pass data back
- Add swipe-to-delete and pull-to-refresh to the list
- Add a search bar with `.searchable()`

**Milestone**: A multi-screen app with navigation, lists, forms, and state management — using hardcoded sample data.

---

### Checkpoint 4: Persist Your Data → Route: ios-data-persistence

**What you'll learn**: SwiftData for structured persistence, UserDefaults for preferences.

**Apply to the project**:
- Convert the `Book` struct to a SwiftData `@Model` class
- Set up a `ModelContainer` in the app entry point
- Replace hardcoded data with `@Query` in `BookListView`
- Save books to SwiftData when the user submits the add form
- Add delete and edit support backed by persistence
- Use `@AppStorage` for a user preference (e.g., default sort order)

**Milestone**: Books persist across app launches. Close the app, reopen it, and your data is still there.

---

### Checkpoint 5: Sync Across Devices → Route: cloudkit-integration

**What you'll learn**: CloudKit containers, sync, conflict handling, sharing.

**Apply to the project**:
- Enable the CloudKit capability in your Xcode project
- Configure the SwiftData `ModelContainer` with a CloudKit container
- Test sync between simulator and a physical device (or two simulators with different iCloud accounts)
- Handle the case where the user isn't signed into iCloud
- Add error handling for sync failures

**Milestone**: Add a book on your phone, see it appear on your iPad (or second simulator). Data syncs via iCloud.

---

### Checkpoint 6: Structure Your Code → Route: ios-app-patterns

**What you'll learn**: MVVM architecture, dependency injection, navigation patterns.

**Apply to the project**:
- Extract business logic from views into `@Observable` view models
- Create a `BookListViewModel` that handles filtering, sorting, and search
- Move data access behind a repository protocol
- Use `@Environment` for dependency injection
- Organize the project into folders by feature (BookList, BookDetail, AddBook, Shared)

**Milestone**: Clean architecture — views are thin, logic is testable, dependencies are injectable.

---

### Checkpoint 7: Ship to Testers → Route: app-store-publishing

**What you'll learn**: Code signing, App Store Connect, TestFlight distribution.

**Apply to the project**:
- Configure your app's bundle identifier and team
- Create an App Store Connect record for ReadingList
- Archive and upload a build to App Store Connect
- Set up a TestFlight internal testing group
- Distribute the build to at least one tester
- Collect and review tester feedback

**Milestone**: Real people are using your app on their devices via TestFlight.

---

### Checkpoint 8: Automate Everything → Route: ios-ci-cd-with-github-actions

**What you'll learn**: GitHub Actions, Fastlane, automated signing and deployment.

**Apply to the project**:
- Push the ReadingList project to a GitHub repository
- Set up Fastlane with Match for code signing
- Create a GitHub Actions workflow that runs tests on pull requests
- Create a deployment workflow that uploads to TestFlight on merge to main
- Make a small change, merge a PR, and watch it deploy automatically

**Milestone**: Merge to main → automatic TestFlight build. The full pipeline works end to end.

## Summit Review

Congratulations — you've built and shipped an iOS app. Your reading list app:
- [x] Has a polished SwiftUI interface with navigation, lists, and forms
- [x] Persists data locally with SwiftData
- [x] Syncs across devices via CloudKit
- [x] Follows MVVM architecture with clean separation of concerns
- [x] Is distributed to testers via TestFlight
- [x] Builds and deploys automatically via GitHub Actions

## Extending the Ascent

Ideas for going further:
- Add a barcode scanner to look up books by ISBN (camera integration via UIKit bridge)
- Implement CloudKit sharing so users can share reading lists with friends
- Add widgets showing currently-reading books (WidgetKit)
- Support multiple reading lists with tags or categories
- Add reading progress tracking (page numbers, percentage)
- Submit to the App Store for public release
