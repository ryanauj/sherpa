---
title: iOS App Patterns
topics:
  - MVVM
  - Architecture
  - Dependency Injection
  - SwiftUI Patterns
related_routes:
  - swift-for-developers
  - swiftui-fundamentals
  - ios-data-persistence
  - cloudkit-integration
---

# iOS App Patterns - Route Map

## Overview

Building a single-screen demo is straightforward. Building a multi-screen app with shared state, network calls, data persistence, and testability requires architectural decisions. This route covers the patterns that the iOS community has settled on for structuring SwiftUI apps, with comparisons to web architecture patterns where helpful.

## What You'll Learn

By following this route, you will:
- Structure a SwiftUI app using MVVM
- Separate concerns between views, view models, and data layers
- Use dependency injection for testability and flexibility
- Manage navigation flow in larger apps
- Handle network requests and map responses to models
- Organize code into modules using Swift packages

## Prerequisites

Before starting this route:
- **Required**: SwiftUI fundamentals (see swiftui-fundamentals)
- **Required**: Swift language proficiency (see swift-for-developers)
- **Helpful**: Experience with MVC or MVVM in web frameworks

## Route Structure

### 1. Why Architecture Matters
- The problem: SwiftUI views doing too much
- Separation of concerns in a mobile context
- Comparison to web app architecture (components, services, stores)

### 2. MVVM in SwiftUI
- Model: data structures and business logic
- View: SwiftUI views (the declarative UI)
- ViewModel: @Observable classes bridging models to views
- Where logic belongs (and where it doesn't)
- How this compares to React's container/presenter pattern

### 3. The Data Layer
- Repository pattern for data access
- Abstracting persistence behind protocols
- Networking: URLSession, Codable, and async/await
- Mapping API responses to domain models

### 4. Dependency Injection
- Why constructor injection matters for iOS
- Using @Environment for SwiftUI-native DI
- Building a simple dependency container
- Testing with mock dependencies

### 5. Navigation Patterns
- Centralized navigation with NavigationStack paths
- Coordinator-style navigation for complex flows
- Deep linking considerations

### 6. Error Handling and Loading States
- Representing async states (loading, loaded, error)
- User-facing error presentation
- Retry patterns

### 7. Project Organization
- Organizing by feature vs by layer
- Using Swift packages for modularization
- Shared vs app-specific code

### 8. Practice Project
- Refactor a monolithic SwiftUI app into a well-structured MVVM architecture with a data layer, dependency injection, and organized navigation

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- Xcode project and package management
- Swift Testing framework
- URLSession and Codable

## Next Steps

After completing this route:
- **iOS Data Persistence** - Add local data storage to your app
- **CloudKit Integration** - Sync data across devices with iCloud
- **App Store Publishing** - Ship your well-structured app
