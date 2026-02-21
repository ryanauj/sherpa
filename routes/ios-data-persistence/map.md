---
title: iOS Data Persistence
topics:
  - SwiftData
  - Core Data
  - UserDefaults
  - Keychain
  - Local Storage
related_routes:
  - swift-for-developers
  - swiftui-fundamentals
  - cloudkit-integration
  - ios-app-patterns
---

# iOS Data Persistence - Route Map

## Overview

iOS apps have several options for storing data locally, from simple key-value storage to full relational databases. This route covers when to use each option and how to implement them, with a focus on SwiftData (Apple's modern persistence framework) and the simpler alternatives for when a database is overkill.

## What You'll Learn

By following this route, you will:
- Choose the right persistence option for different types of data
- Store simple preferences and settings with UserDefaults
- Securely store credentials and tokens with Keychain
- Model and persist structured data with SwiftData
- Query, filter, and sort persisted data
- Understand the relationship between SwiftData and Core Data
- Integrate persistence with SwiftUI views

## Prerequisites

Before starting this route:
- **Required**: Swift language proficiency (see swift-for-developers)
- **Required**: SwiftUI fundamentals (see swiftui-fundamentals)
- **Helpful**: Experience with any ORM or database framework
- **Helpful**: iOS app architecture patterns (see ios-app-patterns)

## Route Structure

### 1. Choosing a Persistence Strategy
- UserDefaults: when and what (small, simple, non-sensitive data)
- Keychain: credentials, tokens, and sensitive data
- File system: documents, caches, and large blobs
- SwiftData: structured, queryable data with relationships
- Core Data: the predecessor (when you'll encounter it)
- Decision framework: which to use when

### 2. UserDefaults and AppStorage
- Storing and retrieving values
- @AppStorage in SwiftUI for reactive preferences
- Supported types and limitations
- What NOT to put in UserDefaults

### 3. Keychain Basics
- Why Keychain over UserDefaults for secrets
- Storing and retrieving credentials
- Keychain access groups for sharing between apps

### 4. SwiftData Fundamentals
- @Model macro for defining data models
- ModelContainer and ModelContext
- Creating, reading, updating, and deleting records
- Relationships between models
- Comparison to ORMs you may know (ActiveRecord, SQLAlchemy, Prisma)

### 5. Querying with SwiftData
- #Predicate for type-safe queries
- SortDescriptor for ordering
- @Query in SwiftUI views
- Filtering, searching, and pagination

### 6. SwiftData and SwiftUI Integration
- Setting up ModelContainer in the app
- @Query property wrapper for live data in views
- Editing and saving in response to user interaction
- Undo support

### 7. Core Data Overview
- When you'll encounter Core Data (existing projects, older tutorials)
- Key differences from SwiftData
- Migration path from Core Data to SwiftData

### 8. Practice Project
- Build a note-taking or task-tracking app using SwiftData with multiple models, queries, and SwiftUI integration. Use UserDefaults for app preferences.

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- SwiftData framework
- Xcode data model editor
- Core Data (overview only)

## Next Steps

After completing this route:
- **CloudKit Integration** - Sync your persisted data across devices via iCloud
- **App Store Publishing** - Ship your data-driven app
