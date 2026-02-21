---
title: UIKit Essentials for SwiftUI Developers
topics:
  - UIKit
  - UIViewController
  - UIKit Integration
  - Storyboards
related_routes:
  - swiftui-fundamentals
  - ios-app-patterns
---

# UIKit Essentials - Route Map

## Overview

SwiftUI is the primary way to build iOS UIs, but UIKit is still everywhere — in existing codebases, third-party libraries, and for features SwiftUI doesn't yet support. This route gives SwiftUI developers enough UIKit literacy to read UIKit code, integrate UIKit components into SwiftUI apps, and handle situations where UIKit is the right tool.

## What You'll Learn

By following this route, you will:
- Understand UIKit's view controller lifecycle and how it differs from SwiftUI
- Read and understand UIKit code in existing projects and documentation
- Wrap UIKit views and view controllers for use in SwiftUI
- Identify when UIKit is needed vs when SwiftUI suffices
- Work with common UIKit components (table views, collection views, gestures)

## Prerequisites

Before starting this route:
- **Required**: SwiftUI fundamentals (see swiftui-fundamentals)
- **Required**: Swift language basics (see swift-for-developers)
- **Helpful**: Familiarity with imperative UI patterns from web development (DOM manipulation)

## Route Structure

### 1. UIKit Mental Model
- Imperative vs declarative (contrast with SwiftUI)
- UIView and UIViewController hierarchy
- View controller lifecycle (viewDidLoad, viewWillAppear, etc.)
- The responder chain
- Storyboards vs programmatic UI (and why SwiftUI replaced both)

### 2. Common UIKit Components
- UITableView and UICollectionView (SwiftUI's List equivalent)
- UINavigationController and UITabBarController
- UIAlertController
- Gesture recognizers

### 3. UIKit in SwiftUI
- UIViewRepresentable for wrapping UIKit views
- UIViewControllerRepresentable for wrapping view controllers
- Coordinator pattern for delegates and data sources
- When to use these (camera, maps, web views, complex text editing)

### 4. SwiftUI in UIKit
- UIHostingController for embedding SwiftUI views in UIKit apps
- Gradual migration strategies

### 5. Practice Project
- Build a SwiftUI app that wraps a UIKit component (e.g., camera picker or custom text view) using UIViewControllerRepresentable

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- Xcode Interface Builder (brief overview)
- UIKit framework documentation

## Next Steps

After completing this route:
- **iOS App Patterns** - Architecture patterns for larger apps
- **iOS Data Persistence** - Local data storage options
