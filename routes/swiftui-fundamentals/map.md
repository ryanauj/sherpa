---
title: SwiftUI Fundamentals
topics:
  - SwiftUI
  - Declarative UI
  - State Management
  - Navigation
  - View Composition
related_routes:
  - swift-for-developers
  - uikit-essentials
  - ios-app-patterns
---

# SwiftUI Fundamentals - Route Map

## Overview

SwiftUI is Apple's declarative UI framework. If you've used React, the mental model will feel familiar — you describe what the UI should look like for a given state, and the framework handles updates. This route covers SwiftUI's core building blocks, with React comparisons where they help clarify concepts.

## What You'll Learn

By following this route, you will:
- Build views using SwiftUI's declarative syntax
- Compose complex UIs from small, reusable view components
- Manage state with @State, @Binding, @Observable, and @Environment
- Build navigation flows with NavigationStack and sheets
- Display dynamic data with List and ForEach
- Apply styling with view modifiers
- Handle user input with forms, text fields, and gestures
- Preview and iterate on UI with Xcode's canvas

## Prerequisites

Before starting this route:
- **Required**: Swift language fundamentals (see swift-for-developers)
- **Required**: Xcode basics (see xcode-essentials)
- **Helpful**: Experience with any component-based UI framework (React, Vue, etc.)

## Route Structure

### 1. SwiftUI Mental Model
- Declarative vs imperative UI (React comparison)
- Views are structs, not objects
- The body property and view builder
- Xcode canvas and live previews

### 2. Built-in Views and Modifiers
- Text, Image, Button, Toggle, TextField, Picker
- View modifiers: padding, background, font, foregroundStyle
- Modifier ordering matters (why .padding().background() differs from .background().padding())
- Stacking views: VStack, HStack, ZStack
- Spacer and frame for layout

### 3. State Management
- @State for local view state (like useState in React)
- @Binding for passing state to child views (like props + callback)
- @Observable for shared model objects (like React context + state)
- @Environment for system-provided values (color scheme, locale, dismiss)
- When to use each and how they compare to React patterns

### 4. Lists and Dynamic Content
- List and ForEach
- Identifiable protocol
- Swipe actions and list styles
- Sections and grouped lists
- Pull to refresh

### 5. Navigation
- NavigationStack and NavigationLink
- Programmatic navigation with navigation paths
- Sheets, full-screen covers, and alerts
- Tab views with TabView
- Comparison to React Router concepts

### 6. User Input and Forms
- Form and Section for settings-style UIs
- TextField, SecureField, TextEditor
- Picker, DatePicker, Slider, Stepper
- Validation patterns

### 7. Practice Project
- Build a multi-screen app with a list view, detail view, navigation, state management, and user input

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- Xcode canvas (SwiftUI previews)
- SF Symbols for iconography
- Apple Human Interface Guidelines

## Next Steps

After completing this route:
- **UIKit Essentials** - For when SwiftUI doesn't cover your needs
- **iOS App Patterns** - Architecture patterns for structuring SwiftUI apps
- **iOS Data Persistence** - Storing data locally on device
