---
title: Swift for Experienced Developers
topics:
  - Swift
  - Type System
  - Optionals
  - Protocols
  - Concurrency
related_routes:
  - xcode-essentials
  - swiftui-fundamentals
  - ios-app-patterns
---

# Swift for Developers - Route Map

## Overview

Swift has a few concepts that trip up developers coming from other languages — optionals, value semantics, protocol-oriented design, and a unique approach to memory management. This route skips "what is a variable" and focuses on what makes Swift different from languages like Python, Java, TypeScript, or Go.

## What You'll Learn

By following this route, you will:
- Use optionals confidently with unwrapping, chaining, and guard statements
- Understand the difference between value types and reference types and when to use each
- Design with protocols and extensions instead of inheritance hierarchies
- Write closures and use them with Swift's standard library
- Handle errors with Swift's typed error system
- Use async/await and structured concurrency for asynchronous work
- Understand ARC (Automatic Reference Counting) and how to avoid retain cycles

## Prerequisites

Before starting this route:
- **Required**: Proficiency in at least one programming language
- **Required**: Xcode installed and basic navigation (see xcode-essentials)
- **Helpful**: Familiarity with static typing and generics

## Route Structure

### 1. Swift Type System Foundations
- Type inference and explicit types
- Structs vs classes (value vs reference semantics)
- Enums with associated values
- Tuples and type aliases

### 2. Optionals
- The problem optionals solve (vs null/nil in other languages)
- Optional binding (if let, guard let)
- Optional chaining
- Nil coalescing (??)
- Force unwrapping and when it's acceptable
- Implicitly unwrapped optionals

### 3. Protocols and Extensions
- Protocols as interfaces (comparison to TypeScript interfaces, Java interfaces)
- Protocol conformance and extensions
- Default implementations
- Protocol-oriented design vs inheritance
- Common standard library protocols (Codable, Hashable, Identifiable)

### 4. Closures and Higher-Order Functions
- Closure syntax and trailing closure shorthand
- Capturing values and capture lists
- map, filter, reduce, compactMap
- Escaping vs non-escaping closures

### 5. Error Handling
- Throwing functions, do/try/catch
- Result type
- Comparison to exceptions in other languages
- try? and try! shortcuts

### 6. Concurrency
- async/await basics
- Tasks and structured concurrency
- Actors and data isolation
- MainActor for UI work
- Comparison to promises/async in JavaScript

### 7. Memory Management
- How ARC works (vs garbage collection)
- Strong, weak, and unowned references
- Retain cycles and how to spot them
- Closures and capture lists for breaking cycles

### 8. Practice Project
- Build a small command-line tool or data processing module that exercises optionals, protocols, closures, async/await, and error handling

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- Xcode Playgrounds (for experimenting with Swift)
- Swift REPL
- Swift standard library documentation

## Next Steps

After completing this route:
- **SwiftUI Fundamentals** - Build user interfaces with Swift
- **iOS App Patterns** - Architecture and design patterns for Swift apps
