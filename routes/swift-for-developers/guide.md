---
title: Swift for Experienced Developers
route_map: /routes/swift-for-developers/map.md
paired_sherpa: /routes/swift-for-developers/sherpa.md
prerequisites:
  - Proficiency in at least one programming language
  - Xcode installed and basic navigation
topics:
  - Swift
  - Type System
  - Optionals
  - Protocols
  - Concurrency
---

# Swift for Experienced Developers - Guide

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/swift-for-developers/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/swift-for-developers/map.md` for the high-level overview.

## Overview

You already know how to program. You know variables, loops, functions, classes, generics. This guide skips all of that and focuses on where Swift will genuinely surprise you coming from Python, Java, TypeScript, or Go.

The biggest shifts: value semantics are everywhere (arrays, strings, dictionaries are all structs), optionals replace null with compiler-enforced safety, protocols replace inheritance as the primary abstraction tool, and memory management is deterministic reference counting rather than garbage collection.

Every example in this guide runs in a Swift Playground (Xcode > File > New > Playground) or the Swift REPL (`swift` in your terminal).

## Learning Objectives

By the end of this guide, you will be able to:
- Use Swift's type system including structs, classes, enums with associated values, and generics
- Handle optionals confidently with unwrapping, chaining, and guard statements
- Design with protocols and extensions instead of class hierarchies
- Write closures and use them with higher-order functions
- Handle errors with throwing functions and the Result type
- Use async/await with structured concurrency and actors
- Understand ARC and prevent retain cycles

## Prerequisites

Before starting:
- **Required**: You can write production code in at least one language (Python, Java, TypeScript, Go, etc.)
- **Required**: Xcode installed (for Playgrounds) or Swift toolchain on Linux
- **Helpful**: Familiarity with static typing and generics

## Setup

Open Xcode and create a new Playground:

1. Xcode > File > New > Playground
2. Choose "Blank" under macOS
3. Name it "SwiftGuide"

Or use the Swift REPL from your terminal:
```bash
swift
```

**Verify Swift version:**
```bash
swift --version
```

You should see something like:
```
swift-driver version: 1.x.x Apple Swift version 5.x.x
```

---

## Section 1: Swift Type System Foundations

### Type Inference and Explicit Types

Swift has type inference similar to TypeScript, Go, or Kotlin. The compiler figures out the type from context, but every value has a concrete type at compile time.

```swift
let name = "Ada"          // String (inferred)
let age: Int = 30          // Int (explicit)
var score = 99.5           // Double (inferred — Swift defaults to Double, not Float)
let isActive = true        // Bool (inferred)
```

`let` is a constant (like `const` in JS/TS or `final` in Java). `var` is a variable. Use `let` by default — the compiler warns you if you use `var` unnecessarily.

One thing that may catch you off guard: Swift has no implicit type conversion.

```swift
let x: Int = 42
let y: Double = 3.14

// let sum = x + y   // Error: cannot add Int and Double
let sum = Double(x) + y   // 45.14 — explicit conversion required
```

If you're coming from Python or JavaScript, this feels strict. If you're coming from Go, this feels familiar.

### Structs vs Classes: Value vs Reference Semantics

This is the single biggest mental shift in Swift. In most languages, you think in terms of objects (reference types). In Swift, you think in terms of values first.

**Structs are value types.** Assigning or passing a struct copies it.

```swift
struct Point {
    var x: Double
    var y: Double
}

var a = Point(x: 1, y: 2)
var b = a          // b is a COPY of a
b.x = 99

print(a.x)  // 1.0 — a is unchanged
print(b.x)  // 99.0
```

**Expected Output:**
```
1.0
99.0
```

**Classes are reference types.** Assigning or passing a class shares the same instance.

```swift
class Cursor {
    var x: Double
    var y: Double

    init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }
}

var c = Cursor(x: 1, y: 2)
var d = c          // d points to the SAME object as c
d.x = 99

print(c.x)  // 99.0 — c changed because d is the same object
print(d.x)  // 99.0
```

**Expected Output:**
```
99.0
99.0
```

Here is the critical insight: **Array, Dictionary, String, and Set are all structs in Swift.** They are value types. When you assign an array to a new variable, you get a copy.

```swift
var list1 = [1, 2, 3]
var list2 = list1      // list2 is a COPY
list2.append(4)

print(list1)  // [1, 2, 3] — unchanged
print(list2)  // [1, 2, 3, 4]
```

**Expected Output:**
```
[1, 2, 3]
[1, 2, 3, 4]
```

In Python, Java, or TypeScript, that assignment would share the same list. In Swift, you get a copy. (Under the hood, Swift uses copy-on-write optimization, so the actual memory copy only happens when the copy is mutated.)

**When to use which:**
- **Struct** (default choice): Data that represents a value. If two `Point(x: 1, y: 2)` are equivalent regardless of identity, use a struct.
- **Class**: When you need shared mutable state, identity, or inheritance. UI view controllers, shared caches, objects that need to be referenced from multiple places.

Swift convention is struct-first. The standard library is almost entirely structs.

### Structs Get Automatic Initializers

Structs generate a memberwise initializer automatically. Classes do not.

```swift
struct User {
    var name: String
    var email: String
}

// Automatic memberwise init — you don't write this yourself
let user = User(name: "Ada", email: "ada@example.com")
```

Classes require you to write `init` explicitly:

```swift
class Account {
    var name: String
    var email: String

    init(name: String, email: String) {
        self.name = name
        self.email = email
    }
}
```

### Mutating Methods on Structs

Because structs are value types, methods that modify `self` must be marked `mutating`. This is the compiler's way of saying "this method changes the value."

```swift
struct Counter {
    var count = 0

    mutating func increment() {
        count += 1
    }
}

var counter = Counter()
counter.increment()
print(counter.count)  // 1
```

**Expected Output:**
```
1
```

You cannot call a `mutating` method on a `let` constant:

```swift
let fixedCounter = Counter()
// fixedCounter.increment()  // Error: cannot use mutating member on immutable value
```

### Enums with Associated Values

Swift enums are algebraic data types — far more powerful than enums in Java, TypeScript, or Go.

Basic enums work like you'd expect:

```swift
enum Direction {
    case north, south, east, west
}

let heading = Direction.north
```

But enums can also carry data via **associated values**:

```swift
enum NetworkResult {
    case success(data: Data, statusCode: Int)
    case failure(error: String)
    case loading(progress: Double)
}

let result = NetworkResult.success(data: Data(), statusCode: 200)
```

You extract associated values with pattern matching:

```swift
switch result {
case .success(let data, let statusCode):
    print("Got \(data.count) bytes, status \(statusCode)")
case .failure(let error):
    print("Error: \(error)")
case .loading(let progress):
    print("Loading: \(progress * 100)%")
}
```

**Expected Output:**
```
Got 0 bytes, status 200
```

This is similar to Rust's enums or TypeScript's discriminated unions, but it's a first-class language feature with compiler-enforced exhaustiveness. The `switch` must handle every case — no forgotten branches.

**Comparison to other languages:**

| Language | Closest equivalent |
|----------|-------------------|
| TypeScript | Discriminated unions (`type Result = { kind: "success", data: Data } \| ...`) |
| Rust | `enum` with variants |
| Java | Sealed classes (Java 17+) |
| Python | No direct equivalent; you'd use a class hierarchy or dataclasses with a tag |
| Go | No direct equivalent; you'd use interfaces or struct-with-tag patterns |

### Tuples and Type Aliases

Tuples group values without defining a struct. They're useful for returning multiple values from a function.

```swift
func divide(_ a: Int, by b: Int) -> (quotient: Int, remainder: Int) {
    return (a / b, a % b)
}

let result = divide(17, by: 5)
print(result.quotient)    // 3
print(result.remainder)   // 2

// Or destructure:
let (q, r) = divide(17, by: 5)
print(q)  // 3
```

**Expected Output:**
```
3
2
3
```

Similar to Go's multiple return values or Python's tuple unpacking, but Swift lets you name the components.

**Type aliases** give existing types a readable name:

```swift
typealias UserID = String
typealias Coordinate = (latitude: Double, longitude: Double)

func distanceBetween(_ a: Coordinate, _ b: Coordinate) -> Double {
    let dx = a.latitude - b.latitude
    let dy = a.longitude - b.longitude
    return (dx * dx + dy * dy).squareRoot()
}
```

### Exercise 1.1: Model a Shape

**Task:** Define a `Shape` enum with cases `circle(radius: Double)`, `rectangle(width: Double, height: Double)`, and `triangle(base: Double, height: Double)`. Write a function `area(of:)` that takes a `Shape` and returns its area. Use a `switch` statement.

<details>
<summary>Hint 1: Getting started</summary>

Define the enum with three cases, each with associated values. The function signature is `func area(of shape: Shape) -> Double`.
</details>

<details>
<summary>Hint 2: Area formulas</summary>

- Circle: pi * radius * radius
- Rectangle: width * height
- Triangle: 0.5 * base * height

Use `Double.pi` for pi.
</details>

<details>
<summary>Solution</summary>

```swift
enum Shape {
    case circle(radius: Double)
    case rectangle(width: Double, height: Double)
    case triangle(base: Double, height: Double)
}

func area(of shape: Shape) -> Double {
    switch shape {
    case .circle(let radius):
        return Double.pi * radius * radius
    case .rectangle(let width, let height):
        return width * height
    case .triangle(let base, let height):
        return 0.5 * base * height
    }
}

print(area(of: .circle(radius: 5)))           // 78.539...
print(area(of: .rectangle(width: 4, height: 3)))  // 12.0
print(area(of: .triangle(base: 6, height: 4)))    // 12.0
```

**Expected Output:**
```
78.53981633974483
12.0
12.0
```
</details>

### Exercise 1.2: Struct vs Class Behavior

**Task:** Predict the output of this code before running it, then verify in a Playground:

```swift
struct Size {
    var width: Double
    var height: Double
}

class Label {
    var text: String
    var size: Size

    init(text: String, size: Size) {
        self.text = text
        self.size = size
    }
}

var label1 = Label(text: "Hello", size: Size(width: 100, height: 50))
var label2 = label1
label2.text = "World"
label2.size.width = 200

print(label1.text)
print(label1.size.width)
```

<details>
<summary>Solution</summary>

```
World
200.0
```

`Label` is a class, so `label2 = label1` makes both variables point to the same object. Changing `label2.text` also changes `label1.text`. Even though `Size` is a struct, it's a property on the shared class instance, so the mutation is visible through both references.
</details>

### Checkpoint 1

Before moving on, make sure you can:
- [ ] Explain the difference between `let` and `var`
- [ ] Explain value semantics vs reference semantics
- [ ] Describe when to use a struct vs a class
- [ ] Define an enum with associated values and pattern match on it
- [ ] Use tuples to return multiple values from a function

---

## Section 2: Optionals

### The Problem Optionals Solve

Every language handles "the absence of a value" differently:

| Language | Approach |
|----------|----------|
| Python | `None` (any variable can be None) |
| Java | `null` (any reference can be null) |
| TypeScript | `null \| undefined` (any variable, unless strict mode) |
| Go | Zero values + `error` returns |
| **Swift** | `Optional<T>` — absence is encoded in the type system |

In most languages, forgetting to check for null causes a runtime crash (NullPointerException, "cannot read property of undefined", AttributeError). Swift prevents this at compile time. A `String` can never be nil — only a `String?` (an `Optional<String>`) can.

### What Optionals Actually Are

Under the hood, `Optional` is just an enum:

```swift
// This is conceptually what Swift defines:
enum Optional<Wrapped> {
    case some(Wrapped)
    case none
}
```

`String?` is syntactic sugar for `Optional<String>`. When you write `let name: String? = "Ada"`, you're really saying `let name: Optional<String> = .some("Ada")`.

This means optionals are not a special runtime concept — they're a regular enum with compiler support.

```swift
let name: String? = "Ada"      // .some("Ada")
let missing: String? = nil      // .none

// You can't use an optional directly as its wrapped type:
// print(name.count)   // Error: value of optional type 'String?' must be unwrapped
```

### Optional Binding: if let and guard let

**if let** unwraps an optional into a local constant:

```swift
let input: String? = "42"

if let value = input {
    print("Got: \(value)")    // value is a String here, not String?
} else {
    print("No value")
}
```

**Expected Output:**
```
Got: 42
```

**Swift 5.7+ shorthand** — when the unwrapped variable has the same name, you can omit the redundant assignment:

```swift
let username: String? = "ada"

if let username {
    print("Hello, \(username)")   // username is String, not String?
}
```

**Expected Output:**
```
Hello, ada
```

**guard let** is the "early return" form. It unwraps the optional and makes the value available for the rest of the scope. If the optional is nil, you must exit the scope.

```swift
func greet(name: String?) {
    guard let name else {
        print("No name provided")
        return
    }
    // name is a String (not optional) from here on
    print("Hello, \(name)!")
}

greet(name: "Ada")    // Hello, Ada!
greet(name: nil)      // No name provided
```

**Expected Output:**
```
Hello, Ada!
No name provided
```

`guard let` is heavily preferred in Swift for handling preconditions. It keeps the "happy path" un-indented and forces you to handle the nil case explicitly with an early return.

### Optional Chaining

Optional chaining lets you call methods or access properties on an optional. If the optional is nil, the entire chain returns nil.

```swift
let text: String? = "Hello, World"
let count = text?.count           // Int? — some(12) or nil
let upper = text?.uppercased()    // String? — some("HELLO, WORLD") or nil

print(count as Any)  // Optional(12)
print(upper as Any)  // Optional("HELLO, WORLD")
```

**Expected Output:**
```
Optional(12)
Optional("HELLO, WORLD")
```

Chains can go multiple levels deep:

```swift
struct Address {
    var city: String
}

struct Person {
    var address: Address?
}

let person: Person? = Person(address: Address(city: "London"))
let city = person?.address?.city   // String? — "London" or nil if any link is nil

print(city as Any)  // Optional("London")
```

**Expected Output:**
```
Optional("London")
```

This is similar to TypeScript's optional chaining (`person?.address?.city`) but the return type is always optional.

### Nil Coalescing

The nil coalescing operator `??` provides a default value when an optional is nil:

```swift
let saved: String? = nil
let displayName = saved ?? "Anonymous"   // "Anonymous"
print(displayName)
```

**Expected Output:**
```
Anonymous
```

This is like Python's `value or "default"` or TypeScript's `value ?? "default"`, except it's type-safe — the default must match the wrapped type.

### Force Unwrapping

The `!` operator force-unwraps an optional. If it's nil, your program crashes.

```swift
let value: String? = "Hello"
print(value!)   // "Hello" — works because value is not nil

let empty: String? = nil
// print(empty!)   // Fatal error: Unexpectedly found nil while unwrapping an Optional value
```

Force unwrapping is the Swift equivalent of "I promise this isn't null." Use it only when you can prove the value exists. In practice, this means almost never — prefer `if let`, `guard let`, or `??`.

### Implicitly Unwrapped Optionals

Declared with `!` instead of `?`:

```swift
var connection: String! = nil   // String! means "optional, but I'll access it like a non-optional"
// print(connection.count)       // Crash — it's still nil

connection = "established"
print(connection.count)          // 11 — works without unwrapping syntax
```

**Expected Output:**
```
11
```

You'll see these mainly in two places:
1. **IBOutlets** in UIKit (set up by Interface Builder after init)
2. **Two-phase initialization** where a property can't be set in `init` but is guaranteed to exist before use

Treat them as a code smell everywhere else.

### Exercise 2.1: Safe Dictionary Lookup

**Task:** Swift dictionaries return optionals on subscript access (since the key may not exist). Write a function `summarize(scores:)` that takes a `[String: Int]` dictionary and prints each student's grade. If a student has no score, print "No score recorded."

```swift
let scores = ["Alice": 95, "Bob": 82]
// Should handle: summarize(scores: scores) for keys "Alice", "Bob", "Charlie"
```

<details>
<summary>Hint 1</summary>

Dictionary subscript returns `Int?`. Use `if let` or `??` to handle the nil case.
</details>

<details>
<summary>Hint 2</summary>

You can iterate over a list of names and look each one up in the dictionary.
</details>

<details>
<summary>Solution</summary>

```swift
func summarize(scores: [String: Int], students: [String]) {
    for student in students {
        if let score = scores[student] {
            print("\(student): \(score)")
        } else {
            print("\(student): No score recorded")
        }
    }
}

let scores = ["Alice": 95, "Bob": 82]
summarize(scores: scores, students: ["Alice", "Bob", "Charlie"])
```

**Expected Output:**
```
Alice: 95
Bob: 82
Charlie: No score recorded
```

An alternative using nil coalescing:

```swift
func summarize(scores: [String: Int], students: [String]) {
    for student in students {
        let display = scores[student].map(String.init) ?? "No score recorded"
        print("\(student): \(display)")
    }
}
```
</details>

### Exercise 2.2: Chained Optional Access

**Task:** Given this model, write a function that safely extracts the first employee's department name. Return "Unknown" if any part of the chain is nil.

```swift
struct Department {
    var name: String
}

struct Employee {
    var department: Department?
}

struct Company {
    var employees: [Employee]?
}
```

<details>
<summary>Hint</summary>

Use optional chaining with `?.first?.` and nil coalescing `??`.
</details>

<details>
<summary>Solution</summary>

```swift
struct Department {
    var name: String
}

struct Employee {
    var department: Department?
}

struct Company {
    var employees: [Employee]?
}

func firstDepartmentName(of company: Company?) -> String {
    return company?.employees?.first?.department?.name ?? "Unknown"
}

let company = Company(employees: [Employee(department: Department(name: "Engineering"))])
print(firstDepartmentName(of: company))   // Engineering
print(firstDepartmentName(of: nil))        // Unknown
print(firstDepartmentName(of: Company(employees: nil)))  // Unknown
```

**Expected Output:**
```
Engineering
Unknown
Unknown
```
</details>

### Checkpoint 2

Before moving on, make sure you can:
- [ ] Explain why Swift uses optionals instead of null
- [ ] Use `if let` and `guard let` to safely unwrap optionals
- [ ] Use optional chaining to access nested optional properties
- [ ] Use `??` to provide default values
- [ ] Explain why force unwrapping (`!`) should be rare

---

## Section 3: Protocols and Extensions

### Protocols as Interfaces

If you know TypeScript interfaces, Java interfaces, or Go interfaces, you know the basic idea. A protocol defines a contract — a set of methods and properties that a conforming type must implement.

```swift
protocol Describable {
    var description: String { get }
    func summarize() -> String
}
```

Types conform to protocols explicitly (unlike Go's implicit interface satisfaction):

```swift
struct Temperature: Describable {
    var celsius: Double

    var description: String {
        return "\(celsius)°C"
    }

    func summarize() -> String {
        return "Temperature is \(description)"
    }
}

let temp = Temperature(celsius: 22.5)
print(temp.summarize())
```

**Expected Output:**
```
Temperature is 22.5°C
```

**Key difference from Go**: In Go, any type that has the right methods satisfies an interface implicitly. In Swift, you must explicitly declare conformance (`: Describable`). This makes it clear which protocols a type conforms to and enables the compiler to provide better diagnostics.

### Extensions: Adding Behavior to Existing Types

Extensions let you add methods, computed properties, and protocol conformance to any type — even types you didn't write.

```swift
extension Int {
    var isEven: Bool {
        return self % 2 == 0
    }

    func repeated(times: Int) -> [Int] {
        return Array(repeating: self, count: times)
    }
}

print(42.isEven)            // true
print(3.repeated(times: 4)) // [3, 3, 3, 3]
```

**Expected Output:**
```
true
[3, 3, 3, 3]
```

This is similar to Kotlin extension functions or C# extension methods. Python doesn't have an equivalent (monkey-patching is the closest, and it's frowned upon). Go doesn't have this — you'd write a standalone function.

### Protocol Conformance via Extensions

A common Swift pattern: add protocol conformance in an extension to keep your code organized.

```swift
struct User {
    var name: String
    var email: String
}

extension User: CustomStringConvertible {
    var description: String {
        return "\(name) <\(email)>"
    }
}

let user = User(name: "Ada", email: "ada@example.com")
print(user)  // Uses CustomStringConvertible
```

**Expected Output:**
```
Ada <ada@example.com>
```

`CustomStringConvertible` is Swift's equivalent of Python's `__str__`, Java's `toString()`, or Go's `String()` method on the `Stringer` interface.

### Default Implementations

Protocols can provide default implementations via extensions. This is how Swift achieves behavior sharing without inheritance.

```swift
protocol Greetable {
    var name: String { get }
    func greet() -> String
}

extension Greetable {
    func greet() -> String {
        return "Hello, I'm \(name)"
    }
}

struct Student: Greetable {
    var name: String
    // greet() comes from the default implementation
}

struct Teacher: Greetable {
    var name: String

    // Override the default
    func greet() -> String {
        return "Good morning, I'm Professor \(name)"
    }
}

print(Student(name: "Ada").greet())
print(Teacher(name: "Turing").greet())
```

**Expected Output:**
```
Hello, I'm Ada
Good morning, I'm Professor Turing
```

This gives you the "mixin" behavior you might get from abstract classes in Java, default interface methods in Java 8+, or protocol extensions in... well, Swift invented this pattern.

### Protocol-Oriented Design vs Inheritance

In Java, you might build a class hierarchy:

```java
// Java approach
abstract class Animal {
    abstract String sound();
    void describe() { ... }
}
class Dog extends Animal { ... }
class Cat extends Animal { ... }
```

In Swift, prefer protocols and composition:

```swift
protocol SoundMaking {
    func sound() -> String
}

protocol Named {
    var name: String { get }
}

struct Dog: SoundMaking, Named {
    var name: String
    func sound() -> String { "Woof" }
}

struct Cat: SoundMaking, Named {
    var name: String
    func sound() -> String { "Meow" }
}

func describe(_ animal: some SoundMaking & Named) {
    print("\(animal.name) says \(animal.sound())")
}

describe(Dog(name: "Rex"))
describe(Cat(name: "Whiskers"))
```

**Expected Output:**
```
Rex says Woof
Whiskers says Meow
```

Notice `some SoundMaking & Named` — the `some` keyword creates an **opaque type**. The function accepts any single concrete type that conforms to both protocols, but the compiler knows the exact type, enabling optimizations.

**`some` vs `any`:**
- `some Protocol` — opaque type. The compiler knows the concrete type. More efficient, but the function can only work with one concrete type per call site.
- `any Protocol` — existential type. The compiler wraps the value in a box. More flexible (you can put mixed types in an array), but with a small performance cost.

```swift
// Can hold mixed types:
let animals: [any SoundMaking] = [Dog(name: "Rex"), Cat(name: "Whiskers")]

// Cannot hold mixed types (all elements must be the same concrete type):
// let animals: [some SoundMaking] = [Dog(name: "Rex"), Cat(name: "Whiskers")]  // Error
```

If you're coming from TypeScript, `any Protocol` is like a regular interface type, and `some Protocol` is more like a generic with a constraint.

### Common Standard Library Protocols

These are the protocols you'll use constantly:

**Codable** — JSON and property list encoding/decoding:

```swift
struct Post: Codable {
    var title: String
    var body: String
    var published: Bool
}

let post = Post(title: "Hello", body: "World", published: true)
let json = try! JSONEncoder().encode(post)
print(String(data: json, encoding: .utf8)!)

let decoded = try! JSONDecoder().decode(Post.self, from: json)
print(decoded.title)
```

**Expected Output:**
```
{"title":"Hello","body":"World","published":true}
Hello
```

If all of a struct's properties are themselves `Codable`, Swift synthesizes the conformance automatically. No manual `toJSON()` or `fromJSON()` needed — no annotations, no decorators, no code generation. Just add `: Codable`.

**Hashable** — enables use as dictionary keys or in sets:

```swift
struct Coordinate: Hashable {
    var x: Int
    var y: Int
}

var visited: Set<Coordinate> = []
visited.insert(Coordinate(x: 0, y: 0))
visited.insert(Coordinate(x: 1, y: 1))
print(visited.count)  // 2
```

**Expected Output:**
```
2
```

**Equatable** and **Comparable** — equality and ordering:

```swift
struct Version: Comparable {
    var major: Int
    var minor: Int

    static func < (lhs: Version, rhs: Version) -> Bool {
        if lhs.major != rhs.major { return lhs.major < rhs.major }
        return lhs.minor < rhs.minor
    }
}

let versions = [Version(major: 2, minor: 1), Version(major: 1, minor: 9), Version(major: 2, minor: 0)]
print(versions.sorted())
```

**Expected Output:**
```
[Version(major: 1, minor: 9), Version(major: 2, minor: 0), Version(major: 2, minor: 1)]
```

**Identifiable** — gives a stable identity (used heavily in SwiftUI):

```swift
struct Task: Identifiable {
    let id = UUID()
    var title: String
}
```

### Exercise 3.1: Design with Protocols

**Task:** Define a `Storable` protocol that requires a `var key: String { get }` and a `func serialize() -> String`. Create two structs — `Setting` (with `name` and `value` properties) and `Preference` (with `category` and `enabled` properties) — that conform to `Storable`. Write a function `saveAll(_ items: [any Storable])` that prints each item's key and serialized form.

<details>
<summary>Hint 1</summary>

The protocol declaration looks like:
```swift
protocol Storable {
    var key: String { get }
    func serialize() -> String
}
```
</details>

<details>
<summary>Hint 2</summary>

For `serialize()`, return a simple string representation. The function parameter uses `[any Storable]` to accept mixed types.
</details>

<details>
<summary>Solution</summary>

```swift
protocol Storable {
    var key: String { get }
    func serialize() -> String
}

struct Setting: Storable {
    var name: String
    var value: String

    var key: String { name }

    func serialize() -> String {
        return "\(name)=\(value)"
    }
}

struct Preference: Storable {
    var category: String
    var enabled: Bool

    var key: String { category }

    func serialize() -> String {
        return "\(category):\(enabled)"
    }
}

func saveAll(_ items: [any Storable]) {
    for item in items {
        print("[\(item.key)] \(item.serialize())")
    }
}

saveAll([
    Setting(name: "theme", value: "dark"),
    Preference(category: "notifications", enabled: true),
    Setting(name: "language", value: "en")
])
```

**Expected Output:**
```
[theme] theme=dark
[notifications] notifications:true
[language] language=en
```
</details>

### Checkpoint 3

Before moving on, make sure you can:
- [ ] Define a protocol and make types conform to it
- [ ] Add methods and computed properties to existing types with extensions
- [ ] Provide default implementations in protocol extensions
- [ ] Explain the difference between `some` and `any` with protocols
- [ ] Use `Codable` to encode/decode JSON

---

## Section 4: Closures and Higher-Order Functions

### Closure Syntax

Swift closures are anonymous functions, like lambdas in Python, arrow functions in TypeScript/JavaScript, or anonymous functions in Go.

The full closure syntax:

```swift
let add = { (a: Int, b: Int) -> Int in
    return a + b
}
print(add(3, 5))  // 8
```

**Expected Output:**
```
8
```

Swift has aggressive syntax shortening for closures:

```swift
let numbers = [5, 2, 8, 1, 9]

// Full form:
let sorted1 = numbers.sorted(by: { (a: Int, b: Int) -> Bool in return a < b })

// Type inference (types inferred from context):
let sorted2 = numbers.sorted(by: { a, b in return a < b })

// Implicit return (single expression):
let sorted3 = numbers.sorted(by: { a, b in a < b })

// Shorthand argument names ($0, $1, ...):
let sorted4 = numbers.sorted(by: { $0 < $1 })

// Trailing closure syntax (when closure is the last argument):
let sorted5 = numbers.sorted { $0 < $1 }

// Operator as closure (< is a function (Int, Int) -> Bool):
let sorted6 = numbers.sorted(by: <)
```

All six produce the same result: `[1, 2, 5, 8, 9]`.

The typical style in Swift codebases is to use trailing closure syntax with shorthand arguments for short closures, and named parameters for anything longer than one line.

### Trailing Closure Syntax

When a function's last parameter is a closure, you can write the closure after the parentheses:

```swift
func perform(times: Int, action: () -> Void) {
    for _ in 0..<times {
        action()
    }
}

// Instead of:
perform(times: 3, action: { print("Hello") })

// Trailing closure:
perform(times: 3) {
    print("Hello")
}
```

**Expected Output:**
```
Hello
Hello
Hello
```

If the closure is the only argument, you can omit the parentheses entirely:

```swift
let names = ["Charlie", "Alice", "Bob"]
let sorted = names.sorted { $0 < $1 }
print(sorted)
```

**Expected Output:**
```
["Alice", "Bob", "Charlie"]
```

### Capturing Values

Closures capture variables from their surrounding scope by reference (for reference types and `var` variables):

```swift
func makeCounter() -> () -> Int {
    var count = 0
    return {
        count += 1
        return count
    }
}

let counter = makeCounter()
print(counter())  // 1
print(counter())  // 2
print(counter())  // 3
```

**Expected Output:**
```
1
2
3
```

The closure captures `count` and each call mutates the same variable. This works the same as JavaScript closures or Python closures (with the `nonlocal` keyword).

**Capture lists** let you control how values are captured. This becomes critical for memory management (covered in Section 7):

```swift
var x = 10
let closure = { [x] in  // Captures x by VALUE at this moment
    print(x)
}
x = 99
closure()  // 10, not 99
```

**Expected Output:**
```
10
```

Without the capture list, `x` would be captured by reference and print `99`.

### map, filter, reduce, compactMap

These work like their equivalents in other languages:

```swift
let numbers = [1, 2, 3, 4, 5]

// map: transform each element
let doubled = numbers.map { $0 * 2 }
print(doubled)  // [2, 4, 6, 8, 10]

// filter: keep elements matching a condition
let evens = numbers.filter { $0 % 2 == 0 }
print(evens)  // [2, 4]

// reduce: combine all elements into one value
let sum = numbers.reduce(0) { $0 + $1 }
print(sum)  // 15

// reduce with operator shorthand:
let product = numbers.reduce(1, *)
print(product)  // 120
```

**Expected Output:**
```
[2, 4, 6, 8, 10]
[2, 4]
15
120
```

**compactMap** is `map` that also filters out nils — extremely useful:

```swift
let strings = ["1", "two", "3", "four", "5"]
let numbers2 = strings.compactMap { Int($0) }
print(numbers2)  // [1, 3, 5]
```

**Expected Output:**
```
[1, 3, 5]
```

`Int($0)` returns `Int?` (nil if the string isn't a valid number). `compactMap` maps and strips the nils in one step. This is one of the most commonly used functions in Swift.

**flatMap** flattens nested arrays:

```swift
let nested = [[1, 2], [3, 4], [5]]
let flat = nested.flatMap { $0 }
print(flat)  // [1, 2, 3, 4, 5]
```

**Expected Output:**
```
[1, 2, 3, 4, 5]
```

### Escaping vs Non-Escaping Closures

By default, closures passed to functions are **non-escaping** — they're called within the function and don't outlive it. If a closure needs to be stored or called later (like a completion handler), it must be marked `@escaping`:

```swift
var completionHandlers: [() -> Void] = []

func registerHandler(handler: @escaping () -> Void) {
    completionHandlers.append(handler)  // Stored for later — must be @escaping
}

func doWorkNow(work: () -> Void) {
    work()  // Called immediately — non-escaping (default)
}
```

Why does this matter?
- Non-escaping closures can be optimized more aggressively by the compiler.
- Escaping closures require `self.` when accessing instance properties (to make capture explicit).
- This distinction doesn't exist in most other languages — JavaScript, Python, and TypeScript closures can always escape.

```swift
class TaskRunner {
    var name = "Runner"

    func runLater(task: @escaping () -> Void) {
        completionHandlers.append(task)
    }

    func configure() {
        runLater {
            print(self.name)  // Must use 'self.' with @escaping
        }
    }
}
```

### Exercise 4.1: Data Pipeline

**Task:** Given an array of raw strings from a CSV row, parse them into a cleaned-up list of temperatures. Each string might be a valid floating-point number or garbage data. Filter out invalid entries, convert to Celsius (the input is Fahrenheit), and return the results sorted ascending.

```swift
let rawReadings = ["98.6", "invalid", "72.0", "", "32.0", "N/A", "212.0"]
// Expected result: [-17.78, 0.0, 22.22, 37.0, 100.0] (approximately)
```

<details>
<summary>Hint 1</summary>

Use `compactMap` to convert strings to `Double?`, filtering out nils automatically.
</details>

<details>
<summary>Hint 2</summary>

Fahrenheit to Celsius: `(f - 32) * 5 / 9`. Chain `.compactMap`, `.map`, and `.sorted`.
</details>

<details>
<summary>Solution</summary>

```swift
let rawReadings = ["98.6", "invalid", "72.0", "", "32.0", "N/A", "212.0"]

let celsius = rawReadings
    .compactMap { Double($0) }
    .map { ($0 - 32) * 5.0 / 9.0 }
    .sorted()

let rounded = celsius.map { (($0 * 100).rounded() / 100) }
print(rounded)
```

**Expected Output:**
```
[0.0, 22.22, 37.0, 100.0]
```

Note: the exact floating-point output may vary slightly. The key steps: `compactMap` removes non-numeric strings, `map` converts to Celsius, `sorted` orders the results.
</details>

### Checkpoint 4

Before moving on, make sure you can:
- [ ] Write closures using trailing closure syntax
- [ ] Use `$0`, `$1` shorthand argument names
- [ ] Use `map`, `filter`, `reduce`, and `compactMap`
- [ ] Explain the difference between escaping and non-escaping closures
- [ ] Explain what a capture list does

---

## Section 5: Error Handling

### Throwing Functions

Swift uses a `throws` keyword and `do/try/catch` pattern — similar in concept to Java/Python/TypeScript exceptions, but with important differences.

```swift
enum ValidationError: Error {
    case tooShort(minimum: Int)
    case invalidCharacter(Character)
    case empty
}

func validateUsername(_ username: String) throws -> String {
    guard !username.isEmpty else {
        throw ValidationError.empty
    }
    guard username.count >= 3 else {
        throw ValidationError.tooShort(minimum: 3)
    }
    for char in username {
        guard char.isLetter || char.isNumber else {
            throw ValidationError.invalidCharacter(char)
        }
    }
    return username
}
```

Calling a throwing function requires `try` inside a `do/catch` block:

```swift
do {
    let name = try validateUsername("ab")
    print("Valid: \(name)")
} catch ValidationError.tooShort(let minimum) {
    print("Username must be at least \(minimum) characters")
} catch ValidationError.invalidCharacter(let char) {
    print("Invalid character: \(char)")
} catch ValidationError.empty {
    print("Username cannot be empty")
} catch {
    print("Unexpected error: \(error)")
}
```

**Expected Output:**
```
Username must be at least 3 characters
```

**Key difference from other languages:** Swift requires you to mark every call that might throw with `try`. You can't accidentally forget to handle errors — the compiler forces it. This is like Go's explicit error returns, but integrated into the type system.

**Typed throws (Swift 6):** You can specify the exact error type a function throws:

```swift
func validate(_ input: String) throws(ValidationError) -> String {
    guard !input.isEmpty else { throw .empty }
    return input
}
```

With typed throws, `catch` blocks know the exact error type without casting.

### The Result Type

`Result` is an enum that represents either success or failure — similar to Rust's `Result`, Go's `(value, error)` return pattern, or TypeScript's discriminated union approach:

```swift
enum FetchError: Error {
    case networkUnavailable
    case invalidResponse
}

func fetchData(from url: String) -> Result<String, FetchError> {
    if url.isEmpty {
        return .failure(.networkUnavailable)
    }
    return .success("Data from \(url)")
}

let result = fetchData(from: "https://api.example.com")

switch result {
case .success(let data):
    print("Got: \(data)")
case .failure(let error):
    print("Failed: \(error)")
}
```

**Expected Output:**
```
Got: Data from https://api.example.com
```

`Result` is especially useful when you need to store an outcome for later processing or pass it through a callback.

### try? and try!

**`try?`** converts a throwing call to an optional — returns nil on error:

```swift
let result1 = try? validateUsername("Ada")     // Optional("Ada")
let result2 = try? validateUsername("")         // nil

print(result1 as Any)
print(result2 as Any)
```

**Expected Output:**
```
Optional("Ada")
nil
```

**`try!`** force-unwraps — crashes if the function throws:

```swift
let name = try! validateUsername("Ada")   // "Ada" — crashes if this throws
print(name)
```

**Expected Output:**
```
Ada
```

Use `try?` when you don't care about the specific error. Use `try!` only in tests or when failure is genuinely impossible.

### Exercise 5.1: Build an Error-Handling Pipeline

**Task:** Write a `parseConfig` function that takes a string of `key=value` lines and returns a `[String: String]` dictionary. It should throw specific errors for:
- Empty input
- Lines missing the `=` separator
- Duplicate keys

```swift
let input = """
host=localhost
port=8080
debug=true
"""
// parseConfig(input) should return ["host": "localhost", "port": "8080", "debug": "true"]
```

<details>
<summary>Hint 1</summary>

Split the input by newlines with `input.split(separator: "\n")`. For each line, split by `"="`. Use `firstIndex(of:)` and `String` slicing to handle values that contain `=`.
</details>

<details>
<summary>Hint 2</summary>

Define an enum `ConfigError: Error` with cases for each error type. Track seen keys in a Set to detect duplicates.
</details>

<details>
<summary>Solution</summary>

```swift
enum ConfigError: Error {
    case emptyInput
    case missingSeparator(line: String)
    case duplicateKey(key: String)
}

func parseConfig(_ input: String) throws -> [String: String] {
    guard !input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        throw ConfigError.emptyInput
    }

    var result: [String: String] = [:]
    let lines = input.split(separator: "\n", omittingEmptySubsequences: true)

    for line in lines {
        guard let equalsIndex = line.firstIndex(of: "=") else {
            throw ConfigError.missingSeparator(line: String(line))
        }
        let key = String(line[line.startIndex..<equalsIndex])
        let value = String(line[line.index(after: equalsIndex)...])

        guard result[key] == nil else {
            throw ConfigError.duplicateKey(key: key)
        }
        result[key] = value
    }

    return result
}

// Test it:
do {
    let config = try parseConfig("host=localhost\nport=8080\ndebug=true")
    print(config)
} catch {
    print("Error: \(error)")
}

do {
    let _ = try parseConfig("no separator here")
} catch ConfigError.missingSeparator(let line) {
    print("Bad line: \(line)")
} catch {
    print("Error: \(error)")
}
```

**Expected Output:**
```
["debug": "true", "host": "localhost", "port": "8080"]
Bad line: no separator here
```

(Dictionary output order may vary since dictionaries are unordered.)
</details>

### Checkpoint 5

Before moving on, make sure you can:
- [ ] Define custom error types as enums conforming to `Error`
- [ ] Write and call throwing functions with `do/try/catch`
- [ ] Use pattern matching in `catch` blocks
- [ ] Use `try?` and `try!` appropriately
- [ ] Explain when to use `Result` vs throwing functions

---

## Section 6: Concurrency

### async/await Basics

If you've used async/await in JavaScript or Python, the syntax will look familiar. But Swift's concurrency model is fundamentally different — it provides compile-time data-race safety.

```swift
func fetchUser(id: Int) async throws -> String {
    // Simulate network delay
    try await Task.sleep(for: .seconds(1))
    return "User \(id)"
}

func fetchProfile() async throws {
    let user = try await fetchUser(id: 42)
    print(user)
}
```

You can only call `async` functions from `async` contexts. The entry point for async code is a `Task`:

```swift
Task {
    do {
        try await fetchProfile()
    } catch {
        print("Error: \(error)")
    }
}
```

**Expected Output** (after ~1 second):
```
User 42
```

**Key difference from JavaScript:** In JavaScript, async functions return Promises that float freely. In Swift, async work is organized into structured task trees. When a parent task is cancelled, all child tasks are cancelled too. There's no equivalent of an "unhandled promise rejection."

### Tasks and Structured Concurrency

**Task** creates a new unit of async work:

```swift
func fetchMultipleUsers() async throws {
    // Sequential:
    let user1 = try await fetchUser(id: 1)
    let user2 = try await fetchUser(id: 2)
    print("\(user1), \(user2)")
}
```

The above runs sequentially — `user2` waits for `user1`. To run concurrently, use `async let`:

```swift
func fetchMultipleUsersConcurrently() async throws {
    async let user1 = fetchUser(id: 1)
    async let user2 = fetchUser(id: 2)

    // Both requests are in flight simultaneously
    let results = try await (user1, user2)
    print("\(results.0), \(results.1)")
}
```

`async let` is like starting two fetch calls in parallel and awaiting both. Similar to `Promise.all` in JavaScript, but with automatic cancellation if one fails.

**Task groups** let you spawn a dynamic number of concurrent tasks:

```swift
func fetchAllUsers(ids: [Int]) async throws -> [String] {
    try await withThrowingTaskGroup(of: String.self) { group in
        for id in ids {
            group.addTask {
                try await fetchUser(id: id)
            }
        }

        var results: [String] = []
        for try await result in group {
            results.append(result)
        }
        return results
    }
}
```

### Actors and Data Isolation

Actors are reference types (like classes) that protect their mutable state from concurrent access. Think of them like a class with a built-in serial queue — only one caller can access the actor's state at a time.

```swift
actor BankAccount {
    private var balance: Double

    init(balance: Double) {
        self.balance = balance
    }

    func deposit(_ amount: Double) {
        balance += amount
    }

    func withdraw(_ amount: Double) throws -> Double {
        guard balance >= amount else {
            throw BankError.insufficientFunds
        }
        balance -= amount
        return amount
    }

    func getBalance() -> Double {
        return balance
    }
}

enum BankError: Error {
    case insufficientFunds
}
```

Accessing actor properties or methods from outside requires `await`:

```swift
let account = BankAccount(balance: 100)

Task {
    await account.deposit(50)
    let balance = await account.getBalance()
    print("Balance: \(balance)")
}
```

**Expected Output:**
```
Balance: 150.0
```

The `await` is the compiler telling you "this might suspend because another task could be using the actor right now." You cannot accidentally access actor state without going through this synchronization.

**Comparison to other languages:**

| Language | Equivalent |
|----------|-----------|
| Go | Goroutine + channel / mutex (manual) |
| Java | synchronized blocks / ReentrantLock (manual) |
| Erlang/Elixir | GenServer / processes (closest conceptual match) |
| JavaScript | N/A (single-threaded, no shared mutable state) |

### MainActor for UI Work

In iOS/macOS development, UI updates must happen on the main thread. `@MainActor` enforces this at compile time:

```swift
@MainActor
class ViewModel {
    var title: String = ""

    func updateTitle(_ newTitle: String) {
        title = newTitle  // Guaranteed to run on main thread
    }
}
```

If you try to call a `@MainActor` method from a non-main context, the compiler requires `await`:

```swift
let viewModel = ViewModel()

Task {
    // This runs on some background thread
    await viewModel.updateTitle("Hello")  // Compiler ensures main thread via await
}
```

### Sendable

`Sendable` is a protocol marking types that are safe to pass across concurrency boundaries. Value types (structs, enums) are implicitly Sendable. Classes must be carefully designed — immutable classes or actors can be Sendable.

```swift
struct Message: Sendable {
    let text: String
    let timestamp: Date
}

// This works — Message is Sendable, safe to pass between tasks:
let message = Message(text: "Hello", timestamp: Date())
Task {
    print(message.text)
}
```

Swift 6 enforces Sendable checking strictly. The compiler will reject code that passes non-Sendable types across task boundaries. This is Swift's compile-time data-race prevention — no language in widespread use (other than Rust) offers comparable safety.

### Exercise 6.1: Concurrent Data Processing

**Task:** Write an async function that takes an array of URLs (as strings), "fetches" each one concurrently (simulate with a sleep and returning the URL length), and returns the results as a dictionary mapping URL to length.

<details>
<summary>Hint 1</summary>

Use `withThrowingTaskGroup` to process all URLs concurrently. Each task returns a `(String, Int)` tuple.
</details>

<details>
<summary>Hint 2</summary>

The task group's generic parameter should be `(String, Int).self`. Collect results into a dictionary inside the `for try await` loop.
</details>

<details>
<summary>Solution</summary>

```swift
func fetchLengths(urls: [String]) async throws -> [String: Int] {
    try await withThrowingTaskGroup(of: (String, Int).self) { group in
        for url in urls {
            group.addTask {
                // Simulate network work
                try await Task.sleep(for: .milliseconds(100))
                return (url, url.count)
            }
        }

        var results: [String: Int] = [:]
        for try await (url, length) in group {
            results[url] = length
        }
        return results
    }
}

// Test it:
Task {
    let lengths = try await fetchLengths(urls: [
        "https://example.com",
        "https://api.github.com/users",
        "https://swift.org"
    ])
    for (url, length) in lengths.sorted(by: { $0.key < $1.key }) {
        print("\(url): \(length)")
    }
}
```

**Expected Output** (order may vary due to concurrency):
```
https://api.github.com/users: 30
https://example.com: 19
https://swift.org: 17
```
</details>

### Checkpoint 6

Before moving on, make sure you can:
- [ ] Write and call async/await functions
- [ ] Use `async let` for concurrent execution
- [ ] Explain what actors do and why `await` is required to access their state
- [ ] Explain the purpose of `@MainActor`
- [ ] Describe what `Sendable` means and why it matters

---

## Section 7: Memory Management

### How ARC Works

Most languages you've used have a garbage collector — a runtime process that periodically scans for unreachable objects and frees them. Swift doesn't have a garbage collector. Instead, it uses **Automatic Reference Counting (ARC)**.

ARC works like this:
1. Every class instance has a reference count.
2. When you create a new reference to an instance, the count goes up.
3. When a reference goes away (variable goes out of scope, set to nil), the count goes down.
4. When the count hits 0, the instance is deallocated immediately.

```swift
class Document {
    let title: String

    init(title: String) {
        self.title = title
        print("\(title) created")
    }

    deinit {
        print("\(title) deallocated")
    }
}

func example() {
    let doc = Document(title: "Report")  // ref count: 1
    print("Using \(doc.title)")
}  // doc goes out of scope, ref count: 0, deallocated immediately

example()
```

**Expected Output:**
```
Report created
Using Report
Report deallocated
```

**Key differences from garbage collection:**

| | ARC (Swift) | GC (Java, Python, Go, JS) |
|---|---|---|
| When | Immediate, deterministic | Periodic, non-deterministic |
| Overhead | Incrementing/decrementing counters | Periodic pauses for collection |
| `deinit` timing | Predictable | Unpredictable (finalizers) |
| Applies to | Classes only (reference types) | All heap objects |
| Weakness | Cannot automatically detect cycles | Handles cycles automatically |

ARC only applies to **classes** (reference types). Structs, enums, and other value types live on the stack or are embedded inline — they don't need reference counting.

### Strong, Weak, and Unowned References

By default, references are **strong** — they increment the reference count. Swift provides two alternatives:

**`weak`** — doesn't increment the count. Automatically becomes `nil` when the instance is deallocated. Must be `var` and optional.

**`unowned`** — doesn't increment the count. Assumes the instance is always alive. Crashes if accessed after deallocation. Like a non-optional `weak`.

```swift
class Person {
    let name: String
    var pet: Pet?

    init(name: String) { self.name = name }
    deinit { print("\(name) deallocated") }
}

class Pet {
    let name: String
    weak var owner: Person?  // Weak to avoid retain cycle

    init(name: String) { self.name = name }
    deinit { print("\(name) deallocated") }
}

var alice: Person? = Person(name: "Alice")
var cat: Pet? = Pet(name: "Whiskers")

alice?.pet = cat
cat?.owner = alice

alice = nil  // Alice's ref count goes to 0, deallocated
cat = nil    // Whiskers' ref count goes to 0, deallocated
```

**Expected Output:**
```
Alice deallocated
Whiskers deallocated
```

Without `weak`, setting `alice = nil` wouldn't deallocate either object because they'd still be pointing at each other (a retain cycle).

### Retain Cycles and How to Spot Them

A retain cycle happens when two (or more) objects hold strong references to each other, preventing either from being deallocated.

**Classic retain cycle:**

```swift
class Department {
    let name: String
    var manager: Manager?

    init(name: String) { self.name = name }
    deinit { print("Department \(name) deallocated") }
}

class Manager {
    let name: String
    var department: Department?  // Strong reference back — creates a cycle!

    init(name: String) { self.name = name }
    deinit { print("Manager \(name) deallocated") }
}

var dept: Department? = Department(name: "Engineering")
var mgr: Manager? = Manager(name: "Alice")

dept?.manager = mgr
mgr?.department = dept

dept = nil  // Neither object is deallocated!
mgr = nil   // Memory leak — both objects still alive
```

**Expected Output:**
```
(nothing — the deinit methods never run)
```

**Fix:** Make one side of the relationship `weak` or `unowned`:

```swift
class FixedManager {
    let name: String
    weak var department: Department?  // Weak breaks the cycle

    init(name: String) { self.name = name }
    deinit { print("Manager \(name) deallocated") }
}
```

**When to use `weak` vs `unowned`:**
- **`weak`**: The referenced object might be deallocated first. The reference becomes nil. (Parent-child when the child doesn't own the parent.)
- **`unowned`**: The referenced object is guaranteed to live at least as long. Crashes if the assumption is wrong. (A credit card always has an owner that outlives it.)

### Closures and Capture Lists for Breaking Cycles

The most common source of retain cycles in real Swift code is closures. When a closure captures `self` and `self` stores the closure, you get a cycle.

```swift
class NetworkManager {
    var onComplete: (() -> Void)?
    var data: String = ""

    func startRequest() {
        onComplete = {
            // This closure captures self strongly
            print("Got: \(self.data)")
        }
    }

    deinit { print("NetworkManager deallocated") }
}

var manager: NetworkManager? = NetworkManager()
manager?.data = "response"
manager?.startRequest()
manager = nil  // Memory leak — self and onComplete reference each other
```

**Expected Output:**
```
(nothing — deinit never runs)
```

**Fix with a capture list:**

```swift
class FixedNetworkManager {
    var onComplete: (() -> Void)?
    var data: String = ""

    func startRequest() {
        onComplete = { [weak self] in
            guard let self else { return }
            print("Got: \(self.data)")
        }
    }

    deinit { print("FixedNetworkManager deallocated") }
}

var fixedManager: FixedNetworkManager? = FixedNetworkManager()
fixedManager?.data = "response"
fixedManager?.startRequest()
fixedManager = nil
```

**Expected Output:**
```
FixedNetworkManager deallocated
```

The `[weak self]` capture list makes `self` a weak reference inside the closure. `guard let self` unwraps it — if `self` has been deallocated, the closure returns early.

**Rule of thumb:** Any time a closure is stored as a property (escaping closure) and it references `self`, you likely need `[weak self]`.

### Exercise 7.1: Find the Retain Cycle

**Task:** This code has a memory leak. Find the retain cycle and fix it.

```swift
class Timer {
    var interval: Double
    var action: (() -> Void)?

    init(interval: Double) {
        self.interval = interval
    }

    func start() {
        action = {
            print("Tick at \(self.interval)s interval")
        }
    }

    deinit { print("Timer deallocated") }
}

var timer: Timer? = Timer(interval: 1.0)
timer?.start()
timer = nil
// Expected: "Timer deallocated" — but it never prints
```

<details>
<summary>Hint</summary>

The closure stored in `action` captures `self` strongly. `self` holds `action`. Neither can be freed.
</details>

<details>
<summary>Solution</summary>

```swift
class Timer {
    var interval: Double
    var action: (() -> Void)?

    init(interval: Double) {
        self.interval = interval
    }

    func start() {
        action = { [weak self] in
            guard let self else { return }
            print("Tick at \(self.interval)s interval")
        }
    }

    deinit { print("Timer deallocated") }
}

var timer: Timer? = Timer(interval: 1.0)
timer?.start()
timer = nil
```

**Expected Output:**
```
Timer deallocated
```

The `[weak self]` capture list breaks the cycle. When `timer` is set to nil, the `Timer` instance's reference count drops to 0, and it's deallocated.
</details>

### Checkpoint 7

Before moving on, make sure you can:
- [ ] Explain how ARC differs from garbage collection
- [ ] Identify when a retain cycle will occur
- [ ] Use `weak` and `unowned` to break cycles between objects
- [ ] Use `[weak self]` in closures to prevent retain cycles
- [ ] Explain when to use `weak` vs `unowned`

---

## Practice Project

### Project Description

Build a command-line task manager that exercises everything you've learned: structs, enums, optionals, protocols, closures, error handling, and async/await. The project models a simple task processing pipeline.

### Requirements

Build a task processing system that:
- Models tasks with a `Priority` enum and status tracking
- Uses a protocol-based storage system
- Validates tasks with throwing functions
- Processes tasks concurrently using async/await
- Uses proper memory management with an actor-based task queue

### Getting Started

Create a new Swift Playground or a Swift package:

```bash
mkdir TaskProcessor && cd TaskProcessor
swift package init --type executable
```

### Step 1: Define the Models

Start with the data types. Use structs, enums, and protocols.

```swift
// Paste into Sources/main.swift (or a Playground)

import Foundation

enum Priority: Int, Comparable, CustomStringConvertible {
    case low = 0
    case medium = 1
    case high = 2
    case critical = 3

    var description: String {
        switch self {
        case .low: return "Low"
        case .medium: return "Medium"
        case .high: return "High"
        case .critical: return "Critical"
        }
    }

    static func < (lhs: Priority, rhs: Priority) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

enum TaskStatus {
    case pending
    case inProgress
    case completed(result: String)
    case failed(error: String)
}

struct TaskItem: Identifiable {
    let id: UUID
    var title: String
    var priority: Priority
    var status: TaskStatus
    var tags: Set<String>

    init(title: String, priority: Priority, tags: Set<String> = []) {
        self.id = UUID()
        self.title = title
        self.priority = priority
        self.status = .pending
        self.tags = tags
    }
}
```

### Step 2: Define the Storage Protocol and Validation

```swift
protocol TaskStore {
    mutating func add(_ task: TaskItem) throws
    mutating func update(_ task: TaskItem) throws
    func find(id: UUID) -> TaskItem?
    func all() -> [TaskItem]
    func filtered(by predicate: (TaskItem) -> Bool) -> [TaskItem]
}

enum TaskError: Error, CustomStringConvertible {
    case emptyTitle
    case duplicateTitle(String)
    case notFound(UUID)

    var description: String {
        switch self {
        case .emptyTitle:
            return "Task title cannot be empty"
        case .duplicateTitle(let title):
            return "A task with title '\(title)' already exists"
        case .notFound(let id):
            return "No task found with ID \(id)"
        }
    }
}
```

### Step 3: Implement the Store

Now implement `TaskStore`. Try this yourself before looking at the solution.

<details>
<summary>If you need a starting point</summary>

```swift
struct InMemoryStore: TaskStore {
    private var tasks: [UUID: TaskItem] = [:]

    // Implement the four protocol methods.
    // Use throws for add (check empty title, duplicate titles)
    // and update (check task exists).
}
```
</details>

<details>
<summary>Full implementation</summary>

```swift
struct InMemoryStore: TaskStore {
    private var tasks: [UUID: TaskItem] = [:]

    mutating func add(_ task: TaskItem) throws {
        guard !task.title.trimmingCharacters(in: .whitespaces).isEmpty else {
            throw TaskError.emptyTitle
        }
        let isDuplicate = tasks.values.contains { $0.title == task.title }
        guard !isDuplicate else {
            throw TaskError.duplicateTitle(task.title)
        }
        tasks[task.id] = task
    }

    mutating func update(_ task: TaskItem) throws {
        guard tasks[task.id] != nil else {
            throw TaskError.notFound(task.id)
        }
        tasks[task.id] = task
    }

    func find(id: UUID) -> TaskItem? {
        return tasks[id]
    }

    func all() -> [TaskItem] {
        return Array(tasks.values)
    }

    func filtered(by predicate: (TaskItem) -> Bool) -> [TaskItem] {
        return tasks.values.filter(predicate)
    }
}
```
</details>

### Step 4: Add an Actor-Based Processor

```swift
actor TaskProcessor {
    private var store: InMemoryStore

    init(store: InMemoryStore) {
        self.store = store
    }

    func addTask(_ task: TaskItem) throws {
        try store.add(task)
    }

    func process(id: UUID) async throws -> String {
        guard var task = store.find(id: id) else {
            throw TaskError.notFound(id)
        }

        task.status = .inProgress
        try store.update(task)

        // Simulate work proportional to priority
        let duration = Double(4 - task.priority.rawValue) * 0.1
        try await Task.sleep(for: .seconds(duration))

        let result = "Processed '\(task.title)' [\(task.priority)]"
        task.status = .completed(result: result)
        try store.update(task)
        return result
    }

    func processAll() async -> [Result<String, Error>] {
        let pending = store.filtered { task in
            if case .pending = task.status { return true }
            return false
        }.sorted { $0.priority > $1.priority }

        var results: [Result<String, Error>] = []
        await withTaskGroup(of: Result<String, Error>.self) { group in
            for task in pending {
                group.addTask {
                    do {
                        let result = try await self.process(id: task.id)
                        return .success(result)
                    } catch {
                        return .failure(error)
                    }
                }
            }

            for await result in group {
                results.append(result)
            }
        }
        return results
    }

    func allTasks() -> [TaskItem] {
        return store.all()
    }
}
```

### Step 5: Put It Together

```swift
@main
struct App {
    static func main() async {
        let store = InMemoryStore()
        let processor = TaskProcessor(store: store)

        // Add tasks
        let tasks = [
            TaskItem(title: "Deploy to production", priority: .critical, tags: ["ops"]),
            TaskItem(title: "Write unit tests", priority: .high, tags: ["testing"]),
            TaskItem(title: "Update docs", priority: .low, tags: ["docs"]),
            TaskItem(title: "Fix login bug", priority: .high, tags: ["bugs", "auth"]),
            TaskItem(title: "Refactor API layer", priority: .medium, tags: ["backend"]),
        ]

        for task in tasks {
            do {
                try await processor.addTask(task)
                print("Added: \(task.title) [\(task.priority)]")
            } catch {
                print("Failed to add: \(error)")
            }
        }

        // Try adding a duplicate
        do {
            try await processor.addTask(TaskItem(title: "Deploy to production", priority: .low))
            print("Should not reach here")
        } catch {
            print("Expected error: \(error)")
        }

        print("\nProcessing all tasks concurrently...")
        let results = await processor.processAll()

        print("\nResults:")
        for result in results {
            switch result {
            case .success(let message):
                print("  OK: \(message)")
            case .failure(let error):
                print("  FAIL: \(error)")
            }
        }

        // Query by tag
        let allTasks = await processor.allTasks()
        let bugTasks = allTasks.filter { $0.tags.contains("bugs") }
        print("\nBug tasks: \(bugTasks.map(\.title))")
    }
}
```

**Expected Output** (order of processing results may vary due to concurrency):
```
Added: Deploy to production [Critical]
Added: Write unit tests [High]
Added: Update docs [Low]
Added: Fix login bug [High]
Added: Refactor API layer [Medium]
Expected error: A task with title 'Deploy to production' already exists

Processing all tasks concurrently...

Results:
  OK: Processed 'Deploy to production' [Critical]
  OK: Processed 'Write unit tests' [High]
  OK: Processed 'Fix login bug' [High]
  OK: Processed 'Refactor API layer' [Medium]
  OK: Processed 'Update docs' [Low]

Bug tasks: ["Fix login bug"]
```

### What This Project Covers

| Concept | Where it appears |
|---------|-----------------|
| Structs and enums | `TaskItem`, `Priority`, `TaskStatus`, `TaskError` |
| Optionals | `find(id:)` returns `TaskItem?` |
| Protocols | `TaskStore` protocol, `Comparable`, `Identifiable`, `CustomStringConvertible` |
| Closures | `filtered(by:)`, `sorted { }`, `allTasks.filter { }` |
| Error handling | `throws`, `do/try/catch`, `Result` |
| Concurrency | `actor TaskProcessor`, `async/await`, `withTaskGroup` |
| Pattern matching | `if case .pending = task.status` |

### Extending the Project

If you want to go further, try:
- Add a `Codable` conformance to `TaskItem` and save/load tasks to a JSON file
- Add a deadline (`Date?`) to tasks and sort by urgency
- Implement task cancellation using `Task.checkCancellation()`
- Add a `@MainActor` view model that observes the processor and reports progress

---

## Summary

You've covered the concepts that make Swift distinct from other languages you know:

- **Value semantics** — Structs (including Array, String, Dictionary) copy on assignment. This is the default, and classes are the exception.
- **Optionals** — Nil safety enforced by the compiler. `if let`, `guard let`, `??`, and optional chaining handle the absence of a value.
- **Protocols and extensions** — Composition over inheritance. Default implementations provide shared behavior without a class hierarchy.
- **Closures** — Aggressive shorthand syntax, trailing closures, and the escaping/non-escaping distinction.
- **Error handling** — `throws`/`do`/`try`/`catch` with typed errors and the `Result` type.
- **Concurrency** — Structured async/await, actors for thread-safe state, compile-time data-race prevention.
- **ARC** — Deterministic reference counting. Retain cycles from closures are the primary pitfall.

## Next Steps

Now that you have a solid Swift foundation, explore:
- **SwiftUI Fundamentals** — Build declarative user interfaces with Swift
- **iOS App Patterns** — Architecture and design patterns for Swift applications

## Additional Resources

- [The Swift Programming Language](https://docs.swift.org/swift-book/) — Official Swift book (free)
- [Swift.org](https://swift.org) — Language evolution, proposals, and downloads
- [Swift Forums](https://forums.swift.org) — Community discussion and Q&A
- [Swift by Sundell](https://swiftbysundell.com) — Articles and podcasts for Swift developers

---

## Appendix: Quick Reference

### Variable Declaration

```swift
let constant = "immutable"      // Prefer let
var variable = "mutable"        // Use only when needed
let typed: String = "explicit"  // Explicit type annotation
```

### Structs and Classes

```swift
struct ValueType {               // Value semantics (copy on assign)
    var property: String
    mutating func modify() { }   // Must mark mutating
}

class ReferenceType {            // Reference semantics (shared)
    var property: String
    init(property: String) {     // Required initializer
        self.property = property
    }
}
```

### Enums

```swift
enum Simple { case a, b, c }
enum WithValues { case success(String), failure(Error) }
enum WithRaw: String { case north = "N", south = "S" }
```

### Optionals

```swift
var opt: String? = "value"       // Optional String
if let value = opt { }           // Optional binding
guard let value = opt else { return }  // Early return
opt?.count                       // Optional chaining (Int?)
opt ?? "default"                 // Nil coalescing
opt!                             // Force unwrap (crashes if nil)
```

### Protocols and Extensions

```swift
protocol MyProtocol {
    var required: String { get }
    func doSomething()
}

extension MyProtocol {
    func doSomething() { }       // Default implementation
}

extension ExistingType: MyProtocol {
    var required: String { "value" }
}
```

### Closures

```swift
{ (params) -> Return in body }  // Full form
{ params in body }              // Inferred types
{ $0 + $1 }                    // Shorthand arguments
collection.map { $0 * 2 }      // Trailing closure
```

### Error Handling

```swift
enum MyError: Error { case failed }

func risky() throws -> String { throw MyError.failed }

do { try risky() } catch { print(error) }
try? risky()                     // Returns nil on error
try! risky()                     // Crashes on error
```

### Concurrency

```swift
func fetch() async throws -> Data { }  // Async throwing function
let result = try await fetch()          // Call with await
async let a = fetch()                   // Concurrent binding
actor Safe { var state: Int = 0 }       // Actor (thread-safe)
@MainActor class VM { }                 // Main thread bound
```

### Memory Management

```swift
class A { weak var b: B? }      // Weak reference (optional, auto-nil)
class B { unowned var a: A }    // Unowned (non-optional, no retain)

closure = { [weak self] in      // Capture list in closures
    guard let self else { return }
}
```

### Common Higher-Order Functions

```swift
[1, 2, 3].map { $0 * 2 }          // [2, 4, 6]
[1, 2, 3].filter { $0 > 1 }       // [2, 3]
[1, 2, 3].reduce(0, +)            // 6
["1", "a"].compactMap { Int($0) }  // [1]
[[1], [2]].flatMap { $0 }         // [1, 2]
```

### String Interpolation and Multi-line Strings

```swift
let name = "World"
print("Hello, \(name)")            // String interpolation
let multi = """
    Line 1
    Line 2
    """                             // Multi-line string
```
