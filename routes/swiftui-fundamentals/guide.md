---
title: SwiftUI Fundamentals
route_map: /routes/swiftui-fundamentals/map.md
paired_sherpa: /routes/swiftui-fundamentals/sherpa.md
prerequisites:
  - Swift language fundamentals
  - Xcode basics
topics:
  - SwiftUI
  - Declarative UI
  - State Management
  - Navigation
  - View Composition
---

# SwiftUI Fundamentals

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/swiftui-fundamentals/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/swiftui-fundamentals/map.md` for the high-level overview.

## Overview

SwiftUI is Apple's declarative UI framework for building apps across iOS, macOS, watchOS, and tvOS. If you're coming from React, you'll recognize the core idea immediately: you describe what the UI should look like for a given state, and the framework figures out how to update the screen. This guide teaches SwiftUI's building blocks through that lens, using React comparisons wherever they make a concept click faster.

## Learning Objectives

By the end of this guide, you will be able to:
- Build views using SwiftUI's declarative syntax
- Compose UIs from small, reusable view components
- Manage state with @State, @Binding, @Observable, and @Environment
- Build navigation flows with NavigationStack, sheets, and tabs
- Display dynamic lists with List and ForEach
- Style views with modifiers and understand modifier ordering
- Handle user input with forms, text fields, pickers, and validation
- Preview and iterate on UI with Xcode's canvas

## Prerequisites

Before starting, you should have:
- Swift language fundamentals (variables, types, structs, protocols, closures)
- Xcode installed and a basic understanding of how to create a project
- Experience with React or another component-based UI framework (the comparisons assume React knowledge)

## Setup

Create a new Xcode project:

1. Open Xcode
2. File > New > Project
3. Choose **App** under iOS
4. Set Product Name to `SwiftUIGuide`
5. Set Interface to **SwiftUI**
6. Set Language to **Swift**
7. Click Create

You should see a starter project with `ContentView.swift` containing a "Hello, world!" view and a live preview canvas on the right side of the editor.

**Verify your setup**: Click the Resume button in the canvas (or press Cmd+Option+P). You should see a phone simulator displaying "Hello, world!" with a globe icon above it.

Throughout this guide, you can paste code examples directly into `ContentView.swift` to try them out. Each section builds concepts independently, so you can replace `ContentView`'s body with any example.

---

## Section 1: SwiftUI Mental Model

### Declarative vs Imperative

In UIKit (Apple's older framework), you write imperative code: create a label, set its text, add it to a view, then later find it again and update its text when state changes. This is like working with the DOM directly in JavaScript.

SwiftUI works like React. You declare what the UI should look like for the current state, and the framework handles creating, updating, and destroying actual screen elements.

**React:**
```jsx
function Greeting({ name }) {
  return <h1>Hello, {name}!</h1>;
}
```

**SwiftUI:**
```swift
struct Greeting: View {
    let name: String

    var body: some View {
        Text("Hello, \(name)!")
            .font(.largeTitle)
    }
}
```

The mapping is straightforward:

| React | SwiftUI |
|-------|---------|
| Function component | Struct conforming to `View` |
| JSX return value | `body` property |
| Props | Init parameters (stored properties) |
| `<Component prop={value} />` | `Component(prop: value)` |

### Views Are Structs

In React, components are functions. In SwiftUI, views are structs that conform to the `View` protocol. This means they're value types — cheap to create and copy. SwiftUI recreates your view structs frequently (on every state change), and the framework diffs the result to update only what changed on screen.

The `View` protocol requires one thing: a computed property called `body` that returns `some View`.

```swift
struct ProfileHeader: View {
    let username: String
    let postCount: Int

    var body: some View {
        VStack {
            Text(username)
                .font(.title)
            Text("\(postCount) posts")
                .foregroundStyle(.secondary)
        }
    }
}
```

Think of `body` like the JSX you return from a React component. It's called every time SwiftUI needs to know what this view looks like.

### View Builders

The `body` property uses a special Swift feature called a result builder (specifically `@ViewBuilder`). This is what lets you list multiple views inside `body` or inside containers like `VStack` without needing commas, return statements, or array syntax.

```swift
var body: some View {
    VStack {
        Text("First")    // No comma, no return
        Text("Second")
        Text("Third")
    }
}
```

If you're coming from React, think of this as JSX fragments — you're composing a tree of views, and the framework handles the rest.

View builders also support `if`/`else` and `if let` for conditional rendering (like ternaries or `&&` in JSX):

```swift
var body: some View {
    VStack {
        Text("Welcome")
        if isLoggedIn {
            Text("Hello, \(username)!")
        } else {
            Text("Please sign in")
        }
    }
}
```

### Xcode Canvas and Previews

Xcode's canvas is SwiftUI's equivalent of a hot-reloading dev server. It shows a live preview of your view as you type. You define previews using the `#Preview` macro:

```swift
#Preview {
    Greeting(name: "World")
}
```

This is conceptually like Storybook stories in the React ecosystem — isolated, configurable previews of your components. You can have multiple previews in one file:

```swift
#Preview("Short Name") {
    Greeting(name: "Jo")
}

#Preview("Long Name") {
    Greeting(name: "Alexander Hamilton")
}
```

**Canvas shortcuts:**
- **Cmd+Option+P**: Resume/refresh the canvas
- **Cmd+Option+Enter**: Toggle canvas visibility

### Checkpoint 1

Before moving on, make sure you understand:
- [ ] SwiftUI is declarative like React — you describe UI for a given state
- [ ] Views are structs with a `body` property (not classes, not functions)
- [ ] View builders let you list child views without commas or returns
- [ ] `#Preview` gives you hot-reload-style previews in Xcode's canvas

---

## Section 2: Built-in Views and Modifiers

### Common Views

SwiftUI provides a library of built-in views. Here are the ones you'll use constantly:

```swift
struct CommonViews: View {
    @State private var name = ""
    @State private var isOn = false

    var body: some View {
        VStack(spacing: 16) {
            // Static text
            Text("Hello, SwiftUI!")
                .font(.title)

            // System icon (SF Symbols — Apple's built-in icon library)
            Image(systemName: "star.fill")
                .foregroundStyle(.yellow)
                .font(.largeTitle)

            // Button with action closure
            Button("Tap Me") {
                print("Button tapped")
            }

            // Text input
            TextField("Enter your name", text: $name)
                .textFieldStyle(.roundedBorder)

            // Toggle switch
            Toggle("Enable notifications", isOn: $isOn)
        }
        .padding()
    }
}
```

**What this looks like**: A vertically stacked column showing a title, a yellow star icon, a tappable button, a text field with placeholder text, and a toggle switch. Everything has 16 points of spacing between items and padding around the edges.

Don't worry about `@State` and the `$` prefix yet — we'll cover those in Section 3. For now, just know they're how SwiftUI handles mutable values (like `useState` in React).

**SF Symbols**: The `Image(systemName:)` initializer uses Apple's built-in icon library. There are thousands of icons available. Download the SF Symbols app from Apple's developer site to browse them — it's like having Material Icons or Font Awesome built into the platform.

### View Modifiers

In React, you style components with CSS (classes, inline styles, or CSS-in-JS). SwiftUI has no CSS. Instead, you chain **view modifiers** — methods that wrap a view with additional behavior or styling.

```swift
Text("Styled Text")
    .font(.headline)
    .foregroundStyle(.blue)
    .padding()
    .background(.yellow)
    .clipShape(RoundedRectangle(cornerRadius: 8))
```

**What this looks like**: Blue text on a yellow rounded-rectangle background, with padding between the text and the background edge.

Each modifier returns a new view that wraps the original. Think of it like nesting components:

```jsx
{/* Conceptual React equivalent */}
<ClipShape shape="roundedRect">
  <Background color="yellow">
    <Padding>
      <ForegroundStyle color="blue">
        <Font size="headline">
          <Text>Styled Text</Text>
        </Font>
      </ForegroundStyle>
    </Padding>
  </Background>
</ClipShape>
```

### Modifier Order Matters

This is the single biggest "gotcha" coming from CSS. In CSS, order rarely matters — `padding` and `background-color` apply to the same box regardless of declaration order. In SwiftUI, each modifier wraps the view in a new layer, so order changes the result.

```swift
// Padding THEN background: red fills behind text AND padding
Text("Option A")
    .padding()
    .background(.red)

// Background THEN padding: red fills behind text only, padding is transparent
Text("Option B")
    .background(.red)
    .padding()
```

**What these look like**: "Option A" has a red rectangle that extends around the text with padding included. "Option B" has a tight red rectangle just behind the text, with transparent space around it.

A good mental model: read modifiers top to bottom as "take this view, then wrap it with X." Padding first creates a bigger view, then background colors that bigger view.

### Stacking Views

SwiftUI uses three stack types to arrange views, similar to flexbox in CSS:

```swift
struct StackExamples: View {
    var body: some View {
        VStack(spacing: 20) {
            // Horizontal stack (flexbox row)
            HStack {
                Image(systemName: "person.fill")
                Text("Alice")
                Spacer()
                Text("Online")
                    .foregroundStyle(.green)
            }
            .padding()
            .background(.gray.opacity(0.1))
            .clipShape(RoundedRectangle(cornerRadius: 8))

            // Vertical stack (flexbox column) — already the outer container

            // Layered stack (positioned/absolute)
            ZStack {
                Circle()
                    .fill(.blue)
                    .frame(width: 80, height: 80)
                Text("AB")
                    .foregroundStyle(.white)
                    .font(.title)
            }
        }
        .padding()
    }
}
```

**What this looks like**: A vertical column containing (1) a row with a person icon, "Alice" on the left, and "Online" in green pushed to the right, all on a light gray rounded background; and (2) a blue circle with white "AB" text centered on top of it.

| SwiftUI | CSS Equivalent | Purpose |
|---------|---------------|---------|
| `VStack` | `flex-direction: column` | Stack children vertically |
| `HStack` | `flex-direction: row` | Stack children horizontally |
| `ZStack` | `position: relative` + `absolute` | Layer children on top of each other |
| `Spacer()` | `flex-grow: 1` | Push content apart |

### Frame and Spacer

`Spacer()` expands to fill available space. `frame()` sets explicit dimensions.

```swift
struct LayoutExample: View {
    var body: some View {
        VStack {
            // Fixed-height header
            Text("Header")
                .frame(maxWidth: .infinity)
                .padding()
                .background(.blue)
                .foregroundStyle(.white)

            // Content fills remaining space
            Spacer()

            Text("I'm in the middle")

            Spacer()

            // Fixed-height footer
            Text("Footer")
                .frame(maxWidth: .infinity)
                .padding()
                .background(.gray.opacity(0.2))
        }
    }
}
```

**What this looks like**: A blue header bar at the top, "I'm in the middle" centered vertically, and a gray footer bar at the bottom.

`frame(maxWidth: .infinity)` makes a view stretch to fill its container's width — like `width: 100%` in CSS.

### Exercise 2.1: Build a Profile Card

**Task:** Build a profile card that shows a colored circle with initials (like an avatar), a name, a subtitle, and a button. Use `VStack`, `ZStack`, `HStack`, modifiers for styling, and `Spacer` for layout.

It should look roughly like this:
```
┌────────────────────────────┐
│         [JD]               │
│       John Doe             │
│    iOS Developer           │
│                            │
│     [ Follow ]             │
└────────────────────────────┘
```

<details>
<summary>Hint 1: Structure</summary>

Use a `VStack` for the overall layout. The avatar is a `ZStack` with a `Circle` and `Text`.
</details>

<details>
<summary>Hint 2: Styling the avatar</summary>

```swift
ZStack {
    Circle()
        .fill(.blue)
        .frame(width: 80, height: 80)
    Text("JD")
        .foregroundStyle(.white)
        .font(.title)
}
```
</details>

<details>
<summary>Solution</summary>

```swift
struct ProfileCard: View {
    var body: some View {
        VStack(spacing: 12) {
            ZStack {
                Circle()
                    .fill(.blue)
                    .frame(width: 80, height: 80)
                Text("JD")
                    .foregroundStyle(.white)
                    .font(.title.bold())
            }

            Text("John Doe")
                .font(.title2)

            Text("iOS Developer")
                .foregroundStyle(.secondary)

            Button("Follow") {
                print("Followed!")
            }
            .buttonStyle(.borderedProminent)
        }
        .padding(24)
        .background(.gray.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }
}

#Preview {
    ProfileCard()
}
```

**What this looks like**: A rounded card with a blue circle avatar showing "JD", the name "John Doe" below it, "iOS Developer" in gray, and a blue "Follow" button at the bottom.
</details>

### Checkpoint 2

Before moving on, make sure you can:
- [ ] Use Text, Image, Button, TextField, and Toggle
- [ ] Apply view modifiers and understand that order matters
- [ ] Lay out views with VStack, HStack, ZStack, and Spacer
- [ ] Use `frame(maxWidth: .infinity)` to fill available width
- [ ] Explain why `.padding().background(.red)` differs from `.background(.red).padding()`

---

## Section 3: State Management

This is where SwiftUI gets interesting — and where React knowledge pays off the most.

### @State: Local View State

`@State` is SwiftUI's `useState`. It's owned by a single view, and when it changes, SwiftUI re-renders that view.

**React:**
```jsx
function Counter() {
  const [count, setCount] = useState(0);
  return (
    <button onClick={() => setCount(count + 1)}>
      Count: {count}
    </button>
  );
}
```

**SwiftUI:**
```swift
struct Counter: View {
    @State private var count = 0

    var body: some View {
        Button("Count: \(count)") {
            count += 1
        }
        .font(.title)
    }
}
```

Key differences from `useState`:
- No setter function needed — you mutate the property directly (`count += 1`)
- Always mark `@State` properties as `private` — they're owned by this view
- The initial value is set once. If a parent passes a different value later, it won't reset the state (same as `useState`'s initial value in React)

### @Binding: Two-Way Props

In React, if a child needs to modify parent state, you pass a callback prop:

```jsx
function ToggleButton({ isOn, setIsOn }) {
  return <button onClick={() => setIsOn(!isOn)}>{isOn ? "ON" : "OFF"}</button>;
}

function Parent() {
  const [isOn, setIsOn] = useState(false);
  return <ToggleButton isOn={isOn} setIsOn={setIsOn} />;
}
```

SwiftUI bundles the value and its setter together as a `Binding`. The `$` prefix creates a binding from a `@State` property:

```swift
struct ToggleButton: View {
    @Binding var isOn: Bool

    var body: some View {
        Button(isOn ? "ON" : "OFF") {
            isOn.toggle()
        }
        .foregroundStyle(isOn ? .green : .red)
        .font(.title)
    }
}

struct ParentView: View {
    @State private var isOn = false

    var body: some View {
        VStack(spacing: 20) {
            Text("The switch is \(isOn ? "ON" : "OFF")")
            ToggleButton(isOn: $isOn)  // $ creates a Binding
        }
    }
}
```

**What this looks like**: Text showing "The switch is OFF" and a red "OFF" button. Tapping the button changes both to show "ON" in green.

The `$` prefix is doing the same thing as passing both `value` and `setValue` in React, just in one token. `$isOn` gives you a `Binding<Bool>` — a reference that can both read and write the original `@State` value.

### @Observable: Shared State

When state needs to be shared across multiple views that aren't in a direct parent-child relationship, you use `@Observable` classes. This is conceptually similar to using React context with a state object.

```swift
@Observable
class TaskStore {
    var tasks: [String] = ["Buy groceries", "Walk the dog"]
    var filter: String = ""

    var filteredTasks: [String] {
        if filter.isEmpty { return tasks }
        return tasks.filter { $0.localizedCaseInsensitiveContains(filter) }
    }

    func add(_ task: String) {
        tasks.append(task)
    }

    func remove(at offsets: IndexSet) {
        tasks.remove(atOffsets: offsets)
    }
}
```

Pass it to views that need it. SwiftUI tracks which properties each view actually reads, and only re-renders views that use properties that changed. This is more fine-grained than React context, which re-renders all consumers on any change.

```swift
struct TaskListView: View {
    var store: TaskStore
    @State private var newTask = ""

    var body: some View {
        VStack {
            HStack {
                TextField("New task", text: $newTask)
                    .textFieldStyle(.roundedBorder)
                Button("Add") {
                    store.add(newTask)
                    newTask = ""
                }
            }
            .padding()

            List {
                ForEach(store.filteredTasks, id: \.self) { task in
                    Text(task)
                }
                .onDelete(perform: store.remove)
            }
        }
    }
}

struct TaskApp: View {
    @State private var store = TaskStore()

    var body: some View {
        TaskListView(store: store)
    }
}
```

**What this looks like**: A text field with an "Add" button at the top, and a list of tasks below. You can type a task, tap Add, and it appears in the list. Swiping left on a task deletes it.

Notice: the root view owns the `@Observable` store with `@State`, and passes it to children as a regular parameter. Children that modify the store's properties trigger re-renders automatically.

### @Environment: System Values

`@Environment` reads system-provided values — like color scheme, locale, accessibility settings, or the dismiss action for sheets. It's similar to using `useContext` for framework-provided contexts in React.

```swift
struct ThemeAwareCard: View {
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        Text("Current theme: \(colorScheme == .dark ? "Dark" : "Light")")
            .padding()
            .background(colorScheme == .dark ? .gray : .yellow)
            .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}
```

You can also inject your own `@Observable` objects into the environment, making them available to any descendant view without passing them through every intermediate view (just like React's `Provider` pattern):

```swift
// Inject into environment
struct MyApp: View {
    @State private var store = TaskStore()

    var body: some View {
        TaskListView()
            .environment(store)  // Any descendant can access this
    }
}

// Read from environment
struct TaskListView: View {
    @Environment(TaskStore.self) private var store

    var body: some View {
        List(store.tasks, id: \.self) { task in
            Text(task)
        }
    }
}
```

### Side Effects: .task, .onAppear, .onChange

React has `useEffect`. SwiftUI splits that concept into more specific modifiers:

| React | SwiftUI | When it runs |
|-------|---------|-------------|
| `useEffect(() => { ... }, [])` | `.onAppear { }` | View appears on screen |
| `useEffect(() => { ... }, [])` with async | `.task { }` | View appears (supports async/await, auto-cancels) |
| `useEffect(() => { ... }, [dep])` | `.onChange(of: dep) { }` | A value changes |

```swift
struct UserProfile: View {
    let userId: String
    @State private var user: User?

    var body: some View {
        Group {
            if let user {
                Text("Hello, \(user.name)")
            } else {
                ProgressView("Loading...")
            }
        }
        .task {
            // Runs when view appears, auto-cancels when view disappears
            user = await fetchUser(userId)
        }
    }
}
```

Prefer `.task` over `.onAppear` for async work — it automatically cancels when the view disappears, preventing the SwiftUI equivalent of updating state on an unmounted component.

### When to Use Each

| Pattern | React Equivalent | Use When |
|---------|-----------------|----------|
| `@State` | `useState` | Local state owned by one view |
| `@Binding` | Prop + callback | Child needs to read and write parent's state |
| `@Observable` class | Context + state | State shared across multiple views |
| `@Environment(\.key)` | `useContext` for framework values | Reading system settings (color scheme, locale, etc.) |
| `@Environment(MyType.self)` | `useContext` for your own contexts | Injecting shared objects without prop drilling |

### Exercise 3.1: Build a Temperature Converter

**Task:** Build a view with a `TextField` for Fahrenheit input, and display the Celsius conversion below it. The conversion should update as you type.

<details>
<summary>Hint 1: State setup</summary>

You need a `@State` string for the text field input. Convert it to a `Double` for the calculation.
</details>

<details>
<summary>Hint 2: Conversion formula</summary>

Celsius = (Fahrenheit - 32) * 5 / 9. Use `Double(string)` to convert, which returns an optional.
</details>

<details>
<summary>Solution</summary>

```swift
struct TemperatureConverter: View {
    @State private var fahrenheitText = ""

    private var celsius: Double? {
        guard let f = Double(fahrenheitText) else { return nil }
        return (f - 32) * 5.0 / 9.0
    }

    var body: some View {
        VStack(spacing: 20) {
            TextField("Fahrenheit", text: $fahrenheitText)
                .textFieldStyle(.roundedBorder)
                .keyboardType(.decimalPad)

            if let celsius {
                Text("\(celsius, specifier: "%.1f")° Celsius")
                    .font(.title2)
            } else if !fahrenheitText.isEmpty {
                Text("Enter a valid number")
                    .foregroundStyle(.red)
            }
        }
        .padding()
    }
}

#Preview {
    TemperatureConverter()
}
```

**What this looks like**: A text field labeled "Fahrenheit". As you type "72", the text "22.2° Celsius" appears below. If you type something non-numeric, a red error message shows.
</details>

### Exercise 3.2: Extract a Reusable Component with @Binding

**Task:** Extract the "toggle button" pattern into a reusable component. Create a `LabeledSwitch` view that takes a `label: String` and a `@Binding var isOn: Bool`. Then use two of them in a parent view with independent state.

<details>
<summary>Hint</summary>

The child view declares `@Binding var isOn: Bool`. The parent uses `@State` for each value and passes `$value` to create the binding.
</details>

<details>
<summary>Solution</summary>

```swift
struct LabeledSwitch: View {
    let label: String
    @Binding var isOn: Bool

    var body: some View {
        HStack {
            Text(label)
            Spacer()
            Button(isOn ? "ON" : "OFF") {
                isOn.toggle()
            }
            .foregroundStyle(isOn ? .green : .red)
        }
        .padding()
        .background(.gray.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

struct SettingsView: View {
    @State private var darkMode = false
    @State private var notifications = true

    var body: some View {
        VStack(spacing: 12) {
            LabeledSwitch(label: "Dark Mode", isOn: $darkMode)
            LabeledSwitch(label: "Notifications", isOn: $notifications)

            Text("Dark: \(darkMode), Notifications: \(notifications)")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding()
    }
}

#Preview {
    SettingsView()
}
```

**What this looks like**: Two rows, each with a label on the left and an ON/OFF button on the right. Tapping each button toggles independently. A debug line at the bottom shows both states.
</details>

### Checkpoint 3

Before moving on, make sure you can:
- [ ] Use `@State` for local mutable state
- [ ] Use `@Binding` (with `$` prefix) to pass mutable state to child views
- [ ] Create an `@Observable` class for shared state
- [ ] Read system values with `@Environment`
- [ ] Explain why `.task` is preferred over `.onAppear` for async work
- [ ] Map each SwiftUI state tool to its React equivalent

---

## Section 4: Lists and Dynamic Content

### List and ForEach

React renders arrays of elements with `.map()`. SwiftUI uses `List` and `ForEach`.

**React:**
```jsx
function FruitList({ fruits }) {
  return (
    <ul>
      {fruits.map(fruit => (
        <li key={fruit.id}>{fruit.name}</li>
      ))}
    </ul>
  );
}
```

**SwiftUI:**
```swift
struct FruitList: View {
    let fruits: [Fruit]

    var body: some View {
        List(fruits) { fruit in
            Text(fruit.name)
        }
    }
}
```

`List` provides native platform scrolling, cell styling, swipe actions, and separators for free. It's a full-featured table view, not just a styled `<ul>`.

### The Identifiable Protocol

React uses `key` props. SwiftUI uses the `Identifiable` protocol. Any struct or class with an `id` property conforms to `Identifiable`:

```swift
struct Fruit: Identifiable {
    let id = UUID()
    let name: String
    let emoji: String
}

struct FruitList: View {
    let fruits = [
        Fruit(name: "Apple", emoji: "🍎"),
        Fruit(name: "Banana", emoji: "🍌"),
        Fruit(name: "Cherry", emoji: "🍒"),
    ]

    var body: some View {
        List(fruits) { fruit in
            HStack {
                Text(fruit.emoji)
                Text(fruit.name)
            }
        }
    }
}
```

**What this looks like**: A scrollable list with three rows, each showing an emoji and fruit name side by side. Rows have standard iOS list styling with separators between them.

If your data doesn't conform to `Identifiable`, you can specify a key path:

```swift
// Using a property as the identifier (like key={item.name} in React)
List(names, id: \.self) { name in
    Text(name)
}
```

### ForEach Inside Other Containers

`List` is a container with built-in scrolling and styling. `ForEach` is just the loop — use it when you need dynamic content inside other containers:

```swift
VStack {
    ForEach(items) { item in
        Text(item.name)
    }
}
```

This is the same distinction as using `.map()` inside a `<div>` vs inside a `<ul>` in React.

### Swipe Actions

```swift
struct TaskList: View {
    @State private var tasks = ["Buy groceries", "Walk the dog", "Write code"]

    var body: some View {
        List {
            ForEach(tasks, id: \.self) { task in
                Text(task)
                    .swipeActions(edge: .trailing) {
                        Button(role: .destructive) {
                            tasks.removeAll { $0 == task }
                        } label: {
                            Label("Delete", systemImage: "trash")
                        }
                    }
                    .swipeActions(edge: .leading) {
                        Button {
                            print("Pinned \(task)")
                        } label: {
                            Label("Pin", systemImage: "pin")
                        }
                        .tint(.yellow)
                    }
            }
        }
    }
}
```

**What this looks like**: A list of tasks. Swiping left reveals a red "Delete" button. Swiping right reveals a yellow "Pin" button.

### Sections and List Styles

```swift
struct GroupedList: View {
    var body: some View {
        List {
            Section("Fruits") {
                Text("Apple")
                Text("Banana")
                Text("Cherry")
            }

            Section("Vegetables") {
                Text("Carrot")
                Text("Broccoli")
                Text("Spinach")
            }
        }
        .listStyle(.insetGrouped)
    }
}
```

**What this looks like**: Two grouped sections with rounded corners and gray headers, styled like the iOS Settings app. "Fruits" header above three items, then "Vegetables" header above three items.

Available list styles: `.automatic`, `.plain`, `.insetGrouped`, `.grouped`, `.sidebar`.

### Pull to Refresh

```swift
struct RefreshableList: View {
    @State private var items = ["Item 1", "Item 2", "Item 3"]

    var body: some View {
        List(items, id: \.self) { item in
            Text(item)
        }
        .refreshable {
            // This closure supports async/await
            await loadData()
        }
    }

    func loadData() async {
        // Simulate network delay
        try? await Task.sleep(for: .seconds(1))
        items.append("Item \(items.count + 1)")
    }
}
```

**What this looks like**: A standard list. Pulling down from the top shows a spinning activity indicator and adds a new item when the refresh completes.

### Exercise 4.1: Build a Contact List

**Task:** Create a `Contact` struct (with `id`, `name`, and `email`) and display a list of contacts. Each row should show the first letter of the name in a colored circle, the name, and the email. Add swipe-to-delete.

<details>
<summary>Hint 1: The Contact struct</summary>

```swift
struct Contact: Identifiable {
    let id = UUID()
    let name: String
    let email: String
}
```
</details>

<details>
<summary>Hint 2: Getting the first letter</summary>

`String(contact.name.prefix(1))` gives you the first character as a string.
</details>

<details>
<summary>Solution</summary>

```swift
struct Contact: Identifiable {
    let id = UUID()
    let name: String
    let email: String
}

struct ContactList: View {
    @State private var contacts = [
        Contact(name: "Alice Johnson", email: "alice@example.com"),
        Contact(name: "Bob Smith", email: "bob@example.com"),
        Contact(name: "Carol Williams", email: "carol@example.com"),
        Contact(name: "Dave Brown", email: "dave@example.com"),
    ]

    var body: some View {
        List {
            ForEach(contacts) { contact in
                HStack(spacing: 12) {
                    ZStack {
                        Circle()
                            .fill(.blue)
                            .frame(width: 40, height: 40)
                        Text(String(contact.name.prefix(1)))
                            .foregroundStyle(.white)
                            .font(.headline)
                    }

                    VStack(alignment: .leading) {
                        Text(contact.name)
                            .font(.headline)
                        Text(contact.email)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .onDelete { offsets in
                contacts.remove(atOffsets: offsets)
            }
        }
    }
}

#Preview {
    ContactList()
}
```

**What this looks like**: A list where each row has a blue circle with the contact's initial, their name in bold, and their email in gray below. Swiping left reveals a delete button.
</details>

### Checkpoint 4

Before moving on, make sure you can:
- [ ] Use `List` and `ForEach` to display dynamic data
- [ ] Make your data types conform to `Identifiable`
- [ ] Add swipe actions to list rows
- [ ] Use `Section` to group list content
- [ ] Add pull-to-refresh with `.refreshable`

---

## Section 5: Navigation

### NavigationStack

`NavigationStack` is SwiftUI's navigation container. It manages a stack of views — push a view to go deeper, pop to go back. If you've used React Router, think of `NavigationStack` as `<BrowserRouter>` and `NavigationLink` as `<Link>`.

```swift
struct ContentView: View {
    var body: some View {
        NavigationStack {
            List {
                NavigationLink("Settings") {
                    Text("Settings Screen")
                        .font(.largeTitle)
                }

                NavigationLink("Profile") {
                    Text("Profile Screen")
                        .font(.largeTitle)
                }
            }
            .navigationTitle("Home")
        }
    }
}
```

**What this looks like**: A list with "Home" as a large title at the top. Each row has a disclosure indicator (chevron) on the right. Tapping "Settings" pushes a new screen with "Settings Screen" text and a back button.

Key points:
- `NavigationStack` wraps the root view — only use it once at the top level
- `.navigationTitle()` goes on the view inside the stack, not on `NavigationStack` itself
- Child views automatically get a back button

### NavigationLink with Data

For real apps, you usually navigate to a detail view with some data:

```swift
struct Recipe: Identifiable {
    let id = UUID()
    let name: String
    let ingredients: [String]
    let instructions: String
}

struct RecipeList: View {
    let recipes: [Recipe]

    var body: some View {
        NavigationStack {
            List(recipes) { recipe in
                NavigationLink(recipe.name) {
                    RecipeDetail(recipe: recipe)
                }
            }
            .navigationTitle("Recipes")
        }
    }
}

struct RecipeDetail: View {
    let recipe: Recipe

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Ingredients")
                    .font(.headline)
                ForEach(recipe.ingredients, id: \.self) { ingredient in
                    Text("• \(ingredient)")
                }

                Text("Instructions")
                    .font(.headline)
                Text(recipe.instructions)
            }
            .padding()
        }
        .navigationTitle(recipe.name)
    }
}
```

### Programmatic Navigation

Sometimes you need to navigate in response to something other than a tap — like after a form submission. Use navigation paths:

```swift
struct ContentView: View {
    @State private var path = NavigationPath()

    var body: some View {
        NavigationStack(path: $path) {
            VStack(spacing: 20) {
                Button("Go to Page A") {
                    path.append("pageA")
                }
                Button("Go to Page B") {
                    path.append("pageB")
                }
                Button("Go Deep (A then B)") {
                    path.append("pageA")
                    path.append("pageB")
                }
            }
            .navigationTitle("Home")
            .navigationDestination(for: String.self) { value in
                Text("You're on: \(value)")
                    .font(.largeTitle)
                    .navigationTitle(value)
            }
        }
    }
}
```

**What this looks like**: Three buttons. "Go to Page A" pushes one screen. "Go Deep (A then B)" pushes two screens at once, so you navigate back through both.

`.navigationDestination(for:)` maps a data type to a destination view — similar to defining routes in React Router.

### Sheets and Alerts

Sheets are modal views that slide up from the bottom. Alerts are system dialogs. Both are controlled by boolean state.

```swift
struct ModalExamples: View {
    @State private var showSheet = false
    @State private var showAlert = false

    var body: some View {
        VStack(spacing: 20) {
            Button("Show Sheet") {
                showSheet = true
            }

            Button("Show Alert") {
                showAlert = true
            }
        }
        .sheet(isPresented: $showSheet) {
            SheetContent()
        }
        .alert("Are you sure?", isPresented: $showAlert) {
            Button("Cancel", role: .cancel) { }
            Button("Delete", role: .destructive) {
                print("Deleted")
            }
        } message: {
            Text("This action cannot be undone.")
        }
    }
}

struct SheetContent: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Text("This is a sheet!")
                .font(.title)
                .navigationTitle("Sheet")
                .toolbar {
                    Button("Done") {
                        dismiss()
                    }
                }
        }
    }
}
```

**What this looks like**: Two buttons. "Show Sheet" slides up a modal with "This is a sheet!" and a "Done" button in the toolbar. "Show Alert" shows a system dialog with "Cancel" and a red "Delete" button.

The `@Environment(\.dismiss)` value is how a presented view dismisses itself — you'll use this pattern frequently with sheets.

### TabView

`TabView` creates the tab bar at the bottom of the screen, like tabs in a mobile app.

```swift
struct MainTabView: View {
    var body: some View {
        TabView {
            Tab("Home", systemImage: "house") {
                NavigationStack {
                    Text("Home Screen")
                        .navigationTitle("Home")
                }
            }

            Tab("Search", systemImage: "magnifyingglass") {
                NavigationStack {
                    Text("Search Screen")
                        .navigationTitle("Search")
                }
            }

            Tab("Profile", systemImage: "person") {
                NavigationStack {
                    Text("Profile Screen")
                        .navigationTitle("Profile")
                }
            }
        }
    }
}
```

**What this looks like**: A tab bar at the bottom with three icons (house, magnifying glass, person) and labels. Tapping each tab shows a different screen. Each tab has its own independent navigation stack.

Each tab gets its own `NavigationStack` so that navigation within one tab doesn't affect the others.

### Exercise 5.1: Build a Master-Detail Navigation

**Task:** Create a list of programming languages (name and description). Tapping a language navigates to a detail screen showing its full description. Add a "Favorites" sheet that can be opened from a toolbar button.

<details>
<summary>Hint 1: Structure</summary>

You need a `Language` struct, a list view wrapped in `NavigationStack`, and a detail view. Use `.toolbar` to add a button that triggers a `.sheet`.
</details>

<details>
<summary>Hint 2: Toolbar button</summary>

```swift
.toolbar {
    Button {
        showFavorites = true
    } label: {
        Image(systemName: "star")
    }
}
```
</details>

<details>
<summary>Solution</summary>

```swift
struct Language: Identifiable {
    let id = UUID()
    let name: String
    let description: String
}

let sampleLanguages = [
    Language(name: "Swift", description: "A powerful and intuitive programming language created by Apple for building apps across all Apple platforms."),
    Language(name: "Kotlin", description: "A modern programming language that makes Android development faster and more enjoyable."),
    Language(name: "Rust", description: "A systems programming language focused on safety, speed, and concurrency."),
    Language(name: "TypeScript", description: "A typed superset of JavaScript that compiles to plain JavaScript."),
]

struct LanguageList: View {
    @State private var showFavorites = false

    var body: some View {
        NavigationStack {
            List(sampleLanguages) { language in
                NavigationLink(language.name) {
                    LanguageDetail(language: language)
                }
            }
            .navigationTitle("Languages")
            .toolbar {
                Button {
                    showFavorites = true
                } label: {
                    Image(systemName: "star")
                }
            }
            .sheet(isPresented: $showFavorites) {
                NavigationStack {
                    Text("Your favorites will appear here")
                        .navigationTitle("Favorites")
                        .toolbar {
                            Button("Done") {
                                showFavorites = false
                            }
                        }
                }
            }
        }
    }
}

struct LanguageDetail: View {
    let language: Language

    var body: some View {
        ScrollView {
            Text(language.description)
                .padding()
        }
        .navigationTitle(language.name)
    }
}

#Preview {
    LanguageList()
}
```

**What this looks like**: A list of language names with a star icon in the top-right toolbar. Tapping a language navigates to a detail screen with its description. Tapping the star opens a sheet.
</details>

### Checkpoint 5

Before moving on, make sure you can:
- [ ] Wrap views in `NavigationStack` and add navigation titles
- [ ] Use `NavigationLink` for push navigation
- [ ] Present sheets with `.sheet(isPresented:)` and dismiss them
- [ ] Show alerts with `.alert(isPresented:)`
- [ ] Create a tab-based layout with `TabView`
- [ ] Explain why each tab gets its own `NavigationStack`

---

## Section 6: User Input and Forms

### Form and Section

`Form` gives you the iOS Settings-style layout for free. It's a special container that automatically styles its children as form rows.

```swift
struct SettingsView: View {
    @State private var username = ""
    @State private var notificationsEnabled = true
    @State private var fontSize = 14.0

    var body: some View {
        NavigationStack {
            Form {
                Section("Account") {
                    TextField("Username", text: $username)
                    Text("Member since 2024")
                }

                Section("Preferences") {
                    Toggle("Notifications", isOn: $notificationsEnabled)

                    HStack {
                        Text("Font Size")
                        Slider(value: $fontSize, in: 10...24, step: 1)
                        Text("\(Int(fontSize))")
                            .frame(width: 30)
                    }
                }
            }
            .navigationTitle("Settings")
        }
    }
}
```

**What this looks like**: An iOS Settings-style screen with two grouped sections. "Account" has a text field and a read-only label. "Preferences" has a toggle switch and a slider with the current value displayed.

### Text Input Views

SwiftUI provides several text input views:

```swift
struct TextInputExamples: View {
    @State private var name = ""
    @State private var password = ""
    @State private var bio = ""

    var body: some View {
        Form {
            Section("Credentials") {
                TextField("Name", text: $name)

                // Dots out the input (like <input type="password">)
                SecureField("Password", text: $password)
            }

            Section("About You") {
                // Multi-line text input (like <textarea>)
                TextEditor(text: $bio)
                    .frame(minHeight: 100)
            }
        }
    }
}
```

### Pickers

Pickers let users choose from a set of options. SwiftUI renders them differently depending on context (inline, menu, wheel, etc.):

```swift
struct PickerExamples: View {
    @State private var selectedColor = "Red"
    @State private var birthDate = Date()
    @State private var volume = 50.0
    @State private var quantity = 1

    let colors = ["Red", "Green", "Blue", "Purple"]

    var body: some View {
        Form {
            // Menu picker (tap to see dropdown)
            Picker("Favorite Color", selection: $selectedColor) {
                ForEach(colors, id: \.self) { color in
                    Text(color).tag(color)
                }
            }

            // Date picker
            DatePicker("Birthday", selection: $birthDate, displayedComponents: .date)

            // Slider (continuous value)
            HStack {
                Text("Volume")
                Slider(value: $volume, in: 0...100)
                Text("\(Int(volume))%")
                    .frame(width: 45)
            }

            // Stepper (discrete value)
            Stepper("Quantity: \(quantity)", value: $quantity, in: 1...10)
        }
    }
}
```

**What this looks like**: A settings-style form with a tappable color picker showing the current selection, a date picker that expands into a calendar, a volume slider, and a quantity stepper with plus/minus buttons.

All these inputs use `@Binding` (via the `$` prefix) to connect to your state — just like controlled inputs in React.

### Validation Patterns

SwiftUI doesn't have a built-in form validation system (no Formik or React Hook Form equivalent). You build validation with computed properties and conditional rendering:

```swift
struct SignupForm: View {
    @State private var email = ""
    @State private var password = ""
    @State private var confirmPassword = ""

    private var isEmailValid: Bool {
        email.contains("@") && email.contains(".")
    }

    private var isPasswordValid: Bool {
        password.count >= 8
    }

    private var doPasswordsMatch: Bool {
        password == confirmPassword && !password.isEmpty
    }

    private var isFormValid: Bool {
        isEmailValid && isPasswordValid && doPasswordsMatch
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("Email", text: $email)
                        .keyboardType(.emailAddress)
                        .textInputAutocapitalization(.never)

                    if !email.isEmpty && !isEmailValid {
                        Text("Please enter a valid email address")
                            .foregroundStyle(.red)
                            .font(.caption)
                    }
                }

                Section {
                    SecureField("Password", text: $password)

                    if !password.isEmpty && !isPasswordValid {
                        Text("Password must be at least 8 characters")
                            .foregroundStyle(.red)
                            .font(.caption)
                    }

                    SecureField("Confirm Password", text: $confirmPassword)

                    if !confirmPassword.isEmpty && !doPasswordsMatch {
                        Text("Passwords don't match")
                            .foregroundStyle(.red)
                            .font(.caption)
                    }
                }

                Section {
                    Button("Sign Up") {
                        print("Signing up with \(email)")
                    }
                    .disabled(!isFormValid)
                }
            }
            .navigationTitle("Sign Up")
        }
    }
}
```

**What this looks like**: A form with email and password fields. Red validation messages appear below invalid fields as the user types. The "Sign Up" button is grayed out until all fields are valid.

The `.disabled()` modifier is key — it prevents interaction and visually dims the view, like the `disabled` attribute on HTML buttons.

### Exercise 6.1: Build a Settings Form

**Task:** Build a settings form with:
- A "Display Name" text field
- A "Theme" picker with Light/Dark/System options
- A "Max Items" stepper (range 5 to 50, step 5)
- A "Save" button that's disabled when the display name is empty

<details>
<summary>Hint: Using an enum with Picker</summary>

```swift
enum Theme: String, CaseIterable {
    case light = "Light"
    case dark = "Dark"
    case system = "System"
}

// In your Picker:
Picker("Theme", selection: $selectedTheme) {
    ForEach(Theme.allCases, id: \.self) { theme in
        Text(theme.rawValue).tag(theme)
    }
}
```
</details>

<details>
<summary>Solution</summary>

```swift
enum Theme: String, CaseIterable {
    case light = "Light"
    case dark = "Dark"
    case system = "System"
}

struct AppSettings: View {
    @State private var displayName = ""
    @State private var selectedTheme = Theme.system
    @State private var maxItems = 20

    var body: some View {
        NavigationStack {
            Form {
                Section("Profile") {
                    TextField("Display Name", text: $displayName)
                }

                Section("Appearance") {
                    Picker("Theme", selection: $selectedTheme) {
                        ForEach(Theme.allCases, id: \.self) { theme in
                            Text(theme.rawValue).tag(theme)
                        }
                    }
                }

                Section("Content") {
                    Stepper("Max Items: \(maxItems)", value: $maxItems, in: 5...50, step: 5)
                }

                Section {
                    Button("Save") {
                        print("Saved: \(displayName), \(selectedTheme.rawValue), \(maxItems)")
                    }
                    .disabled(displayName.isEmpty)
                }
            }
            .navigationTitle("Settings")
        }
    }
}

#Preview {
    AppSettings()
}
```

**What this looks like**: A Settings-style form with sections for Profile (text field), Appearance (theme picker), Content (stepper showing current value), and a Save button that's disabled until you enter a display name.
</details>

### Checkpoint 6

Before moving on, make sure you can:
- [ ] Use `Form` and `Section` for settings-style layouts
- [ ] Work with TextField, SecureField, TextEditor
- [ ] Use Picker, DatePicker, Slider, and Stepper
- [ ] Build validation logic with computed properties
- [ ] Disable buttons conditionally with `.disabled()`

---

## Practice Project: Recipe Book

### Project Description

Build a recipe book app that ties together everything you've learned: lists, navigation, state management, forms, and user input. The app has a list of recipes that you can browse, view details for, and add new ones through a form.

### Requirements

- A main screen with a list of recipes (name and short description)
- Tapping a recipe navigates to a detail view showing ingredients and instructions
- A "+" button in the toolbar opens a sheet with a form to add new recipes
- The form requires a name, at least one ingredient, and instructions
- The form has a "Save" button that's disabled until the required fields are filled in
- New recipes appear in the list after saving

### Getting Started

Create a new SwiftUI view file or replace your `ContentView.swift` with the starter code below.

**Step 1: Define the data model**

```swift
struct Recipe: Identifiable {
    let id = UUID()
    var name: String
    var description: String
    var ingredients: [String]
    var instructions: String
    var isFavorite: Bool = false
}
```

**Step 2: Plan the views**

You'll need:
1. `RecipeBookApp` — The entry point with a `NavigationStack`
2. `RecipeListView` — Shows the list of recipes
3. `RecipeDetailView` — Shows one recipe's full details
4. `AddRecipeView` — A form sheet for creating a new recipe

**Step 3: Think about state**

- Where should the list of recipes live? (The top-level view that owns the list)
- What state does the "Add Recipe" form need? (Local `@State` for each field)
- How does the form communicate the new recipe back? (A closure or `@Binding`)

### Hints

<details>
<summary>If you're not sure where to start</summary>

Start with a hardcoded list of 2-3 recipes in a `@State` array. Get the list view and detail view working with navigation first, then add the "add recipe" form.
</details>

<details>
<summary>Passing data from the sheet back to the list</summary>

Use a closure pattern: the parent passes an `onSave` closure to the sheet. When the sheet's Save button is tapped, it calls the closure with the new recipe. The parent appends it to the array.

```swift
.sheet(isPresented: $showAddRecipe) {
    AddRecipeView { newRecipe in
        recipes.append(newRecipe)
    }
}
```
</details>

<details>
<summary>Managing ingredients in the form</summary>

Use a `@State` array of strings for ingredients, with a text field and "Add" button to append new ones. Display current ingredients in a `ForEach` with swipe-to-delete.
</details>

### Example Solution

<details>
<summary>Click to see one possible solution</summary>

```swift
import SwiftUI

struct Recipe: Identifiable {
    let id = UUID()
    var name: String
    var description: String
    var ingredients: [String]
    var instructions: String
    var isFavorite: Bool = false
}

struct RecipeBookView: View {
    @State private var recipes = [
        Recipe(
            name: "Pasta Aglio e Olio",
            description: "Simple garlic and olive oil pasta",
            ingredients: ["Spaghetti", "Garlic (6 cloves)", "Olive oil", "Red pepper flakes", "Parsley", "Parmesan"],
            instructions: "Cook pasta. Sauté thinly sliced garlic in olive oil until golden. Add pepper flakes. Toss with pasta and cooking water. Top with parsley and parmesan."
        ),
        Recipe(
            name: "Avocado Toast",
            description: "Quick and customizable breakfast",
            ingredients: ["Bread", "Avocado", "Lemon juice", "Salt", "Red pepper flakes"],
            instructions: "Toast bread. Mash avocado with lemon juice and salt. Spread on toast. Add pepper flakes and any other toppings you like."
        ),
    ]
    @State private var showAddRecipe = false

    var body: some View {
        NavigationStack {
            List {
                ForEach(recipes) { recipe in
                    NavigationLink {
                        RecipeDetailView(recipe: recipe)
                    } label: {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(recipe.name)
                                .font(.headline)
                            Text(recipe.description)
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("My Recipes")
            .toolbar {
                Button {
                    showAddRecipe = true
                } label: {
                    Image(systemName: "plus")
                }
            }
            .sheet(isPresented: $showAddRecipe) {
                AddRecipeView { newRecipe in
                    recipes.append(newRecipe)
                }
            }
        }
    }
}

struct RecipeDetailView: View {
    let recipe: Recipe

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text(recipe.description)
                    .foregroundStyle(.secondary)

                VStack(alignment: .leading, spacing: 8) {
                    Text("Ingredients")
                        .font(.title2.bold())

                    ForEach(recipe.ingredients, id: \.self) { ingredient in
                        HStack(alignment: .top) {
                            Text("•")
                            Text(ingredient)
                        }
                    }
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("Instructions")
                        .font(.title2.bold())

                    Text(recipe.instructions)
                }
            }
            .padding()
        }
        .navigationTitle(recipe.name)
    }
}

struct AddRecipeView: View {
    @Environment(\.dismiss) private var dismiss

    var onSave: (Recipe) -> Void

    @State private var name = ""
    @State private var description = ""
    @State private var ingredientText = ""
    @State private var ingredients: [String] = []
    @State private var instructions = ""

    private var isValid: Bool {
        !name.isEmpty && !ingredients.isEmpty && !instructions.isEmpty
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Basics") {
                    TextField("Recipe Name", text: $name)
                    TextField("Short Description", text: $description)
                }

                Section("Ingredients") {
                    HStack {
                        TextField("Add ingredient", text: $ingredientText)
                        Button("Add") {
                            let trimmed = ingredientText.trimmingCharacters(in: .whitespaces)
                            if !trimmed.isEmpty {
                                ingredients.append(trimmed)
                                ingredientText = ""
                            }
                        }
                        .disabled(ingredientText.trimmingCharacters(in: .whitespaces).isEmpty)
                    }

                    ForEach(ingredients, id: \.self) { ingredient in
                        Text(ingredient)
                    }
                    .onDelete { offsets in
                        ingredients.remove(atOffsets: offsets)
                    }
                }

                Section("Instructions") {
                    TextEditor(text: $instructions)
                        .frame(minHeight: 100)
                }
            }
            .navigationTitle("Add Recipe")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        let recipe = Recipe(
                            name: name,
                            description: description,
                            ingredients: ingredients,
                            instructions: instructions
                        )
                        onSave(recipe)
                        dismiss()
                    }
                    .disabled(!isValid)
                }
            }
        }
    }
}

#Preview {
    RecipeBookView()
}
```

**Key points in this solution:**

- `RecipeBookView` owns the `@State` array of recipes and passes a closure to the add form
- `RecipeDetailView` takes a recipe as a plain parameter (no binding needed — it's read-only)
- `AddRecipeView` uses local `@State` for form fields and calls `onSave` with a constructed recipe
- The form validates with a computed property and disables Save when invalid
- `@Environment(\.dismiss)` handles dismissing the sheet from both Cancel and Save
- Ingredients use a text field + button + list pattern with swipe-to-delete
</details>

### Extending the Project

If you want to keep going, try:
- Add a favorite toggle (heart icon) on each recipe row and detail view
- Add sections in the list (Favorites and All Recipes)
- Add a search bar with `.searchable(text:)`
- Persist recipes across app launches using `@AppStorage` or file storage
- Add a tab bar with "Recipes" and "Favorites" tabs

---

## Summary

You've covered the core building blocks of SwiftUI:

- **Mental Model**: SwiftUI is declarative like React — views are structs, `body` is your render function
- **Views and Modifiers**: Built-in views styled with chainable modifiers (order matters)
- **State**: `@State` for local state, `@Binding` for two-way child props, `@Observable` for shared models, `@Environment` for system/injected values
- **Lists**: `List` and `ForEach` for dynamic content, `Identifiable` for stable identity
- **Navigation**: `NavigationStack` for push navigation, sheets for modals, `TabView` for tabs
- **Forms**: `Form`/`Section` for settings UI, validation with computed properties

### Skills You've Gained

You can now:
- Build multi-screen SwiftUI apps with lists, detail views, and forms
- Manage state at different levels of your view hierarchy
- Apply the React mental model to SwiftUI development
- Use Xcode previews for rapid iteration

---

## Quick Reference: React to SwiftUI

| React | SwiftUI | Notes |
|-------|---------|-------|
| Function component | `struct MyView: View` | Views are value types (structs) |
| JSX return | `var body: some View` | Uses `@ViewBuilder` result builder |
| Props | Init parameters | `MyView(title: "Hello")` |
| `useState` | `@State private var` | Mutate directly, no setter function |
| Prop + callback | `@Binding var` | `$value` creates a binding |
| `useContext` | `@Environment` | For system values or injected objects |
| Context Provider | `.environment(object)` | Inject `@Observable` for descendants |
| External store | `@Observable class` | Fine-grained: only re-renders on read properties |
| `useEffect(fn, [])` | `.task { }` | Async, auto-cancels on disappear |
| `useEffect(fn, [dep])` | `.onChange(of: dep) { }` | Runs when a value changes |
| `.map(item => <X>)` | `ForEach(items) { item in }` | Needs `Identifiable` or `id:` parameter |
| CSS classes/styles | View modifiers | Chainable, order matters |
| `<Link to="...">` | `NavigationLink { }` | Wrap in `NavigationStack` |
| React Router | `NavigationStack` + `navigationDestination` | Value-type navigation paths |
| Portal / Modal | `.sheet(isPresented:)` | Sheets, alerts, full-screen covers |
| `key` prop | `Identifiable` protocol | `id` property for stable identity |
| `children` | `@ViewBuilder` closure | Pass views as trailing closures |
| CSS `display: flex` | `HStack` / `VStack` | No CSS, stacks are the layout model |
| `flex-grow: 1` | `Spacer()` | Pushes content apart |
| `<input disabled>` | `.disabled(true)` | Modifier on any view |
| Fragments `<>...</>` | `Group { }` | Group views without adding layout |

---

## Next Steps

Now that you understand SwiftUI fundamentals, explore:

- **iOS App Patterns** — Architecture patterns (MVVM, etc.) for structuring larger SwiftUI apps
- **UIKit Essentials** — For when SwiftUI doesn't cover your needs
- **iOS Data Persistence** — Storing data locally with SwiftData, UserDefaults, and files

## Additional Resources

- [SwiftUI Documentation](https://developer.apple.com/documentation/swiftui): Apple's official reference
- [SwiftUI Tutorials](https://developer.apple.com/tutorials/swiftui): Apple's step-by-step tutorials
- [SF Symbols](https://developer.apple.com/sf-symbols/): Browse the built-in icon library
- [Human Interface Guidelines](https://developer.apple.com/design/human-interface-guidelines/): Apple's design guidelines for iOS apps
