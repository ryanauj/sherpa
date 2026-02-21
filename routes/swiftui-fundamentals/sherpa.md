---
title: SwiftUI Fundamentals
route_map: /routes/swiftui-fundamentals/map.md
paired_guide: /routes/swiftui-fundamentals/guide.md
topics:
  - SwiftUI
  - Declarative UI
  - State Management
  - Navigation
  - View Composition
---

# SwiftUI Fundamentals - Sherpa (AI Teaching Guide)

**Purpose**: This sherpa guide helps AI assistants teach SwiftUI fundamentals to developers with React experience. It uses React comparisons throughout to bridge familiar concepts into SwiftUI's world.

**Route Map**: See `/routes/swiftui-fundamentals/map.md` for the high-level overview of this route.
**Paired Guide**: The human-focused content is at `/routes/swiftui-fundamentals/guide.md`.

---

## Teaching Overview

### Learning Objectives
By the end of this session, the learner should be able to:
- Build views using SwiftUI's declarative syntax and explain how it maps to React concepts
- Compose UIs from small, reusable view structs using stacks, modifiers, and composition
- Manage state with @State, @Binding, @Observable, and @Environment, and explain when to use each
- Display dynamic data with List and ForEach, including swipe actions and sections
- Build navigation flows with NavigationStack, sheets, and TabView
- Handle user input with Forms, TextFields, Pickers, and validation patterns
- Preview and iterate on UI with Xcode's canvas and #Preview macro

### Prior Sessions
Before starting, check `.sessions/index.md` and `.sessions/swiftui-fundamentals/` for prior session history. If the learner has completed previous sessions on this route, review the summaries to understand what they've covered and pick up where they left off.

### Prerequisites to Verify
Before starting, verify the learner has:
- Swift language fundamentals (structs, protocols, closures, optionals, enums with associated values)
- Xcode installed and basic familiarity (creating projects, running on simulator)
- Comfort with React (function components, hooks, JSX) — this route assumes React fluency

**If prerequisites are missing**: If Swift is weak, suggest they review the swift-for-developers route first. If Xcode is unfamiliar, suggest xcode-essentials. If they don't know React, the comparisons won't help — adjust teaching to remove React analogies and use general programming concepts instead.

### Learner Preferences Configuration

Learners can configure their preferred learning style by creating a `.sherpa-config.yml` file in the repository root (gitignored by default). Configuration options include:

**Teaching Style:**
- `tone`: objective, encouraging, humorous (default: objective and respectful)
- `explanation_depth`: concise, balanced, detailed
- `pacing`: learner-led, balanced, structured

**Assessment Format:**
- `quiz_type`: multiple_choice, explanation, mixed (default: mixed)
- `quiz_frequency`: after_each_section, after_major_topics, end_of_route
- `feedback_style`: immediate, summary, detailed

**Example `.sherpa-config.yml`:**
```yaml
teaching:
  tone: encouraging
  explanation_depth: balanced
  pacing: learner-led

assessment:
  quiz_type: mixed
  quiz_frequency: after_major_topics
  feedback_style: immediate
```

If no configuration file exists, use defaults (objective tone, mixed assessments, balanced pacing).

### Assessment Strategies

Use a combination of assessment types to verify understanding:

**Multiple Choice Questions:**
- Present 3-4 answer options
- Include one correct answer and plausible distractors
- Good for checking factual knowledge quickly
- Example: "What does @Binding do? A) Creates local state B) Reads a value from the environment C) Creates a two-way reference to state owned by a parent view D) Observes changes to an @Observable class"

**Explanation Questions:**
- Ask learner to explain concepts in their own words
- Assess deeper understanding and ability to apply knowledge
- Example: "How does SwiftUI know which views to re-render when state changes, and how does this compare to React's reconciliation?"

**Mixed Approach (Recommended):**
- Use multiple choice for quick checks after introducing modifiers, property wrappers, or navigation patterns
- Use explanation questions for core concepts like the SwiftUI mental model and state management decisions
- Adapt based on learner responses and confidence level

---

## React-to-SwiftUI Reference

Use this table throughout the session when bridging concepts:

| React | SwiftUI |
|-------|---------|
| Function component | Struct conforming to `View` |
| `useState` | `@State` |
| Props | Init parameters |
| Callback prop | `@Binding` |
| `useContext` / Provider | `@Environment` / `.environment()` |
| `useEffect` | `.task()`, `.onAppear()`, `.onChange()` |
| External store (Redux, Zustand) | `@Observable` class |
| Conditional render (ternary/&&) | `if`/`else` in `body` |
| `.map()` | `ForEach` |
| CSS / styled-components | View modifiers |
| React Router | `NavigationStack` / `NavigationLink` |

Refer back to this table whenever introducing a concept that has a React parallel. The learner thinks in React — always anchor new concepts to what they already know.

---

## Teaching Flow

### Introduction

**What to Cover:**
- SwiftUI is Apple's declarative UI framework — same paradigm as React, different execution
- They'll be writing views that describe UI as a function of state — familiar territory
- The differences are in the details: structs vs functions, property wrappers vs hooks, modifiers vs CSS
- By the end, they'll build a multi-screen app with navigation, state, lists, and forms

**Opening Questions to Assess Level:**
1. "Have you built anything with SwiftUI before, or is this your first time?"
2. "What kind of app do you want to build? That'll help me tailor examples."
3. "How comfortable are you with Swift structs, protocols, and closures?"

**Adapt based on responses:**
- If they've dabbled in SwiftUI: Focus on filling gaps, move faster through basics, ask what confused them before
- If completely new to SwiftUI: Start from scratch, lean heavily on React comparisons
- If they have a specific app idea: Use their app as a running example where possible
- If Swift is shaky: Spend more time explaining Swift-specific mechanics (property wrappers, protocol conformance)

**Opening framing:**
"SwiftUI is going to feel familiar and foreign at the same time. The declarative model is React — you describe what the UI should look like for a given state. But the mechanics are different. There's no virtual DOM, no JSX, no CSS. Instead you've got structs, property wrappers, and a modifier chain system. Let's start with the mental model, then build up from there."

---

### Section 1: SwiftUI Mental Model

**Core Concept to Teach:**
SwiftUI views are lightweight structs that describe UI. The framework creates, compares, and destroys them constantly. The `body` property returns what the view looks like right now. This is analogous to a React function component's return value, but with important structural differences.

**How to Explain:**
1. Start with what's the same: "In React, a function component returns JSX describing the UI. In SwiftUI, a struct's `body` property returns a view tree describing the UI. Same idea."
2. Then what's different: "But SwiftUI views are value types — structs, not objects. They're cheap to create and the framework recreates them constantly. You never call `new` on a view or hold a reference to one."
3. Show the simplest possible view to establish the pattern

**Example to Present:**

React equivalent they already know:
```jsx
function Greeting() {
  return <Text>Hello, World!</Text>;
}
```

SwiftUI equivalent:
```swift
struct Greeting: View {
    var body: some View {
        Text("Hello, World!")
    }
}
```

Walk through the pieces:
- `struct Greeting: View` — a struct conforming to the `View` protocol (like implementing an interface)
- `var body: some View` — a computed property that returns the view's content. `some View` means "some specific type that conforms to View" (opaque return type)
- The body is called by the framework whenever SwiftUI needs to know what this view looks like

**The View Builder:**
"The `body` property uses a feature called `@ViewBuilder`, which lets you write multiple views and control flow (`if`/`else`, `ForEach`) directly in the body — no array wrapping needed. It's like JSX but without the curly braces for expressions."

```swift
struct ProfileCard: View {
    var name: String
    var isOnline: Bool

    var body: some View {
        VStack {
            Text(name)
                .font(.title)
            if isOnline {
                Text("Online")
                    .foregroundStyle(.green)
            } else {
                Text("Offline")
                    .foregroundStyle(.gray)
            }
        }
    }
}
```

"In React you'd use a ternary or && for conditional rendering. In SwiftUI, you write plain `if`/`else` inside the body. The `@ViewBuilder` handles it."

**Xcode Previews:**
"React has hot reload. SwiftUI has the Xcode canvas — a live preview that updates as you type. You define previews with the `#Preview` macro:"

```swift
#Preview {
    Greeting()
}

#Preview("Online Profile") {
    ProfileCard(name: "Alice", isOnline: true)
}

#Preview("Offline Profile") {
    ProfileCard(name: "Bob", isOnline: false)
}
```

"You can have multiple previews to see different states side by side — like Storybook stories, but built into the IDE."

**Common Misconceptions:**
- Misconception: "Views are like UIKit view objects that persist on screen" → Clarify: "SwiftUI views are value types — lightweight descriptions. The framework creates and destroys them frequently. Think of them as render output, not persistent objects"
- Misconception: "`some View` means it can return different types" → Clarify: "`some View` means it returns one specific type, but the compiler figures out what it is. If you need to return different types, you need explicit type erasure or conditional logic inside a single container"
- Misconception: "I should store state as regular properties on the struct" → Clarify: "Structs are recreated constantly. Regular properties get reset. You need `@State` for persistent local state — we'll cover that in Section 3"

**Verification Questions:**
1. "How does a SwiftUI view struct compare to a React function component?"
2. "What happens when SwiftUI needs to update the UI — does it mutate the existing view or create a new one?"
3. Multiple choice: "What does `some View` mean in `var body: some View`? A) The body can return any View type each time it's called B) The body returns a specific View type determined by the compiler C) The body returns an optional View D) The body returns multiple Views"

**Good answer indicators:**
- They understand views are descriptions, not persistent objects
- They know views are recreated (B is correct for the multiple choice)
- They can map the pattern to React function components

**If they struggle:**
- Anchor harder to React: "Think of the struct as a function component that happens to be written as a struct. The `body` property is the return statement."
- If `some View` is confusing: "Don't worry about `some View` right now. Just think of it as 'the return type of my view.' The compiler handles the details."
- If structs vs classes is confusing: "Structs are copied, not shared. SwiftUI exploits this — it can freely create and discard views because they're just data."

---

### Section 2: Built-in Views and Modifiers

**Core Concept to Teach:**
SwiftUI provides built-in views (Text, Image, Button, etc.) and a modifier system for styling and layout. Modifiers wrap views in layers — order matters because each modifier creates a new view wrapping the previous one.

**How to Explain:**
1. "In React, you style with CSS classes or styled-components. In SwiftUI, there's no CSS at all. Styling is done with view modifiers — method calls chained onto views."
2. "Each modifier wraps the view in a new layer. This means order matters in ways that CSS doesn't."
3. Start with basic views, then show modifiers, then explain ordering

**Basic Views:**

```swift
// Text — like React's <p> or <span>
Text("Hello, World!")

// Image — like React's <img>
// SF Symbols are Apple's built-in icon library (thousands of icons)
Image(systemName: "star.fill")

// Button — like React's <button onClick={...}>
Button("Tap Me") {
    print("Button tapped")
}

// Button with a custom label
Button {
    print("Tapped")
} label: {
    Label("Settings", systemImage: "gear")
}

// Toggle — like a React checkbox with onChange
Toggle("Airplane Mode", isOn: $isAirplaneOn)

// TextField — like React's <input type="text">
TextField("Enter name", text: $name)
```

"Notice the `$` prefix on some of these — `$isAirplaneOn`, `$name`. That creates a Binding, which is a two-way connection to state. We'll cover this in Section 3. For now, just know that interactive controls need Bindings to the state they modify."

**View Modifiers:**

```swift
Text("Welcome")
    .font(.largeTitle)
    .fontWeight(.bold)
    .foregroundStyle(.blue)
    .padding()
    .background(.yellow)
    .cornerRadius(8)
```

"Each line is a modifier that wraps the view. It reads top to bottom: start with 'Welcome' text, make it large title, bold, blue, add padding around it, put a yellow background behind it, round the corners."

**Modifier Order Matters — Key Insight:**

```swift
// Padding THEN background: background covers the padded area
Text("Hello")
    .padding()
    .background(.red)

// Background THEN padding: background only covers the text, padding is outside
Text("Hello")
    .background(.red)
    .padding()
```

"This is the biggest mental shift from CSS. In CSS, padding and background are independent properties on the same box. In SwiftUI, each modifier wraps the view in a new layer. `.padding()` creates a larger view, then `.background(.red)` paints behind that larger view. Reverse the order and you paint behind the small text first, then add transparent padding around it."

Present this mental model: "Think of Russian nesting dolls. Each modifier adds a new doll around the previous one. The order you add dolls determines the final appearance."

**Layout with Stacks:**

"In React, you use flexbox (`display: flex`, `flex-direction`). In SwiftUI, you use explicit stack views:"

```swift
// VStack = flex-direction: column
VStack {
    Text("Top")
    Text("Middle")
    Text("Bottom")
}

// HStack = flex-direction: row
HStack {
    Image(systemName: "star.fill")
    Text("Favorites")
}

// ZStack = position: absolute layering
ZStack {
    Color.blue          // fills the background
    Text("On top")      // layered on top
        .foregroundStyle(.white)
}
```

"VStack, HStack, and ZStack replace flexbox. V is vertical (column), H is horizontal (row), Z is depth (layering). You can nest them freely."

**Spacer and Alignment:**

```swift
HStack {
    Text("Left")
    Spacer()      // pushes items apart — like flex: 1 or justify-content: space-between
    Text("Right")
}

VStack(alignment: .leading) {  // like align-items: flex-start
    Text("Title")
        .font(.headline)
    Text("Subtitle")
        .font(.subheadline)
        .foregroundStyle(.secondary)
}
.frame(maxWidth: .infinity, alignment: .leading)  // like width: 100%
```

"Spacer is a flexible view that expands to fill available space. It's your primary tool for pushing things apart — like a spring between views."

**Common Misconceptions:**
- Misconception: "Modifiers are like CSS properties — order doesn't matter" → Clarify: "Every modifier wraps the view in a new layer. `.padding().background(.red)` and `.background(.red).padding()` produce visually different results"
- Misconception: "I can apply global styles like CSS classes" → Clarify: "There are no selectors or cascading styles. Every view is styled individually with modifiers. You can create custom modifiers or view extensions for reuse, but there's no CSS-like inheritance"
- Misconception: "VStack is like `<div>`" → Clarify: "Stacks are layout containers with explicit direction. There's no generic wrapper like `<div>`. If you just need to group views, use `Group` (no layout) or a stack (with layout)"

**Verification Questions:**
1. "What's the difference between `.padding().background(.red)` and `.background(.red).padding()`?"
2. "What's the SwiftUI equivalent of React's flexbox row layout?"
3. Multiple choice: "What does `Spacer()` do inside an HStack? A) Adds blank text B) Creates invisible padding on all sides C) Expands to fill available horizontal space D) Adds a fixed 8-point gap"

**Good answer indicators:**
- They understand modifier order creates different wrapping layers
- They know HStack for row layout (C is correct for the multiple choice)
- They can articulate why there's no CSS equivalent

**If they struggle:**
- Draw the nesting doll analogy on paper (or describe it step by step)
- Have them experiment in Xcode preview — change modifier order and see results live
- If stack layout is confusing, compare directly: "VStack = `flex-direction: column`, HStack = `flex-direction: row`"

**Exercise 2.1:**
"Build a contact card view with: the person's name in large bold text, their email in smaller gray text below, a horizontal row with a phone icon and phone number, all left-aligned with padding and a light gray background with rounded corners."

**How to Guide Them:**
1. First ask: "What stacks would you use to lay this out?"
2. If stuck on structure: "Think about it as a VStack for the vertical content, with an HStack for the phone row"
3. If stuck on styling: "Remember, modifiers chain: `.font(.title)`, `.foregroundStyle(.gray)`, `.padding()`, `.background()`"
4. If stuck on icons: "Use `Image(systemName: \"phone.fill\")` — SF Symbols are available everywhere in SwiftUI"

**Solution:**
```swift
struct ContactCard: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Jane Smith")
                .font(.title)
                .fontWeight(.bold)
            Text("jane@example.com")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            HStack {
                Image(systemName: "phone.fill")
                Text("(555) 123-4567")
            }
            .font(.subheadline)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}
```

---

### Section 3: State Management

**Core Concept to Teach:**
SwiftUI uses property wrappers to manage state. Each wrapper serves a different purpose, and choosing the right one is critical. This maps to React hooks, but the ownership model is more explicit.

**How to Explain:**
1. "In React, `useState` handles local state. SwiftUI has `@State` — same idea, different mechanism."
2. "But SwiftUI goes further. Where React has various patterns for shared state (context, Redux, Zustand), SwiftUI has built-in property wrappers for each pattern."
3. Walk through each property wrapper with its React equivalent

**@State — Local View State (like useState):**

React:
```jsx
function Counter() {
  const [count, setCount] = useState(0);
  return <button onClick={() => setCount(count + 1)}>{count}</button>;
}
```

SwiftUI:
```swift
struct Counter: View {
    @State private var count = 0

    var body: some View {
        Button("Count: \(count)") {
            count += 1
        }
    }
}
```

"@State tells SwiftUI: 'this view owns this piece of state. When it changes, re-render.'"

Key points to emphasize:
- `@State` should always be `private` — it's owned by this view
- You mutate it directly (`count += 1`) instead of calling a setter function
- SwiftUI stores the actual value outside the struct so it persists across re-renders
- Only use @State for simple, local, owned state (strings, ints, bools)

**Critical @State gotcha:**
"@State initializes once. If a parent passes a value and you use it to initialize @State, the State captures the initial value and ignores future parent updates:"

```swift
struct ChildView: View {
    // BUG: This captures the initial value only.
    // If parent changes 'startingCount', this view won't update.
    @State private var count: Int

    init(startingCount: Int) {
        _count = State(initialValue: startingCount)
    }

    var body: some View {
        Text("Count: \(count)")
    }
}
```

"In React terms: imagine if `useState(props.initialValue)` literally only read `props.initialValue` on first mount and ignored all future prop changes. That's how @State works. If you need the value to stay in sync with the parent, use a Binding or pass it as a plain property."

**@Binding — Two-Way State Reference (like props + callback):**

React pattern:
```jsx
function ToggleRow({ label, isOn, onToggle }) {
  return <Switch checked={isOn} onChange={onToggle} />;
}

// Parent
function Settings() {
  const [airplane, setAirplane] = useState(false);
  return <ToggleRow label="Airplane" isOn={airplane} onToggle={() => setAirplane(!airplane)} />;
}
```

SwiftUI equivalent:
```swift
struct ToggleRow: View {
    let label: String
    @Binding var isOn: Bool  // two-way reference to parent's state

    var body: some View {
        Toggle(label, isOn: $isOn)
    }
}

// Parent
struct Settings: View {
    @State private var airplaneMode = false

    var body: some View {
        ToggleRow(label: "Airplane Mode", isOn: $airplaneMode)
    }
}
```

"The `$` prefix creates a Binding from a @State variable. `$airplaneMode` is a `Binding<Bool>` — a two-way reference. The child can read and write the parent's state directly. No callback prop needed."

"In React, you pass a value down and a callback up. In SwiftUI, `@Binding` combines both into a single two-way reference. The `$` prefix is the syntax for creating one."

**@Observable — Shared State Object (like Context + external store):**

"For state shared across multiple views, SwiftUI uses `@Observable` classes. This replaces the older `ObservableObject`/`@Published` pattern."

```swift
@Observable
class ShoppingCart {
    var items: [String] = []
    var total: Double = 0.0

    func addItem(_ item: String, price: Double) {
        items.append(item)
        total += price
    }
}
```

```swift
struct CartBadge: View {
    var cart: ShoppingCart  // just a regular property

    var body: some View {
        Text("\(cart.items.count) items")  // re-renders when items.count changes
    }
}

struct CartTotal: View {
    var cart: ShoppingCart

    var body: some View {
        Text("$\(cart.total, specifier: "%.2f")")  // re-renders when total changes
    }
}
```

"@Observable uses fine-grained observation. SwiftUI tracks which properties each view actually reads. CartBadge only re-renders when `items` changes. CartTotal only re-renders when `total` changes. In React, a context change re-renders every consumer — SwiftUI is more surgical."

"You can pass @Observable objects as plain properties, or inject them via the environment:"

```swift
struct ContentView: View {
    @State private var cart = ShoppingCart()

    var body: some View {
        VStack {
            CartBadge(cart: cart)
            CartTotal(cart: cart)
        }
    }
}
```

**@Environment — System-Provided Values (like useContext for system values):**

```swift
struct AdaptiveView: View {
    @Environment(\.colorScheme) var colorScheme
    @Environment(\.dismiss) var dismiss

    var body: some View {
        VStack {
            Text(colorScheme == .dark ? "Dark Mode" : "Light Mode")
            Button("Close") {
                dismiss()
            }
        }
    }
}
```

"@Environment reads values provided by the system or parent views. `colorScheme` tells you if the user is in dark mode. `dismiss` lets you close a presented sheet. Think of it as `useContext` where the providers are the system and SwiftUI framework."

You can also inject custom @Observable objects via the environment:
```swift
// Inject into the environment
ContentView()
    .environment(cart)

// Read from the environment
struct SomeChildView: View {
    @Environment(ShoppingCart.self) var cart

    var body: some View {
        Text("\(cart.items.count) items")
    }
}
```

**Side Effects — .task(), .onAppear(), .onChange():**

"React uses `useEffect` for side effects. SwiftUI splits this into specific modifiers:"

```swift
struct UserProfile: View {
    @State private var user: User?

    var body: some View {
        Group {
            if let user {
                Text(user.name)
            } else {
                ProgressView()
            }
        }
        .task {
            // Runs when view appears, cancels when view disappears.
            // Like useEffect with [] dependency array, but with automatic cleanup.
            user = try? await fetchUser()
        }
        .onChange(of: selectedTab) {
            // Runs when selectedTab changes.
            // Like useEffect with [selectedTab] dependency array.
            print("Tab changed")
        }
    }
}
```

"`.task {}` is the workhorse. It runs async work tied to the view lifecycle — starts on appear, cancels on disappear. No manual cleanup needed. Use it instead of `.onAppear()` whenever the work is async."

**When to Use Each — Decision Guide:**

Present this decision tree:
- **Does only this view need the state?** → `@State`
- **Does a child need to read AND write a parent's state?** → `@Binding` (parent passes `$stateVar`)
- **Do multiple views need to share a mutable data model?** → `@Observable` class, passed as property or via `.environment()`
- **Do you need a system-provided value (color scheme, locale, dismiss)?** → `@Environment(\.keyPath)`
- **Do you need to run side effects on appear or state change?** → `.task {}` or `.onChange(of:)`

**Common Misconceptions:**
- Misconception: "@State works like React state for everything" → Clarify: "@State is only for local, simple, owned state. For shared state, use @Observable. For passed-down state, use @Binding"
- Misconception: "I can initialize @State from a parent prop and it'll stay in sync" → Clarify: "@State captures the initial value once. Parent updates won't propagate. Use @Binding if you need the child to reflect parent changes"
- Misconception: "@Observable requires @Published like the old ObservableObject" → Clarify: "@Observable (Swift 5.9+) handles observation automatically — no @Published needed. Just mark the class @Observable and use regular properties"
- Misconception: "Bindings are like React refs" → Clarify: "Bindings are two-way state connections, not references to DOM elements. The `$` prefix creates a Binding from a @State variable"

**Verification Questions:**
1. "When would you use @Binding instead of @State?"
2. "What happens if you initialize @State from a parent's prop and the parent updates that prop?"
3. "How does @Observable's observation granularity compare to React context?"
4. Multiple choice: "You have a form view that needs to edit a `User` object owned by its parent. What property wrapper should the form use? A) @State B) @Binding C) @Environment D) @Observable"

**Good answer indicators:**
- They understand @State is for owned local state, @Binding is for state owned elsewhere (B is correct)
- They know @State only captures the initial value
- They can articulate that @Observable provides finer-grained updates than React context

**If they struggle:**
- Map each wrapper back to React: "@State = useState, @Binding = value prop + onChange callback bundled together, @Observable = your Redux store, @Environment = useContext"
- If the $-prefix is confusing: "The dollar sign is just syntax for 'give me a Binding to this state, not the value itself.' It's like passing a reference instead of a copy"
- Walk through a concrete example: a parent with @State, a child with @Binding, showing data flow in both directions

**Exercise 3.1:**
"Build a temperature converter: a parent view with @State for a Fahrenheit value, and a child view that receives a @Binding and shows both Fahrenheit and the computed Celsius value. Include a TextField to edit the temperature."

**How to Guide Them:**
1. First ask: "Which view should own the state? The parent or the child?"
2. If stuck on the Binding: "The parent has `@State private var fahrenheit: Double`. Pass it to the child with `$fahrenheit`"
3. If stuck on computation: "Celsius is a computed value from Fahrenheit — it doesn't need its own state"

**Solution:**
```swift
struct TemperatureConverter: View {
    @State private var fahrenheit: Double = 72

    var body: some View {
        VStack(spacing: 20) {
            Text("Temperature Converter")
                .font(.title)
            TemperatureInput(fahrenheit: $fahrenheit)
        }
        .padding()
    }
}

struct TemperatureInput: View {
    @Binding var fahrenheit: Double

    private var celsius: Double {
        (fahrenheit - 32) * 5 / 9
    }

    var body: some View {
        VStack(alignment: .leading) {
            TextField("Fahrenheit", value: $fahrenheit, format: .number)
                .textFieldStyle(.roundedBorder)
                .keyboardType(.decimalPad)
            Text("\(celsius, specifier: "%.1f")°C")
                .font(.headline)
        }
    }
}
```

---

### Section 4: Lists and Dynamic Content

**Core Concept to Teach:**
SwiftUI's List and ForEach display dynamic data. List provides the scrollable container with platform-appropriate styling. ForEach iterates over data to produce views. Items must be Identifiable so SwiftUI can track them across updates — like React's `key` prop.

**How to Explain:**
1. "In React, you render lists with `.map()` and give each item a `key`. In SwiftUI, you use `ForEach` and conform your data to the `Identifiable` protocol — same concept, different mechanism."
2. "List adds the scrollable, styled container. ForEach just iterates. You can use ForEach inside a List, or ForEach alone inside a ScrollView."

**Identifiable Protocol:**

```swift
struct TodoItem: Identifiable {
    let id = UUID()    // Identifiable requires an 'id' property
    var title: String
    var isComplete: Bool
}
```

"In React, you pass `key={item.id}` to help React track list items. In SwiftUI, the `Identifiable` protocol serves the same purpose — it tells SwiftUI how to identify each item across updates."

**Basic List:**

```swift
struct TodoList: View {
    @State private var todos = [
        TodoItem(title: "Buy groceries", isComplete: false),
        TodoItem(title: "Walk the dog", isComplete: true),
        TodoItem(title: "Write SwiftUI code", isComplete: false),
    ]

    var body: some View {
        List(todos) { todo in
            HStack {
                Image(systemName: todo.isComplete ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(todo.isComplete ? .green : .gray)
                Text(todo.title)
            }
        }
    }
}
```

"This is like React's `todos.map(todo => <TodoRow key={todo.id} ... />)` wrapped in a scroll container with dividers and platform styling."

**ForEach for More Control:**

```swift
var body: some View {
    List {
        ForEach(todos) { todo in
            TodoRow(todo: todo)
        }
        .onDelete { indexSet in
            todos.remove(atOffsets: indexSet)
        }
    }
}
```

"When you use ForEach explicitly inside a List, you can attach modifiers like `.onDelete` for swipe-to-delete. This is a common pattern."

**Swipe Actions:**

```swift
List {
    ForEach(todos) { todo in
        TodoRow(todo: todo)
            .swipeActions(edge: .trailing) {
                Button(role: .destructive) {
                    deleteTodo(todo)
                } label: {
                    Label("Delete", systemImage: "trash")
                }
            }
            .swipeActions(edge: .leading) {
                Button {
                    toggleComplete(todo)
                } label: {
                    Label("Complete", systemImage: "checkmark")
                }
                .tint(.green)
            }
    }
}
```

**Sections and Grouped Lists:**

```swift
List {
    Section("Incomplete") {
        ForEach(todos.filter { !$0.isComplete }) { todo in
            TodoRow(todo: todo)
        }
    }
    Section("Complete") {
        ForEach(todos.filter { $0.isComplete }) { todo in
            TodoRow(todo: todo)
        }
    }
}
.listStyle(.insetGrouped)  // iOS Settings-style grouped sections
```

"Sections group list content with headers and footers. `.listStyle(.insetGrouped)` gives you the iOS Settings look."

**Pull to Refresh:**

```swift
List(todos) { todo in
    TodoRow(todo: todo)
}
.refreshable {
    await loadTodos()  // async function to reload data
}
```

"One modifier. That's it. No RefreshControl setup, no scroll offset tracking. SwiftUI handles the UI and the gesture."

**Common Misconceptions:**
- Misconception: "I can use indices as IDs like React's `key={index}`" → Clarify: "Technically you can use indices with `ForEach(0..<items.count)`, but it causes the same problems as index keys in React — insertions and deletions confuse the diffing. Use Identifiable"
- Misconception: "List and ForEach are the same thing" → Clarify: "List is the scrollable container with styling. ForEach is the iteration. You can use ForEach inside a ScrollView without a List for custom layouts"
- Misconception: "I need to manage scroll position manually" → Clarify: "List handles scrolling automatically. For programmatic scroll control, use ScrollViewReader"

**Verification Questions:**
1. "Why does SwiftUI require list items to be Identifiable, and what's the React equivalent?"
2. "When would you use ForEach inside a List vs just passing data directly to List?"
3. Multiple choice: "What does `.listStyle(.insetGrouped)` do? A) Makes the list horizontal B) Adds the iOS Settings-style grouped appearance C) Enables infinite scrolling D) Removes list separators"

**Good answer indicators:**
- They connect Identifiable to React's key prop
- They know ForEach inside List enables features like .onDelete (B is correct)
- They understand List provides platform styling

**If they struggle:**
- "Think of List as a `<ul>` with built-in scroll and styling, and ForEach as `.map()`. You usually use them together."
- If Identifiable is confusing: "It's just a protocol that says 'I have a unique id property.' Add `let id = UUID()` and you're done."

**Exercise 4.1:**
"Build a list of contacts with sections for Favorites and Everyone Else. Each contact should show a name, phone icon, and have swipe-to-delete."

**How to Guide Them:**
1. "Start by defining a Contact struct that's Identifiable"
2. "Create sample data with an `isFavorite` property"
3. "Use Section with filtered ForEach for each group"
4. "Add `.onDelete` or `.swipeActions` for deletion"

---

### Section 5: Navigation

**Core Concept to Teach:**
SwiftUI uses NavigationStack for hierarchical navigation (push/pop), sheets for modal presentation, and TabView for tab-based navigation. NavigationStack replaced the deprecated NavigationView in iOS 16 and supports programmatic navigation via navigation paths.

**How to Explain:**
1. "If you've used React Router, NavigationStack is the closest equivalent — but it's more opinionated. There's no route config file. Navigation is driven by views and state."
2. "SwiftUI has three navigation patterns: push (drill down), modal (sheets), and tabs. Each has its own mechanism."

**NavigationStack and NavigationLink:**

```swift
struct ContactsList: View {
    let contacts: [Contact]

    var body: some View {
        NavigationStack {
            List(contacts) { contact in
                NavigationLink(value: contact) {
                    Text(contact.name)
                }
            }
            .navigationTitle("Contacts")
            .navigationDestination(for: Contact.self) { contact in
                ContactDetail(contact: contact)
            }
        }
    }
}

struct ContactDetail: View {
    let contact: Contact

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(contact.name)
                .font(.largeTitle)
            Text(contact.email)
            Text(contact.phone)
        }
        .padding()
        .navigationTitle(contact.name)
    }
}
```

"NavigationStack wraps your content and manages a stack of views. NavigationLink pushes a new view when tapped. `.navigationDestination(for:)` defines what view to show for a given data type — like a route definition."

**Programmatic Navigation:**

```swift
struct AppNavigation: View {
    @State private var path = NavigationPath()

    var body: some View {
        NavigationStack(path: $path) {
            Button("Go to Settings") {
                path.append("settings")  // push programmatically
            }
            .navigationDestination(for: String.self) { value in
                if value == "settings" {
                    SettingsView()
                }
            }
        }
    }
}
```

"Bind a `NavigationPath` to the stack and you can push/pop programmatically by appending or removing from the path. This is like `navigate('/settings')` in React Router, but the path is a state array you control."

**Sheets and Full-Screen Covers:**

```swift
struct MainView: View {
    @State private var showSettings = false
    @State private var selectedContact: Contact?

    var body: some View {
        VStack {
            Button("Open Settings") {
                showSettings = true
            }
        }
        .sheet(isPresented: $showSettings) {
            SettingsView()  // slides up as a modal sheet
        }
        .sheet(item: $selectedContact) { contact in
            // Shows when selectedContact is non-nil,
            // dismisses when it becomes nil
            ContactDetail(contact: contact)
        }
    }
}
```

"Sheets are modals that slide up from the bottom. `.sheet(isPresented:)` is toggled by a Bool. `.sheet(item:)` is triggered by setting an optional value — the sheet shows when it's non-nil and dismisses when it becomes nil."

**Dismissing a Sheet from Inside:**

```swift
struct SettingsView: View {
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationStack {
            Form { /* settings content */ }
                .navigationTitle("Settings")
                .toolbar {
                    Button("Done") {
                        dismiss()
                    }
                }
        }
    }
}
```

**Alerts:**

```swift
struct DeleteConfirmation: View {
    @State private var showAlert = false

    var body: some View {
        Button("Delete Account") {
            showAlert = true
        }
        .alert("Are you sure?", isPresented: $showAlert) {
            Button("Delete", role: .destructive) {
                deleteAccount()
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("This action cannot be undone.")
        }
    }
}
```

**TabView:**

```swift
struct AppTabView: View {
    var body: some View {
        TabView {
            Tab("Home", systemImage: "house") {
                HomeView()
            }
            Tab("Search", systemImage: "magnifyingglass") {
                SearchView()
            }
            Tab("Profile", systemImage: "person") {
                ProfileView()
            }
        }
    }
}
```

"TabView creates a tab bar at the bottom of the screen. Each Tab defines a tab with a label, icon, and content view. Wrap each tab's content in its own NavigationStack if it needs independent navigation."

**Comparison to React Router:**

| React Router | SwiftUI |
|-------------|---------|
| `<BrowserRouter>` | `NavigationStack` |
| `<Route path="/detail/:id">` | `.navigationDestination(for: Type.self)` |
| `<Link to="/detail/1">` | `NavigationLink(value: item)` |
| `navigate('/path')` | `path.append(value)` |
| Modal route / portal | `.sheet()` |
| Tab layout | `TabView` |

**Common Misconceptions:**
- Misconception: "NavigationView is the current API" → Clarify: "NavigationView is deprecated. Use NavigationStack (iOS 16+)"
- Misconception: "Sheets need a navigation stack inside them" → Clarify: "Only if you want a title bar or toolbar in the sheet. A sheet can contain any view"
- Misconception: "I can navigate from anywhere without NavigationStack" → Clarify: "NavigationLink and .navigationDestination only work inside a NavigationStack. Sheets work anywhere"

**Verification Questions:**
1. "How does NavigationStack compare to React Router's BrowserRouter?"
2. "When would you use a sheet instead of a NavigationLink?"
3. "How do you dismiss a sheet from inside it?"
4. Multiple choice: "What's the role of `.navigationDestination(for:)`? A) It creates a NavigationLink B) It defines what view to show when a value of that type is pushed C) It sets the navigation title D) It creates a tab"

**Good answer indicators:**
- They can map NavigationStack to React Router concepts
- They know sheets are for modal presentation, NavigationLink for hierarchical drill-down (B is correct)
- They know `@Environment(\.dismiss)` for sheet dismissal

**If they struggle:**
- "Think of NavigationStack as your router, NavigationLink as your Link component, and .navigationDestination as your route definition"
- If programmatic navigation is confusing: "NavigationPath is just an array. Append to push, remove to pop. It's state-driven navigation"

**Exercise 5.1:**
"Build a two-screen app: a list of items and a detail screen. Tapping an item pushes to detail. The detail screen shows the item's info and has a 'Edit' button that presents a sheet."

**How to Guide Them:**
1. "Start with NavigationStack wrapping a List"
2. "Use NavigationLink(value:) with .navigationDestination"
3. "In the detail view, add a @State Bool for the sheet and a Button that toggles it"

---

### Section 6: User Input and Forms

**Core Concept to Teach:**
SwiftUI provides a Form container for settings-style UIs and a set of input controls (TextField, Picker, DatePicker, Toggle, Slider, Stepper). All input controls use @Binding to connect to state. Validation is handled manually — there's no built-in form validation framework like Formik.

**How to Explain:**
1. "If you've built a settings page or a sign-up form in React, Form is SwiftUI's equivalent container. It automatically groups inputs into a platform-appropriate layout."
2. "Every input control takes a Binding (`$stateVar`) so it can read and write state. This is always the same pattern."

**Form and Section:**

```swift
struct SettingsView: View {
    @State private var username = ""
    @State private var notificationsOn = true
    @State private var selectedColor = "Blue"
    @State private var fontSize: Double = 14
    let colors = ["Red", "Blue", "Green", "Purple"]

    var body: some View {
        Form {
            Section("Profile") {
                TextField("Username", text: $username)
                Toggle("Enable Notifications", isOn: $notificationsOn)
            }
            Section("Appearance") {
                Picker("Accent Color", selection: $selectedColor) {
                    ForEach(colors, id: \.self) { color in
                        Text(color)
                    }
                }
                Slider(value: $fontSize, in: 10...24, step: 1) {
                    Text("Font Size")
                }
            }
        }
    }
}
```

"Form creates the grouped, scrollable settings layout. Section creates visual groups with optional headers. Every input control connects to @State with a Binding."

**Text Input Controls:**

```swift
Section("Account") {
    TextField("Email", text: $email)
        .textContentType(.emailAddress)
        .keyboardType(.emailAddress)
        .autocorrectionDisabled()
        .textInputAutocapitalization(.never)

    SecureField("Password", text: $password)
        .textContentType(.password)

    TextEditor(text: $bio)
        .frame(height: 100)
}
```

"TextField is single-line, SecureField masks input (for passwords), TextEditor is multi-line. `.textContentType` helps the system offer autofill suggestions."

**Picker Styles:**

```swift
// Default: navigates to a selection list
Picker("Sort By", selection: $sortOrder) {
    Text("Name").tag(SortOrder.name)
    Text("Date").tag(SortOrder.date)
    Text("Size").tag(SortOrder.size)
}

// Segmented control
Picker("View Mode", selection: $viewMode) {
    Text("List").tag(ViewMode.list)
    Text("Grid").tag(ViewMode.grid)
}
.pickerStyle(.segmented)

// DatePicker
DatePicker("Birthday", selection: $birthday, displayedComponents: .date)

// Stepper
Stepper("Quantity: \(quantity)", value: $quantity, in: 1...99)
```

**Validation Patterns:**

"SwiftUI has no built-in form validation. You handle it yourself with computed properties and conditional UI:"

```swift
struct SignUpForm: View {
    @State private var email = ""
    @State private var password = ""
    @State private var confirmPassword = ""

    private var isEmailValid: Bool {
        email.contains("@") && email.contains(".")
    }

    private var isPasswordValid: Bool {
        password.count >= 8
    }

    private var passwordsMatch: Bool {
        password == confirmPassword
    }

    private var canSubmit: Bool {
        isEmailValid && isPasswordValid && passwordsMatch
    }

    var body: some View {
        Form {
            Section {
                TextField("Email", text: $email)
                    .textContentType(.emailAddress)
                    .keyboardType(.emailAddress)
                if !email.isEmpty && !isEmailValid {
                    Text("Enter a valid email address")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }
            Section {
                SecureField("Password", text: $password)
                if !password.isEmpty && !isPasswordValid {
                    Text("Password must be at least 8 characters")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
                SecureField("Confirm Password", text: $confirmPassword)
                if !confirmPassword.isEmpty && !passwordsMatch {
                    Text("Passwords don't match")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }
            Section {
                Button("Sign Up") {
                    submitForm()
                }
                .disabled(!canSubmit)
            }
        }
    }

    private func submitForm() {
        // handle sign up
    }
}
```

"Validation is computed properties. Show errors conditionally. Disable the submit button until valid. No Formik, no Yup, no form library — just state and computed values."

**Common Misconceptions:**
- Misconception: "Form is required for input controls" → Clarify: "Form is a container that provides settings-style layout. Input controls work anywhere — in a VStack, a List, or standalone"
- Misconception: "There's a built-in validation framework" → Clarify: "Validation is manual. Use computed properties and conditional views. This is simpler than it sounds for most forms"
- Misconception: "Picker always shows a dropdown" → Clarify: "Picker's visual style depends on context and `.pickerStyle()`. Inside a Form it navigates to a selection list by default. You can force `.segmented`, `.wheel`, or `.menu` styles"

**Verification Questions:**
1. "How do all SwiftUI input controls connect to state?"
2. "How would you disable a button until form validation passes?"
3. Multiple choice: "What does SecureField do differently from TextField? A) It validates input B) It encrypts the stored string C) It masks the displayed text for sensitive input D) It prevents copy-paste"

**Good answer indicators:**
- They know all controls use @Binding via the $ prefix
- They understand disabling with `.disabled(!canSubmit)` where canSubmit is computed (C is correct)
- They can describe a manual validation approach

**If they struggle:**
- "Every input control follows the same pattern: give it a label and a `$binding`. That's it."
- If validation seems overwhelming: "Start with just one field and one validation rule. It's just a computed Bool and an if statement."

**Exercise 6.1:**
"Build a settings form with: a text field for display name, a toggle for dark mode, a picker for language (English, Spanish, French), a slider for text size, and a 'Save' button that's disabled if the display name is empty."

**How to Guide Them:**
1. "Create @State for each setting"
2. "Wrap everything in a Form with Sections"
3. "Add a computed property for validation"
4. "Disable the save button with `.disabled()`"

---

## Practice Project

**Project Introduction:**
"Let's put everything together. You're going to build a Bookmarks app — a multi-screen app where users can save, view, and organize bookmarks."

**Requirements:**
Present these one at a time:
1. "Create a `Bookmark` model with: title, url, notes (optional), category, dateAdded. Make it Identifiable."
2. "Build a list screen showing all bookmarks grouped by category (use Sections). Support swipe-to-delete."
3. "Tapping a bookmark pushes to a detail view showing all its properties."
4. "Add an 'Add Bookmark' button that presents a sheet with a form to create a new bookmark."
5. "The form should validate that title and URL are not empty before allowing save."
6. "Wrap the main list and a second tab (maybe 'Favorites' or 'Recent') in a TabView."

**Skills Exercised:**
- Structs, Identifiable, state management (@State, @Binding)
- List, ForEach, Sections, swipe actions
- NavigationStack, NavigationLink, .navigationDestination
- Sheets, @Environment(\.dismiss)
- Form, TextField, Picker, validation
- TabView

**Scaffolding Strategy:**
1. **If they want to try alone**: Let them work. Offer to answer questions.
2. **If they want guidance**: Work through it step by step, starting with the data model.
3. **If they're unsure**: "Start with the Bookmark struct and some sample data. Then build the list. We'll add navigation and the form after."

**Checkpoints During Project:**
- After data model: "Show me your Bookmark struct. Does it conform to Identifiable?"
- After list: "Can you see your bookmarks grouped by category?"
- After navigation: "Does tapping a bookmark show the detail view?"
- After form: "Can you add a new bookmark? Does validation work?"
- After tabs: "Do both tabs work independently?"

**Code Review Approach:**
When reviewing their work:
1. Check the data model: "Is Bookmark a struct? Does it have a UUID id?"
2. Check state management: "Where does the bookmark array live? Is it @State in the right place?"
3. Check the form: "Are Bindings correct? Does validation disable the save button?"
4. Ask about decisions: "Why did you put the NavigationStack here?"
5. Suggest improvements: "Could you extract the bookmark row into its own view for reuse?"

**If They Get Stuck:**
- On data model: "Start with a struct with the properties listed. Add `: Identifiable` and `let id = UUID()`"
- On sections: "Filter your bookmarks array by category, then use a Section for each category"
- On navigation: "Wrap the List in a NavigationStack. Use NavigationLink(value:) and .navigationDestination"
- On the form sheet: "Add `@State private var showAddSheet = false` and a button that sets it true. Use `.sheet(isPresented:)`"
- On validation: "Add computed properties like `var canSave: Bool { !title.isEmpty && !url.isEmpty }` and use `.disabled(!canSave)`"

**Extension Ideas if They Finish Early:**
- Add search with `.searchable()` modifier
- Add pull-to-refresh that simulates loading from a server
- Add edit functionality (tap edit on detail to present a pre-filled form)
- Use @Observable for the bookmark store instead of @State array

---

## Wrap-Up and Next Steps

**Review Key Takeaways:**
"Let's review what you learned today:"
- SwiftUI views are structs that describe UI declaratively — same paradigm as React components
- Modifiers wrap views in layers — order matters (no CSS equivalent)
- @State for local state, @Binding for parent-child two-way connections, @Observable for shared models, @Environment for system values
- List + ForEach for dynamic data, with Identifiable replacing React's key prop
- NavigationStack for push navigation, sheets for modals, TabView for tabs
- Forms with manual validation using computed properties

**Ask them to explain one concept:**
"Walk me through how @State and @Binding work together between a parent and child view."
(This reinforces the most important state management pattern)

**Assess Confidence:**
"On a scale of 1-10, how confident do you feel building a SwiftUI app from scratch?"

**Respond based on answer:**
- 1-4: "That's normal — SwiftUI has a lot of surface area. The state management and modifier system are the biggest mental shifts from React. Keep building small views and the patterns will solidify"
- 5-7: "Good progress. You've got the core patterns. Build something real — the edge cases you hit will teach you more than any tutorial"
- 8-10: "Solid foundation. You're ready to build a real app. Next up: app architecture patterns, data persistence, and the places where SwiftUI meets UIKit"

**Suggest Next Steps:**
Based on their progress and interests:
- "To practice: Rebuild a simple React app you've already built in SwiftUI"
- "For architecture: Check out the ios-app-patterns route for structuring real apps"
- "For data: Learn about SwiftData for persistence (ios-data-persistence route)"
- "For UIKit interop: The uikit-essentials route covers wrapping UIKit views in SwiftUI"
- "For reference: Apple's SwiftUI tutorials (developer.apple.com/tutorials/swiftui) are excellent hands-on walkthroughs"

**Encourage Questions:**
"Do you have any questions about anything we covered?"
"What felt most different from React?"
"Is there anything you'd like me to explain with a different example?"

---

## Adaptive Teaching Strategies

### If Learner is Struggling

**Signs:**
- Confused about property wrappers (@State vs @Binding vs @Observable)
- Modifier order doesn't click
- Can't translate React patterns to SwiftUI equivalents

**Strategies:**
- Slow down and map every new concept to its React equivalent first
- Focus on @State only until it clicks, then layer @Binding, then @Observable
- For modifiers, have them experiment in Xcode previews — change order and see the result
- Do exercises together rather than independently
- Build one tiny view at a time instead of jumping to full screens
- Check if Swift itself is the barrier — if closures or protocols are confusing, address that first

### If Learner is Excelling

**Signs:**
- Completes exercises quickly
- Asks about custom modifiers, ViewModifier protocol, or architecture
- Already thinking about how to structure a real app

**Strategies:**
- Move at faster pace, less explanation
- Introduce custom ViewModifier and view extensions for reusable styling
- Discuss @Observable patterns for app-wide state
- Show GeometryReader for responsive layouts
- Introduce animation basics (`.animation()`, `withAnimation {}`)
- Discuss when to reach for UIKit via UIViewRepresentable
- Challenge: "Build a mini app with at least 3 screens, shared state, and a form"

### If Learner Seems Disengaged

**Signs:**
- Short responses
- Not asking questions
- Skipping exercises

**Strategies:**
- Check in: "How are you feeling about this? Is the pace too slow/fast?"
- Connect to their app idea: "How would you use this for the app you want to build?"
- Skip ahead to something they're interested in (navigation, lists, forms)
- Make it more hands-on: build something immediately, explain concepts as they come up
- If React comparisons aren't helping: "Would you prefer I explain without the React analogies?"

### Different Learning Styles

**Visual learners:**
- Use Xcode previews heavily — modify code and see results live
- Describe modifier stacking as visual layers
- Draw out the NavigationStack as a stack of cards

**Hands-on learners:**
- Less explanation upfront, get them building immediately
- "Try adding a modifier and see what happens"
- Learn through building small views and composing them

**Conceptual learners:**
- Explain SwiftUI's diffing algorithm and how it decides what to re-render
- Discuss why views are structs (value semantics, performance)
- Compare SwiftUI's observation system to React's reconciliation
- Explain the View protocol and what `some View` really means

---

## Troubleshooting Common Issues

### Xcode Preview Not Working
- Make sure the canvas is open (Editor > Canvas or Cmd+Option+Return)
- Check for compile errors — previews won't render with errors
- Try cleaning the build folder (Cmd+Shift+K)
- Make sure there's a `#Preview` block in the file
- Restart Xcode if all else fails

### "@State Not Updating the View"
- Check that the variable is marked `@State`
- Make sure you're modifying it (not a copy)
- If initialized from parent data, remember @State captures only the initial value
- Check that the view actually reads the state variable in `body`

### "Binding Error" or "Cannot Convert"
- Make sure you're using `$` prefix when passing to child views or input controls
- Check that the parent actually has `@State` — you can't create a Binding from a plain property
- For previews, use `.constant(value)` to create a static Binding: `ToggleRow(isOn: .constant(true))`

### NavigationLink Not Working
- Must be inside a NavigationStack
- If using `NavigationLink(value:)`, make sure there's a matching `.navigationDestination(for: Type.self)`
- Check that the value type matches between the link and destination

### List Not Updating After Mutation
- Make sure the data source is `@State` or `@Observable`
- Check that items are Identifiable with stable IDs
- If using `ForEach(array.indices)`, switch to Identifiable items — index-based ForEach doesn't handle mutations well

### Sheet Not Dismissing
- Use `@Environment(\.dismiss)` inside the sheet's view
- Make sure the Bool or optional that drives the sheet is being set correctly
- Don't try to dismiss from outside the sheet — the sheet controls its own dismissal

### Modifier Has No Visible Effect
- Check modifier order — `.padding()` before `.background()` vs after produces different results
- Some modifiers only work in certain contexts (`.navigationTitle` only works inside a NavigationStack)
- Check that you're applying the modifier to the right view (not the wrong level of nesting)

---

## Teaching Notes

**Key Emphasis Points:**
- The modifier wrapping model is the biggest conceptual shift from CSS — keep reinforcing it
- State management is where most bugs happen. Make sure they understand @State vs @Binding before building anything complex
- The @State initialization gotcha (captures initial value only) trips up everyone coming from React — address it proactively
- Lean on React comparisons early, but point out where analogies break down

**Pacing Guidance:**
- Don't rush Section 1 (mental model) — if they don't understand views as structs and the modifier chain, everything else will be confusing
- Section 3 (state management) is the most critical section — spend the most time here
- Sections 4-6 can move faster once state management clicks
- Allow plenty of time for the practice project — building ties everything together

**Success Indicators:**
You'll know they've got it when they:
- Correctly choose between @State, @Binding, and @Observable without prompting
- Predict the visual outcome of modifier ordering changes
- Build a new view from scratch without looking at examples
- Ask questions like "how would I share this state across screens?"
- Start thinking about extracting reusable views
- Stop asking "what's the SwiftUI equivalent of..." and start thinking in SwiftUI directly

**Most Common Confusion Points:**
1. **Modifier order**: Why `.padding().background()` and `.background().padding()` differ
2. **@State vs @Binding**: When to use each, and how `$` creates a Binding
3. **@State initialization**: Why it captures the initial value only
4. **NavigationStack structure**: Where to put it, how .navigationDestination works
5. **$-prefix syntax**: When you need it (creating Bindings) vs when you don't (reading values)

**Teaching Philosophy:**
- React experience is a massive advantage — use it. Every new concept should start with "in React you'd do X, in SwiftUI you do Y"
- But watch for where the analogy breaks down and call it out explicitly
- Get them building early. SwiftUI previews make iteration fast — use that
- State management is the heart of SwiftUI. If they get this right, everything else follows
- Don't try to cover everything — focus on the 20% of SwiftUI that handles 80% of real app work
