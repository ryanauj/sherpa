---
title: UIKit Essentials for SwiftUI Developers
route_map: /routes/uikit-essentials/map.md
paired_sherpa: /routes/uikit-essentials/sherpa.md
prerequisites:
  - SwiftUI fundamentals
  - Swift language basics
  - Xcode basics
topics:
  - UIKit
  - UIViewController
  - UIKit Integration
  - Storyboards
---

# UIKit Essentials for SwiftUI Developers

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/uikit-essentials/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/uikit-essentials/map.md` for the high-level overview.

## Overview

UIKit is Apple's original UI framework for iOS, shipping with the first iPhone SDK in 2008. SwiftUI was built to replace it, but UIKit is still everywhere — in existing codebases, third-party libraries, Stack Overflow answers, Apple documentation, and for features SwiftUI doesn't fully support yet. This guide gives you enough UIKit literacy to read UIKit code confidently, integrate UIKit components into your SwiftUI apps, and work on projects that mix both frameworks.

This isn't about becoming a UIKit developer. Think of it as learning to read a second language — you don't need to write novels in it, but you need to understand what you're looking at.

## Learning Objectives

By the end of this guide, you will be able to:
- Explain UIKit's imperative paradigm and contrast it with SwiftUI's declarative model
- Read and understand UIKit code in existing projects and documentation
- Describe the UIViewController lifecycle and what each method does
- Identify common UIKit components and map them to their SwiftUI equivalents
- Wrap UIKit views and view controllers for use in SwiftUI
- Implement the Coordinator pattern for handling UIKit delegates
- Embed SwiftUI views in UIKit apps using UIHostingController

## Prerequisites

Before starting, you should have:
- SwiftUI fundamentals — views, state management, modifiers, navigation (see swiftui-fundamentals route)
- Swift language basics — classes, protocols, closures, optionals (see swift-for-developers route)
- Xcode installed and familiarity with running apps in the simulator (see xcode-essentials route)

This guide assumes you think in SwiftUI and uses it as the reference point for explaining UIKit concepts.

## Setup

No special setup needed beyond what you already have for SwiftUI development. You'll be working in the same Xcode projects you're used to. When we wrap UIKit components, we'll do it inside SwiftUI apps.

For code examples, you can create a new iOS App project in Xcode (SwiftUI interface) and paste examples into your views, or create new Swift files as needed.

---

## Section 1: UIKit Mental Model

### Imperative vs Declarative

You already know declarative UI from SwiftUI: you describe what the UI should look like for a given state, and the framework handles the rest. UIKit works the opposite way — you imperatively create UI elements, configure them, position them, and manually update them when state changes.

Here's a simple counter in SwiftUI:

```swift
struct CounterView: View {
    @State private var count = 0

    var body: some View {
        VStack {
            Text("Count: \(count)")
                .font(.title)
            Button("Increment") {
                count += 1
            }
        }
        .padding()
    }
}
```

And the same thing in UIKit:

```swift
class CounterViewController: UIViewController {
    private var count = 0
    private let countLabel = UILabel()

    override func viewDidLoad() {
        super.viewDidLoad()

        // Create and configure the label
        countLabel.text = "Count: \(count)"
        countLabel.font = .preferredFont(forTextStyle: .title1)

        // Create and configure the button
        let button = UIButton(type: .system)
        button.setTitle("Increment", for: .normal)
        button.addTarget(self, action: #selector(incrementTapped),
                        for: .touchUpInside)

        // Arrange in a stack
        let stack = UIStackView(arrangedSubviews: [countLabel, button])
        stack.axis = .vertical
        stack.alignment = .center
        stack.spacing = 8
        stack.translatesAutoresizingMaskIntoConstraints = false

        // Add to the view hierarchy and position
        view.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            stack.centerYAnchor.constraint(equalTo: view.centerYAnchor)
        ])
    }

    @objc private func incrementTapped() {
        count += 1
        countLabel.text = "Count: \(count)"  // Manual UI update!
    }
}
```

Notice the differences:
- **Class, not struct** — UIKit views are reference types that persist in memory. SwiftUI views are value types that get recreated.
- **Manual setup** — You create each element, set its properties, add it to the view hierarchy, and position it with constraints.
- **Manual updates** — When `count` changes, you have to update the label yourself. In SwiftUI, changing `@State` triggers an automatic re-render. In UIKit, changing a property does nothing visible until you explicitly push the change to the UI.
- **Target-action for events** — Instead of a closure (`Button("Tap") { ... }`), UIKit uses `addTarget` to connect a UI control to a method. The `@objc` attribute is required because this mechanism uses the Objective-C runtime.

This manual update responsibility is the root cause of most UIKit bugs you'll encounter in older code — someone forgot to update the UI when state changed, or updated it inconsistently.

### UIView and UIViewController

UIKit has two fundamental building blocks:

**UIView** — A rectangular area on screen that draws content and handles touch events. Everything visible in a UIKit app is a UIView or a subclass of it (UILabel, UIButton, UIImageView, etc.). Views form a tree hierarchy — a parent view contains child views, similar to the DOM.

**UIViewController** — Manages a screen's worth of content. Each view controller has a root `view` property (a UIView) and handles the lifecycle of that screen — when it loads, appears, disappears, and is destroyed. Think of it as a React page component that also manages mounting and unmounting.

In practice, a UIKit app is a tree of view controllers, each managing a tree of views.

### The View Controller Lifecycle

View controllers have lifecycle methods that get called at specific points. This is the most important UIKit concept to understand, because you'll see these methods in every UIKit codebase:

```
init → loadView → viewDidLoad → viewWillAppear → viewDidAppear
                                                         ↓
                               viewWillDisappear ← viewDidDisappear
```

Here's what each one does:

| Lifecycle Method | When It's Called | SwiftUI Equivalent |
|---|---|---|
| `viewDidLoad()` | Once, when the view loads into memory | Initial body evaluation |
| `viewWillAppear(_:)` | Every time the view is about to appear | `onAppear` |
| `viewDidAppear(_:)` | After the view is visible on screen | `onAppear` (roughly) |
| `viewWillDisappear(_:)` | When the view is about to leave screen | `onDisappear` |
| `viewDidDisappear(_:)` | After the view is no longer visible | `onDisappear` (roughly) |

The critical distinction: `viewDidLoad` runs once. `viewWillAppear` runs every time the screen appears — including when the user navigates back to it. This is a frequent source of bugs when developers put one-time setup in `viewWillAppear` or data refresh in `viewDidLoad`.

```swift
class ProfileViewController: UIViewController {
    override func viewDidLoad() {
        super.viewDidLoad()
        // One-time setup: create UI elements, configure layout
        setupUI()
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        // Runs every time this screen appears
        // Good for refreshing data that might have changed
        refreshProfile()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        // Runs when leaving the screen
        // Good for saving state or stopping timers
        saveProgress()
    }
}
```

Always call `super` when overriding lifecycle methods — the parent class has its own setup and teardown work to do.

### The Responder Chain

UIKit has a concept called the responder chain — a hierarchy of objects that can handle events. When you tap the screen, the event starts at the most specific view under your finger and travels up through parent views until something handles it. You usually don't interact with the responder chain directly, but it explains why UIKit uses the target-action pattern and delegation rather than closures — these were designed around Objective-C patterns from 2008.

### Storyboards and Interface Builder

You'll occasionally encounter references to Storyboards and Interface Builder (IB). These are visual tools in Xcode for building UIKit interfaces by dragging and dropping components. They store the UI as XML files (`.storyboard` or `.xib`).

Many teams have moved away from storyboards in favor of programmatic UIKit or SwiftUI, but you'll see them in older projects. The key annotations to recognize:

```swift
// @IBOutlet — a connection from code to a UI element in a storyboard
@IBOutlet weak var nameLabel: UILabel!

// @IBAction — a method triggered by a UI element in a storyboard
@IBAction func submitTapped(_ sender: UIButton) {
    // Handle the tap
}
```

If you see `@IBOutlet` or `@IBAction` in code, it means the UI was built visually in a storyboard or xib file. You don't need to learn Interface Builder — just recognize these annotations when you encounter them.

### SwiftUI-to-UIKit Quick Reference

Keep this table handy — it maps every major SwiftUI concept to its UIKit equivalent:

| SwiftUI | UIKit |
|---------|-------|
| `View` struct | `UIView` / `UIViewController` class |
| `body` property | `viewDidLoad()` + manual setup |
| State drives UI automatically | You update UI manually |
| `List` | `UITableView` |
| `LazyVGrid` / `LazyHGrid` | `UICollectionView` |
| `NavigationStack` | `UINavigationController` |
| `TabView` | `UITabBarController` |
| `.alert()` modifier | `UIAlertController` |
| `TextField` | `UITextField` |
| Tap gesture modifier | `UITapGestureRecognizer` |
| View modifiers | Setting properties + constraints |
| `@State` / `@Observable` | Properties + manual UI update calls |
| `onAppear` / `onDisappear` | `viewWillAppear` / `viewWillDisappear` |

### Checkpoint 1

Before moving on, make sure you understand:
- [ ] UIKit is imperative — you create, configure, and manually update UI elements
- [ ] UIKit views are classes (persistent objects), not structs (value types)
- [ ] `viewDidLoad` runs once; `viewWillAppear` runs every time the screen appears
- [ ] `@IBOutlet` and `@IBAction` indicate storyboard-connected UI

---

## Section 2: Common UIKit Components

### The Delegation Pattern

Before looking at specific components, you need to understand delegation — UIKit's primary communication pattern. In SwiftUI, you pass closures or bindings to child views. In UIKit, components define protocols (delegates), and you assign an object as the delegate. The component calls the protocol's methods when events happen.

```swift
// UIKit defines a protocol
protocol UITableViewDelegate: AnyObject {
    func tableView(_ tableView: UITableView,
                   didSelectRowAt indexPath: IndexPath)
    // ... many more optional methods
}

// Your view controller adopts it
class MyViewController: UIViewController, UITableViewDelegate {
    func tableView(_ tableView: UITableView,
                   didSelectRowAt indexPath: IndexPath) {
        print("Selected row \(indexPath.row)")
    }
}
```

Think of delegates as callback props in React. Instead of passing `onSelect={(row) => ...}`, you assign `self` as the delegate, and UIKit calls your method. It's more formal and indirect, but the concept is the same.

UIKit also uses a related pattern called **data source**. Some components split their callbacks into two protocols:
- **Delegate** — handles behavior and events (taps, scrolling, sizing)
- **Data source** — provides content (how many rows, what's in each cell)

You'll see both on `UITableView` and `UICollectionView`.

### UITableView (SwiftUI's List)

`UITableView` is the UIKit equivalent of SwiftUI's `List`. It displays a scrollable vertical list of cells. Here's a complete example:

```swift
class FruitsViewController: UIViewController,
    UITableViewDataSource, UITableViewDelegate {

    let fruits = ["Apple", "Banana", "Cherry", "Date", "Elderberry"]

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "Fruits"

        let tableView = UITableView()
        tableView.dataSource = self
        tableView.delegate = self
        tableView.register(UITableViewCell.self,
                          forCellReuseIdentifier: "FruitCell")
        tableView.translatesAutoresizingMaskIntoConstraints = false

        view.addSubview(tableView)
        NSLayoutConstraint.activate([
            tableView.topAnchor.constraint(equalTo: view.topAnchor),
            tableView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            tableView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            tableView.trailingAnchor.constraint(equalTo: view.trailingAnchor)
        ])
    }

    // DATA SOURCE: How many rows?
    func tableView(_ tableView: UITableView,
                   numberOfRowsInSection section: Int) -> Int {
        return fruits.count
    }

    // DATA SOURCE: What's in each row?
    func tableView(_ tableView: UITableView,
                   cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(
            withIdentifier: "FruitCell", for: indexPath)
        cell.textLabel?.text = fruits[indexPath.row]
        return cell
    }

    // DELEGATE: What happens on tap?
    func tableView(_ tableView: UITableView,
                   didSelectRowAt indexPath: IndexPath) {
        print("Selected: \(fruits[indexPath.row])")
        tableView.deselectRow(at: indexPath, animated: true)
    }
}
```

**What's happening here:**
1. `register` tells the table view which cell class to use and gives it a reuse identifier
2. `dataSource = self` and `delegate = self` make this view controller the data provider and event handler
3. `numberOfRowsInSection` tells UIKit how many rows to display
4. `cellForRowAt` creates (or recycles) a cell for each row. `dequeueReusableCell` is UIKit's cell recycling — it reuses cells that have scrolled off screen rather than creating new ones. This is how UIKit handles long lists efficiently.
5. `didSelectRowAt` handles row taps
6. `indexPath` identifies a cell by section and row — most simple lists have one section

The SwiftUI equivalent is roughly:
```swift
List(fruits, id: \.self) { fruit in
    Text(fruit)
}
```

Yes, really. That's the entire table view in SwiftUI.

### UINavigationController (SwiftUI's NavigationStack)

`UINavigationController` manages a stack of view controllers, providing a navigation bar and back button. In SwiftUI, `NavigationStack` handles this declaratively. In UIKit, you push and pop view controllers explicitly:

```swift
// Create a navigation controller with a root view controller
let rootVC = FruitsViewController()
let navController = UINavigationController(rootViewController: rootVC)

// Push a new screen onto the stack (like NavigationLink)
let detailVC = FruitDetailViewController()
navigationController?.pushViewController(detailVC, animated: true)

// Pop back (usually handled by the back button automatically)
navigationController?.popViewController(animated: true)

// Set the navigation title (like .navigationTitle())
title = "Fruits"
```

### UITabBarController (SwiftUI's TabView)

`UITabBarController` provides a tab bar at the bottom of the screen, switching between child view controllers:

```swift
let tabBarController = UITabBarController()
tabBarController.viewControllers = [
    UINavigationController(rootViewController: homeVC),
    UINavigationController(rootViewController: searchVC),
    UINavigationController(rootViewController: profileVC)
]

// Configure tab bar items
homeVC.tabBarItem = UITabBarItem(
    title: "Home",
    image: UIImage(systemName: "house"),
    selectedImage: UIImage(systemName: "house.fill")
)
searchVC.tabBarItem = UITabBarItem(
    title: "Search",
    image: UIImage(systemName: "magnifyingglass"),
    selectedImage: nil
)
```

Same concept as SwiftUI's `TabView`, but you create the controller, assign child view controllers, and configure tab items imperatively.

### UIAlertController (SwiftUI's .alert())

In SwiftUI, you bind an alert to a boolean with `.alert()`. In UIKit, you create a `UIAlertController`, add actions, and present it:

```swift
let alert = UIAlertController(
    title: "Delete Item?",
    message: "This action cannot be undone.",
    preferredStyle: .alert
)

alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
alert.addAction(UIAlertAction(title: "Delete", style: .destructive) { _ in
    self.deleteItem()
})

present(alert, animated: true)
```

You can also use `.actionSheet` as the preferred style for bottom-sheet style action lists.

### Gesture Recognizers

SwiftUI has gesture modifiers (`.onTapGesture`, `.gesture(DragGesture())`). UIKit uses gesture recognizer objects that you attach to views:

```swift
// Tap gesture
let tap = UITapGestureRecognizer(
    target: self, action: #selector(viewTapped))
someView.addGestureRecognizer(tap)

@objc func viewTapped() {
    print("View was tapped")
}

// Long press gesture
let longPress = UILongPressGestureRecognizer(
    target: self, action: #selector(viewLongPressed))
longPress.minimumPressDuration = 0.5
someView.addGestureRecognizer(longPress)
```

Same target-action pattern as buttons. Create a recognizer, attach it to a view, point it at a method.

### Exercise 2.1: Read UIKit Code

This exercise tests your ability to read UIKit code — the core skill this route teaches.

**Task:** Look at the following UIKit code and answer: What does this screen display? What happens when you interact with it?

```swift
class ColorsViewController: UIViewController,
    UITableViewDataSource, UITableViewDelegate {

    let colors: [(name: String, color: UIColor)] = [
        ("Red", .systemRed),
        ("Blue", .systemBlue),
        ("Green", .systemGreen),
        ("Purple", .systemPurple)
    ]

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "Colors"

        let tableView = UITableView()
        tableView.dataSource = self
        tableView.delegate = self
        tableView.register(UITableViewCell.self,
                          forCellReuseIdentifier: "ColorCell")
        tableView.translatesAutoresizingMaskIntoConstraints = false

        view.addSubview(tableView)
        NSLayoutConstraint.activate([
            tableView.topAnchor.constraint(equalTo: view.topAnchor),
            tableView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            tableView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            tableView.trailingAnchor.constraint(equalTo: view.trailingAnchor)
        ])
    }

    func tableView(_ tableView: UITableView,
                   numberOfRowsInSection section: Int) -> Int {
        return colors.count
    }

    func tableView(_ tableView: UITableView,
                   cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(
            withIdentifier: "ColorCell", for: indexPath)
        let item = colors[indexPath.row]
        cell.textLabel?.text = item.name
        cell.backgroundColor = item.color.withAlphaComponent(0.2)
        return cell
    }

    func tableView(_ tableView: UITableView,
                   didSelectRowAt indexPath: IndexPath) {
        let item = colors[indexPath.row]
        view.backgroundColor = item.color
        tableView.deselectRow(at: indexPath, animated: true)
    }
}
```

<details>
<summary>Hint 1: Focus on the data source methods first</summary>
The data source methods tell you what the table displays. `numberOfRowsInSection` returns `colors.count` (4 rows). `cellForRowAt` sets each cell's text and background.
</details>

<details>
<summary>Hint 2: Then look at the delegate method</summary>
`didSelectRowAt` is the tap handler. It sets `view.backgroundColor` — that's the root view behind the table.
</details>

<details>
<summary>Solution</summary>

This screen displays a scrollable list with four rows: "Red", "Blue", "Green", and "Purple". Each row has a lightly tinted background matching its color (20% opacity). The navigation title is "Colors".

When you tap a row, the background color of the entire screen changes to that color (behind the table). The row selection highlight animates away via `deselectRow`.

The SwiftUI equivalent would be roughly:
```swift
struct ColorsView: View {
    let colors: [(name: String, color: Color)] = [
        ("Red", .red), ("Blue", .blue),
        ("Green", .green), ("Purple", .purple)
    ]
    @State private var bgColor: Color = .clear

    var body: some View {
        List(colors, id: \.name) { item in
            Text(item.name)
                .listRowBackground(item.color.opacity(0.2))
                .onTapGesture { bgColor = item.color }
        }
        .navigationTitle("Colors")
        .background(bgColor)
    }
}
```
</details>

### Checkpoint 2

Before moving on, make sure you understand:
- [ ] Delegation is UIKit's primary communication pattern — structured callback protocols
- [ ] `UITableView` uses a data source (content) and delegate (events)
- [ ] `dequeueReusableCell` recycles cells for performance
- [ ] Navigation in UIKit is explicit — you push and pop view controllers
- [ ] You can read a UIKit view controller and describe what it displays

---

## Section 3: UIKit in SwiftUI

This is the most practically useful section. Even if you build everything in SwiftUI, you'll occasionally need a UIKit component — for camera access, web views, maps with features SwiftUI doesn't expose, or third-party UIKit libraries.

### UIViewRepresentable

`UIViewRepresentable` is a protocol for wrapping a `UIView` subclass so you can use it in SwiftUI. It has two required methods:

```swift
struct ActivityIndicator: UIViewRepresentable {
    var isAnimating: Bool

    func makeUIView(context: Context) -> UIActivityIndicatorView {
        // Called once — create and return the UIKit view
        let indicator = UIActivityIndicatorView(style: .large)
        return indicator
    }

    func updateUIView(_ uiView: UIActivityIndicatorView, context: Context) {
        // Called when SwiftUI state changes — update the UIKit view
        if isAnimating {
            uiView.startAnimating()
        } else {
            uiView.stopAnimating()
        }
    }
}
```

**How it works:**
- `makeUIView(context:)` — Creates the UIKit view. Called once when the SwiftUI view first appears. Think of it as the constructor.
- `updateUIView(_:context:)` — Called whenever SwiftUI state changes that affects this view. This is the bridge: SwiftUI tells you state changed, and you manually update the UIKit view. Same idea as UIKit's manual updates, but triggered by SwiftUI's state system.

Using it in SwiftUI is seamless:

```swift
struct LoadingView: View {
    @State private var isLoading = true

    var body: some View {
        VStack(spacing: 20) {
            ActivityIndicator(isAnimating: isLoading)

            Button(isLoading ? "Stop" : "Start") {
                isLoading.toggle()
            }
        }
    }
}
```

### The Coordinator Pattern

Many UIKit components communicate through delegates. SwiftUI doesn't have delegates. The **Coordinator** bridges this gap — it's a class that acts as the UIKit component's delegate and forwards events back to SwiftUI via `@Binding`.

Here's a UISearchBar wrapper that demonstrates the full pattern:

```swift
struct SearchBar: UIViewRepresentable {
    @Binding var text: String
    var placeholder: String

    func makeUIView(context: Context) -> UISearchBar {
        let searchBar = UISearchBar()
        searchBar.placeholder = placeholder
        searchBar.delegate = context.coordinator  // Connect delegate
        return searchBar
    }

    func updateUIView(_ uiView: UISearchBar, context: Context) {
        uiView.text = text  // Push SwiftUI state to UIKit
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, UISearchBarDelegate {
        var parent: SearchBar

        init(_ parent: SearchBar) {
            self.parent = parent
        }

        func searchBar(_ searchBar: UISearchBar,
                       textDidChange searchText: String) {
            parent.text = searchText  // Push UIKit event to SwiftUI
        }

        func searchBarSearchButtonClicked(_ searchBar: UISearchBar) {
            searchBar.resignFirstResponder()  // Dismiss keyboard
        }
    }
}
```

**The data flow works in two directions:**
1. **SwiftUI → UIKit**: When `text` changes in SwiftUI, `updateUIView` is called, which sets `uiView.text`
2. **UIKit → SwiftUI**: When the user types, UIKit calls `searchBar(_:textDidChange:)` on the Coordinator, which sets `parent.text`, updating the `@Binding`

**About the Coordinator:**
- `makeCoordinator()` is called before `makeUIView`, creating the Coordinator instance
- The Coordinator is a class (reference type) that persists for the lifetime of the view
- It conforms to `NSObject` (required by many UIKit delegate protocols) and the delegate protocol
- It holds a reference to the parent struct to access `@Binding` properties

**Using it in SwiftUI:**
```swift
struct SearchableList: View {
    @State private var searchText = ""
    let items = ["Apple", "Banana", "Cherry", "Date", "Elderberry"]

    var filteredItems: [String] {
        if searchText.isEmpty { return items }
        return items.filter {
            $0.localizedCaseInsensitiveContains(searchText)
        }
    }

    var body: some View {
        VStack {
            SearchBar(text: $searchText, placeholder: "Search fruits...")
            List(filteredItems, id: \.self) { item in
                Text(item)
            }
        }
    }
}
```

### UIViewControllerRepresentable

Same pattern as `UIViewRepresentable`, but for wrapping `UIViewController` subclasses. The most common use case is system pickers and controllers:

```swift
struct ImagePicker: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    @Environment(\.dismiss) var dismiss

    func makeUIViewController(context: Context)
        -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = .photoLibrary
        return picker
    }

    func updateUIViewController(
        _ uiViewController: UIImagePickerController,
        context: Context) {
        // No updates needed — the picker manages its own state
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject,
        UIImagePickerControllerDelegate,
        UINavigationControllerDelegate {

        var parent: ImagePicker

        init(_ parent: ImagePicker) {
            self.parent = parent
        }

        func imagePickerController(
            _ picker: UIImagePickerController,
            didFinishPickingMediaWithInfo
                info: [UIImagePickerController.InfoKey: Any]
        ) {
            if let image = info[.originalImage] as? UIImage {
                parent.selectedImage = image
            }
            parent.dismiss()
        }

        func imagePickerControllerDidCancel(
            _ picker: UIImagePickerController) {
            parent.dismiss()
        }
    }
}
```

**Key points:**
- `makeUIViewController` / `updateUIViewController` — same pattern as UIViewRepresentable
- The Coordinator conforms to both `UIImagePickerControllerDelegate` and `UINavigationControllerDelegate` (UIImagePickerController requires both)
- On image selection, the Coordinator writes to the `@Binding` and dismisses
- `updateUIViewController` is empty because the picker manages its own internal state — we only care about the final result

**Using it in SwiftUI:**
```swift
struct PhotoPickerView: View {
    @State private var selectedImage: UIImage?
    @State private var showingPicker = false

    var body: some View {
        VStack(spacing: 20) {
            if let image = selectedImage {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 300)
            } else {
                ContentUnavailableView("No Photo",
                    systemImage: "photo",
                    description: Text("Tap the button to pick a photo"))
            }

            Button("Choose Photo") {
                showingPicker = true
            }
        }
        .padding()
        .sheet(isPresented: $showingPicker) {
            ImagePicker(selectedImage: $selectedImage)
        }
    }
}
```

### When Do You Actually Need This?

Before wrapping a UIKit component, check if SwiftUI has a native version. Apple adds more SwiftUI components with each release. Here are cases where UIKit wrapping is still necessary or beneficial:

| Need | Why UIKit? |
|------|-----------|
| Camera capture | `UIImagePickerController` with `.camera` source type |
| Web content | `WKWebView` — SwiftUI has no native web view |
| Advanced maps | `MKMapView` when SwiftUI's `Map` doesn't expose features you need |
| Document scanning | `VNDocumentCameraViewController` |
| Rich text editing | `UITextView` for features `TextEditor` doesn't support |
| Third-party libraries | Any library that provides UIKit views or view controllers |
| Mail compose | `MFMailComposeViewController` |

### Exercise 3.1: Wrap a UIKit View

**Task:** Create a `UIViewRepresentable` wrapper for `UISwitch` (UIKit's toggle switch) that binds to a `Bool` value. Then use it in a SwiftUI view alongside a native SwiftUI `Toggle` to verify they both work.

<details>
<summary>Hint 1: Structure</summary>

You need a struct conforming to `UIViewRepresentable` with a `@Binding var isOn: Bool`. Implement `makeUIView` and `updateUIView`. Since `UISwitch` uses target-action (not delegation), you'll need a Coordinator with a target-action method.
</details>

<details>
<summary>Hint 2: Target-action in the Coordinator</summary>

In `makeUIView`, use `addTarget` to connect the switch's `.valueChanged` event to a method on the Coordinator:
```swift
uiSwitch.addTarget(context.coordinator,
                   action: #selector(Coordinator.switchChanged(_:)),
                   for: .valueChanged)
```
The Coordinator method needs `@objc` since target-action uses the Objective-C runtime.
</details>

<details>
<summary>Hint 3: Coordinator implementation</summary>

The Coordinator should inherit from `NSObject`, hold a reference to the parent, and have an `@objc` method that reads `sender.isOn` and writes it to `parent.isOn`.
</details>

<details>
<summary>Solution</summary>

```swift
struct WrappedSwitch: UIViewRepresentable {
    @Binding var isOn: Bool

    func makeUIView(context: Context) -> UISwitch {
        let uiSwitch = UISwitch()
        uiSwitch.addTarget(
            context.coordinator,
            action: #selector(Coordinator.switchChanged(_:)),
            for: .valueChanged)
        return uiSwitch
    }

    func updateUIView(_ uiView: UISwitch, context: Context) {
        uiView.isOn = isOn
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject {
        var parent: WrappedSwitch

        init(_ parent: WrappedSwitch) {
            self.parent = parent
        }

        @objc func switchChanged(_ sender: UISwitch) {
            parent.isOn = sender.isOn
        }
    }
}

// Test it alongside a native Toggle
struct SwitchComparisonView: View {
    @State private var isEnabled = false

    var body: some View {
        Form {
            HStack {
                Text("UIKit Switch")
                Spacer()
                WrappedSwitch(isOn: $isEnabled)
            }
            Toggle("SwiftUI Toggle", isOn: $isEnabled)
            Text("Value: \(isEnabled ? "ON" : "OFF")")
        }
    }
}
```

Both the UIKit switch and SwiftUI toggle control the same `@State` variable. Toggling either one updates both, proving the two-way binding works correctly.
</details>

### Exercise 3.2: Wrap a View Controller

**Task:** Create a `UIViewControllerRepresentable` wrapper for `UIColorPickerViewController` — a system color picker. It should use a `@Binding var selectedColor: Color` and dismiss itself when the user finishes.

<details>
<summary>Hint 1: Protocol conformance</summary>

The Coordinator needs to conform to `UIColorPickerViewControllerDelegate`. The key delegate method is `colorPickerViewController(_:didSelect:continuously:)`.
</details>

<details>
<summary>Hint 2: Color conversion</summary>

`UIColorPickerViewController` works with `UIColor`. You need to convert to SwiftUI's `Color` type. Use `Color(uiColor:)` for the conversion.
</details>

<details>
<summary>Solution</summary>

```swift
struct ColorPicker: UIViewControllerRepresentable {
    @Binding var selectedColor: Color
    @Environment(\.dismiss) var dismiss

    func makeUIViewController(context: Context)
        -> UIColorPickerViewController {
        let picker = UIColorPickerViewController()
        picker.delegate = context.coordinator
        picker.selectedColor = UIColor(selectedColor)
        return picker
    }

    func updateUIViewController(
        _ uiViewController: UIColorPickerViewController,
        context: Context) {
        uiViewController.selectedColor = UIColor(selectedColor)
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject,
        UIColorPickerViewControllerDelegate {

        var parent: ColorPicker

        init(_ parent: ColorPicker) {
            self.parent = parent
        }

        func colorPickerViewController(
            _ viewController: UIColorPickerViewController,
            didSelect color: UIColor,
            continuously: Bool
        ) {
            parent.selectedColor = Color(uiColor: color)
        }

        func colorPickerViewControllerDidFinish(
            _ viewController: UIColorPickerViewController
        ) {
            parent.dismiss()
        }
    }
}

// Usage
struct ColorPickerDemo: View {
    @State private var chosenColor: Color = .blue
    @State private var showingPicker = false

    var body: some View {
        VStack(spacing: 20) {
            RoundedRectangle(cornerRadius: 12)
                .fill(chosenColor)
                .frame(width: 200, height: 200)

            Button("Pick Color") {
                showingPicker = true
            }
        }
        .sheet(isPresented: $showingPicker) {
            ColorPicker(selectedColor: $chosenColor)
        }
    }
}
```

Note: SwiftUI actually has its own `ColorPicker` view. This exercise is for practicing the wrapping pattern — in a real app, you'd use SwiftUI's native version.
</details>

### Checkpoint 3

Before moving on, make sure you understand:
- [ ] `UIViewRepresentable` wraps UIKit views with `makeUIView` and `updateUIView`
- [ ] `UIViewControllerRepresentable` wraps view controllers with the same pattern
- [ ] The Coordinator bridges UIKit delegates/target-action to SwiftUI `@Binding`
- [ ] Data flows two ways: SwiftUI → `updateUIView` → UIKit, and UIKit → Coordinator → `@Binding` → SwiftUI
- [ ] Check for native SwiftUI components before wrapping UIKit ones

---

## Section 4: SwiftUI in UIKit

This section covers the reverse direction: embedding SwiftUI views inside a UIKit app. This is relevant if you're working on a codebase that's primarily UIKit and wants to adopt SwiftUI incrementally.

### UIHostingController

`UIHostingController` wraps any SwiftUI view and turns it into a regular `UIViewController`. It's the simplest bridging mechanism in either direction:

```swift
import SwiftUI

struct SettingsView: View {
    var body: some View {
        Form {
            Section("Account") {
                Text("Email: user@example.com")
                Text("Plan: Premium")
            }
            Section("Preferences") {
                Toggle("Notifications", isOn: .constant(true))
                Toggle("Dark Mode", isOn: .constant(false))
            }
        }
    }
}

// Anywhere in UIKit code:
let settingsVC = UIHostingController(rootView: SettingsView())
navigationController?.pushViewController(settingsVC, animated: true)
```

That's it. Wrap your SwiftUI view in `UIHostingController`, and it becomes a regular view controller you can push, present, or embed.

### Embedding as a Child View Controller

When you want a SwiftUI view as part of a UIKit screen (not a full screen), use child view controller containment:

```swift
class DashboardViewController: UIViewController {
    override func viewDidLoad() {
        super.viewDidLoad()

        // 1. Create the hosting controller
        let chartView = UIHostingController(rootView: SwiftUIChartView())

        // 2. Add as child view controller
        addChild(chartView)

        // 3. Add the view and set up constraints
        chartView.view.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(chartView.view)
        NSLayoutConstraint.activate([
            chartView.view.leadingAnchor.constraint(
                equalTo: view.leadingAnchor),
            chartView.view.trailingAnchor.constraint(
                equalTo: view.trailingAnchor),
            chartView.view.topAnchor.constraint(
                equalTo: view.safeAreaLayoutGuide.topAnchor),
            chartView.view.heightAnchor.constraint(
                equalToConstant: 300)
        ])

        // 4. Notify the child that it moved
        chartView.didMove(toParent: self)
    }
}
```

The four steps for child view controller containment:
1. Create the `UIHostingController`
2. Call `addChild()` to establish the parent-child relationship
3. Add the hosted view to the hierarchy and set constraints
4. Call `didMove(toParent:)` to complete the transition

### Passing Data to SwiftUI

Pass data into SwiftUI views through their initializer, just like you would in a pure SwiftUI app:

```swift
struct UserProfileView: View {
    let username: String
    let onLogout: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            Text("Welcome, \(username)")
                .font(.title)
            Button("Log Out", role: .destructive, action: onLogout)
        }
    }
}

// In UIKit:
let profileVC = UIHostingController(
    rootView: UserProfileView(
        username: currentUser.name,
        onLogout: { [weak self] in
            self?.handleLogout()
        }
    )
)
```

Use `[weak self]` in closures to avoid retain cycles between the UIKit view controller and the SwiftUI view.

### Gradual Migration Strategy

If you're working on a UIKit codebase that wants to adopt SwiftUI:

1. **Start with leaf screens** — settings, about, profile — that don't depend on complex UIKit integration
2. **Use UIHostingController** to embed them in the existing UIKit navigation
3. **Move to feature screens** — replace more screens with SwiftUI over time
4. **Flip the container** — eventually make SwiftUI the root app structure and wrap remaining UIKit screens with `UIViewControllerRepresentable`

This is how most large apps are migrating, including many of Apple's own apps. UIKit and SwiftUI coexist with minimal overhead.

### Checkpoint 4

Before moving on, make sure you understand:
- [ ] `UIHostingController` wraps a SwiftUI view into a `UIViewController`
- [ ] You can push, present, or embed a UIHostingController like any view controller
- [ ] Child view controller containment has four steps: addChild, add view, set constraints, didMove
- [ ] `@State`, `@Observable`, `@Environment` — all SwiftUI state management works inside hosted views

---

## Practice Project

### Project Description

Build a SwiftUI app that wraps `UIImagePickerController` to let users pick photos from their library and display them. This integrates everything from the route: you'll use `UIViewControllerRepresentable` to bridge UIKit into SwiftUI, implement a Coordinator for delegation, and handle the two-way data flow.

This project will help you:
- Apply UIViewControllerRepresentable in practice
- Implement the Coordinator pattern for delegate handling
- Manage the bridge between UIKit events and SwiftUI state

### Requirements

Build a SwiftUI app that:
1. Shows a "Choose Photo" button and an empty state when no photo is selected
2. Tapping the button presents a `UIImagePickerController` as a sheet
3. After the user picks a photo, it displays in the main view
4. The user can pick a different photo by tapping the button again
5. Handle cancellation (dismissing the picker without selecting)

### Getting Started

**Step 1: Create a new project**
Create a new iOS App project in Xcode with SwiftUI interface.

**Step 2: Plan your approach**
You need three pieces:
1. The `UIViewControllerRepresentable` wrapper for `UIImagePickerController`
2. A `Coordinator` that handles `UIImagePickerControllerDelegate`
3. A SwiftUI view that presents the picker and displays the result

**Step 3: Build the wrapper**
Start with the `UIViewControllerRepresentable` struct. You'll need:
- A `@Binding var selectedImage: UIImage?`
- Access to `@Environment(\.dismiss)` to close the picker
- `makeUIViewController` that creates and configures the picker
- `makeCoordinator` and a `Coordinator` class

**Step 4: Build the SwiftUI view**
Create the main view with:
- `@State` for the selected image and picker presentation
- A conditional display (image or placeholder)
- A button that toggles the sheet
- A `.sheet` modifier presenting your wrapper

### Hints and Tips

<details>
<summary>If you're not sure how to structure the Representable</summary>

Look back at the ImagePicker example in Section 3 — it's the same component. Try building it from memory first, then refer back if needed.
</details>

<details>
<summary>If delegate methods aren't being called</summary>

Make sure you:
1. Set `picker.delegate = context.coordinator` in `makeUIViewController`
2. Your Coordinator conforms to both `UIImagePickerControllerDelegate` and `UINavigationControllerDelegate`
3. Your Coordinator inherits from `NSObject`
</details>

<details>
<summary>If the image doesn't appear after picking</summary>

Check that your Coordinator is writing to `parent.selectedImage` in the `didFinishPickingMediaWithInfo` delegate method. The key to extract the image is `.originalImage`:
```swift
if let image = info[.originalImage] as? UIImage {
    parent.selectedImage = image
}
```
</details>

### Example Solution

<details>
<summary>Click to see one possible solution</summary>

```swift
import SwiftUI

// UIKit wrapper
struct ImagePicker: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    @Environment(\.dismiss) var dismiss

    func makeUIViewController(context: Context)
        -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.delegate = context.coordinator
        picker.sourceType = .photoLibrary
        return picker
    }

    func updateUIViewController(
        _ uiViewController: UIImagePickerController,
        context: Context
    ) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject,
        UIImagePickerControllerDelegate,
        UINavigationControllerDelegate {

        var parent: ImagePicker

        init(_ parent: ImagePicker) {
            self.parent = parent
        }

        func imagePickerController(
            _ picker: UIImagePickerController,
            didFinishPickingMediaWithInfo
                info: [UIImagePickerController.InfoKey: Any]
        ) {
            if let image = info[.originalImage] as? UIImage {
                parent.selectedImage = image
            }
            parent.dismiss()
        }

        func imagePickerControllerDidCancel(
            _ picker: UIImagePickerController
        ) {
            parent.dismiss()
        }
    }
}

// Main SwiftUI view
struct ContentView: View {
    @State private var selectedImage: UIImage?
    @State private var showingPicker = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                if let image = selectedImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 400)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .shadow(radius: 4)
                } else {
                    ContentUnavailableView(
                        "No Photo Selected",
                        systemImage: "photo.on.rectangle",
                        description: Text(
                            "Tap the button below to choose a photo"))
                }

                Button {
                    showingPicker = true
                } label: {
                    Label(
                        selectedImage == nil
                            ? "Choose Photo" : "Change Photo",
                        systemImage: "photo.on.rectangle.angled"
                    )
                }
                .buttonStyle(.borderedProminent)
            }
            .padding()
            .navigationTitle("Photo Picker")
            .sheet(isPresented: $showingPicker) {
                ImagePicker(selectedImage: $selectedImage)
            }
        }
    }
}
```

**Key points in this solution:**
- The ImagePicker struct handles all UIKit bridging — the SwiftUI view doesn't know or care that UIKit is involved
- The Coordinator handles both selection and cancellation
- `@Environment(\.dismiss)` is used to close the sheet from the Coordinator
- The main view uses `@State` to track both the image and picker visibility
- `ContentUnavailableView` provides a clean empty state (available iOS 17+)
</details>

### Extending the Project

If you want to go further:
- Add source type selection — let the user choose between camera (on a real device) and photo library
- Add a "Remove Photo" button that clears the selection
- Wrap `WKWebView` to display a web page alongside the photo
- Add multiple photo selection using `PHPickerViewController` instead

---

## Summary

### Key Takeaways

- **UIKit is imperative**: You create objects, configure them, position them with constraints, and manually update them when state changes. SwiftUI handles all of this automatically.
- **View controllers manage screens**: They have a lifecycle (`viewDidLoad`, `viewWillAppear`, etc.) that you override to run code at the right time.
- **Delegation is everywhere**: UIKit components communicate through delegate protocols — structured callbacks that your code implements.
- **UIViewRepresentable bridges UIKit into SwiftUI**: Two methods (`makeUIView`, `updateUIView`) plus a Coordinator for delegate handling.
- **UIHostingController bridges SwiftUI into UIKit**: Wrap any SwiftUI view to use it as a regular view controller.
- **You don't need to master UIKit**: Literacy — reading it, understanding the patterns, integrating when necessary — is the goal.

### Skills You've Gained

You can now:
- Read UIKit code and explain what it does
- Map UIKit components to their SwiftUI equivalents
- Wrap UIKit views and view controllers for use in SwiftUI
- Implement Coordinators for UIKit delegate handling
- Embed SwiftUI views in UIKit apps
- Decide when UIKit integration is necessary vs when SwiftUI suffices

### Self-Assessment

Take a moment to reflect:
- Can you look at a UIKit view controller and describe what it displays?
- Do you understand the Coordinator pattern well enough to wrap a UIKit component you haven't seen before?
- Could you explain to someone why a UIKit codebase might want to adopt SwiftUI incrementally?

---

## Next Steps

### Continue Learning

Ready for more? Here are your next options:

**Build on this topic:**
- Practice wrapping more UIKit components — `WKWebView`, `MKMapView`, `MFMailComposeViewController`
- Browse UIKit code on GitHub to practice reading — search for `UITableViewDataSource` implementations

**Explore related routes:**
- [iOS App Patterns](/routes/ios-app-patterns/map.md) — Architecture patterns (MVC, MVVM) for structuring larger apps
- [iOS Data Persistence](/routes/ios-data-persistence/map.md) — Local data storage with Core Data, UserDefaults, and more

### Additional Resources

**Documentation:**
- Apple's UIKit documentation — particularly the UIViewController lifecycle
- Apple's "Interfacing with UIKit" SwiftUI tutorial

**Practice:**
- Wrap `WKWebView` in UIViewRepresentable — a practical exercise combining views, delegates, and navigation delegates
- Wrap `MKMapView` with annotations — shows how to bridge more complex UIKit APIs with multiple delegate methods
