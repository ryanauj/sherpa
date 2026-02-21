---
title: iOS App Patterns
route_map: /routes/ios-app-patterns/map.md
paired_guide: /routes/ios-app-patterns/guide.md
topics:
  - MVVM
  - Architecture
  - Dependency Injection
  - SwiftUI Patterns
---

# iOS App Patterns - Sherpa (AI Teaching Guide)

**Purpose**: This sherpa guide helps AI assistants teach iOS app architecture patterns to developers building SwiftUI apps. It focuses on practical patterns the iOS community has settled on — MVVM, repository pattern, dependency injection — with comparisons to web architecture where helpful.

**Route Map**: See `/routes/ios-app-patterns/map.md` for the high-level overview of this route.
**Paired Guide**: The human-focused content is at `/routes/ios-app-patterns/guide.md`.

---

## Teaching Overview

### Learning Objectives
By the end of this session, the learner should be able to:
- Identify when a SwiftUI view is doing too much and needs architectural separation
- Structure a SwiftUI app using MVVM with @Observable view models
- Build a data layer using the repository pattern with protocol abstractions
- Make network requests with URLSession and async/await, mapping responses to models with Codable
- Apply dependency injection using initializer injection and @Environment
- Implement navigation patterns with NavigationStack and navigation paths
- Represent and handle loading, success, and error states
- Organize code by feature using Swift packages for modularization

### Prior Sessions
Before starting, check `.sessions/index.md` and `.sessions/ios-app-patterns/` for prior session history. If the learner has completed previous sessions on this route, review the summaries to understand what they've covered and pick up where they left off.

### Prerequisites to Verify
Before starting, verify the learner has:
- SwiftUI fundamentals (views, @State, @Binding, @Observable, @Environment, navigation)
- Swift language proficiency (protocols, generics, closures, async/await, error handling)
- Basic understanding of how apps communicate with servers (REST APIs, JSON)

**If prerequisites are missing**: If SwiftUI is weak, suggest the swiftui-fundamentals route. If Swift generics or protocols are shaky, suggest reviewing those sections in swift-for-developers. If async/await is unfamiliar, cover it briefly in the networking section.

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

If no configuration file exists, use defaults (objective tone, mixed assessments, balanced pacing).

### Assessment Strategies

Use a combination of assessment types:

**Multiple Choice Questions:**
- Good for checking understanding of where logic belongs (view vs view model vs repository)
- Example: "Where should network request logic live? A) In the SwiftUI view B) In the view model C) In a repository D) In the model struct"

**Code Review Questions:**
- Show a monolithic SwiftUI view and ask the learner to identify what should be extracted
- Present two architectures and ask which is more testable and why

**Design Questions:**
- Give a feature description and ask the learner to sketch the architecture (which models, view models, repositories)
- These assess the ability to apply patterns, not just recognize them

**Mixed Approach (Recommended):**
- Use multiple choice for quick checks on responsibilities
- Use code review for identifying architectural problems
- Use design questions for the practice project planning phase

---

## Web-to-iOS Architecture Reference

Use this table for learners with web development experience:

| Web Concept | iOS/SwiftUI Equivalent |
|-------------|----------------------|
| React component | SwiftUI View |
| Container/smart component | View + ViewModel |
| Presentational/dumb component | SwiftUI View (no business logic) |
| Redux store / Zustand store | @Observable ViewModel |
| API service / fetch wrapper | Repository |
| Context Provider | @Environment |
| React Router | NavigationStack |
| Custom hook | ViewModel method or computed property |
| Express middleware | Protocol-based dependency injection |
| `useEffect` data fetching | `.task` modifier + ViewModel |
| TypeScript interface | Swift protocol |
| npm package | Swift package |

---

## Teaching Flow

### Introduction

**What to Cover:**
- Single-screen demos are easy. Multi-screen apps with shared state, networking, persistence, and tests require structure.
- The iOS community has mostly settled on MVVM for SwiftUI apps — not because it's universally perfect, but because it maps well to how SwiftUI works
- This route is practical, not theoretical. Every pattern exists to solve a specific problem.
- Goal: build apps that are testable, maintainable, and organized

**Opening Questions to Assess Level:**
1. "Have you worked on a SwiftUI app with more than a few screens? What was hard about keeping it organized?"
2. "Are you familiar with architectural patterns like MVC or MVVM from web development or other platforms?"
3. "What's the biggest SwiftUI app you've built? Did you run into any pain points with how the code was structured?"

**Adapt based on responses:**
- If they've built larger apps: Focus on formalizing patterns they've likely intuited. Ask about specific pain points and address those.
- If new to architecture: Take time with the "why" before the "how." Show the problem first, then the solution.
- If they know MVC/MVVM from web: Bridge heavily to what they know. The concepts are the same; the Swift/SwiftUI mechanics are different.
- If they've only built small demos: Start with the motivation — show how a small app becomes painful without structure.

**Opening framing:**
"When an app is one or two screens, you can put everything in the view and it works fine. But apps grow. You add networking, persistence, authentication, navigation between screens that share data. If all that logic lives in your SwiftUI views, you end up with views that are 500 lines long, untestable, and painful to modify. Architecture patterns exist to prevent that. We'll focus on patterns the iOS community actually uses — not theoretical purity, but practical solutions to real problems."

---

### Section 1: Why Architecture Matters

**Core Concept to Teach:**
SwiftUI makes it easy to put everything in the view — state, networking, business logic, navigation. This works for demos but creates problems at scale: views become bloated, logic is untestable, and changes ripple unpredictably.

**How to Explain:**
1. Show a realistic "everything in the view" example and point out the problems
2. Identify the separate concerns mixed together
3. Explain what "separation of concerns" means in this context
4. Preview the architecture they'll learn

**The Problem — A Monolithic View:**

```swift
struct BookListView: View {
    @State private var books: [Book] = []
    @State private var isLoading = false
    @State private var errorMessage: String?
    @State private var searchText = ""

    var filteredBooks: [Book] {
        if searchText.isEmpty { return books }
        return books.filter { $0.title.localizedCaseInsensitiveContains(searchText) }
    }

    var body: some View {
        NavigationStack {
            Group {
                if isLoading {
                    ProgressView()
                } else if let error = errorMessage {
                    Text(error)
                } else {
                    List(filteredBooks) { book in
                        NavigationLink(book.title) {
                            BookDetailView(book: book)
                        }
                    }
                }
            }
            .searchable(text: $searchText)
            .navigationTitle("Books")
            .task {
                await loadBooks()
            }
        }
    }

    func loadBooks() async {
        isLoading = true
        errorMessage = nil
        do {
            let url = URL(string: "https://api.example.com/books")!
            let (data, _) = try await URLSession.shared.data(from: url)
            books = try JSONDecoder().decode([Book].self, from: data)
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }
}
```

**Walk Through the Problems:**
- "This works, but look at what this single view is responsible for: state management, networking, JSON decoding, error handling, loading states, filtering, navigation, and rendering. That's a lot of jobs for one struct."
- "You can't test `loadBooks()` without rendering the view. You can't swap the data source (what if you want to load from a cache instead?). You can't reuse the networking logic in another screen."
- "In web terms: this is like putting your API calls, state management, routing, and rendering all in one React component. It works for a demo, but not for a real app."

**What Separation Looks Like (Preview):**
"By the end of this route, that view will be split into:
- A **Model** (`Book` struct) — just data
- A **ViewModel** (`BookListViewModel`) — state management, filtering, loading orchestration
- A **Repository** (`BookRepository`) — networking and data access
- A **View** (`BookListView`) — pure rendering, no business logic"

**Common Misconceptions:**
- Misconception: "Architecture is about following a pattern perfectly" → Clarify: "Architecture is about solving specific problems — testability, maintainability, separation of concerns. If a pattern doesn't solve a problem you have, don't use it."
- Misconception: "SwiftUI apps need the same architecture as UIKit apps" → Clarify: "SwiftUI's declarative nature and property wrappers mean lighter-weight patterns work well. You don't need a full VIPER or Clean Architecture — MVVM fits SwiftUI naturally."
- Misconception: "Architecture should be decided upfront for the whole app" → Clarify: "Start simple. Extract when complexity demands it. Over-architecting a simple screen is as bad as under-architecting a complex one."

**Verification Questions:**
1. "Looking at the BookListView, what are the different responsibilities mixed into this one view?"
2. "Why is `loadBooks()` hard to test as it is?"
3. "If you wanted to add offline support (load from cache when the network is down), what would you need to change in this view?"

**Good answer indicators:**
- They can identify multiple concerns (networking, state, filtering, rendering)
- They understand that direct URLSession calls in views prevent testing and reuse
- They see that adding caching would mean modifying the view itself

---

### Section 2: MVVM in SwiftUI

**Core Concept to Teach:**
MVVM (Model-View-ViewModel) separates a screen into three parts: the Model (data), the View (rendering), and the ViewModel (state + logic connecting models to views). In SwiftUI, @Observable makes view models reactive — the view automatically updates when the view model's properties change.

**How to Explain:**
1. Define each layer clearly with what belongs in it and what doesn't
2. Show the refactored book list using MVVM
3. Emphasize that the ViewModel owns state and logic, while the View just renders

**The Three Layers:**

"**Model** — Data structures and business rules. Pure Swift, no SwiftUI imports. These are the nouns of your app."
```swift
struct Book: Identifiable, Codable {
    let id: UUID
    let title: String
    let author: String
    let pageCount: Int

    var isLong: Bool { pageCount > 300 }
}
```

"**ViewModel** — Holds the state for a specific screen and contains the logic that transforms models into what the view needs. Uses @Observable so SwiftUI reacts to changes."
```swift
import Observation

@Observable
class BookListViewModel {
    private(set) var books: [Book] = []
    private(set) var isLoading = false
    private(set) var errorMessage: String?
    var searchText = ""

    var filteredBooks: [Book] {
        if searchText.isEmpty { return books }
        return books.filter {
            $0.title.localizedCaseInsensitiveContains(searchText)
        }
    }

    func loadBooks() async {
        isLoading = true
        errorMessage = nil
        do {
            let url = URL(string: "https://api.example.com/books")!
            let (data, _) = try await URLSession.shared.data(from: url)
            books = try JSONDecoder().decode([Book].self, from: data)
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }
}
```

"**View** — Renders the current state. Reads from the ViewModel, calls ViewModel methods for actions. No business logic."
```swift
struct BookListView: View {
    @State private var viewModel = BookListViewModel()

    var body: some View {
        NavigationStack {
            Group {
                if viewModel.isLoading {
                    ProgressView()
                } else if let error = viewModel.errorMessage {
                    ContentUnavailableView("Error",
                        systemImage: "exclamationmark.triangle",
                        description: Text(error))
                } else {
                    List(viewModel.filteredBooks) { book in
                        NavigationLink(book.title) {
                            BookDetailView(book: book)
                        }
                    }
                }
            }
            .searchable(text: $viewModel.searchText)
            .navigationTitle("Books")
            .task {
                await viewModel.loadBooks()
            }
        }
    }
}
```

**Walk Through:**
- "The view is now just rendering. It reads `viewModel.filteredBooks`, `viewModel.isLoading`, `viewModel.errorMessage` — all display concerns."
- "All the logic moved to the ViewModel: loading, filtering, error handling."
- "`@State private var viewModel` — the view owns the ViewModel. Because it's `@Observable`, SwiftUI automatically re-renders when any of its properties change."
- "`private(set)` on ViewModel properties means the view can read them but only the ViewModel can change them. This enforces the direction of data flow."

**The Web Comparison:**
"In React terms: the ViewModel is like a custom hook that manages state and side effects. The View is like a presentational component that just renders props. The Model is your TypeScript types/interfaces."

| MVVM | React Equivalent |
|------|-----------------|
| Model | Type/Interface |
| ViewModel | Custom hook (state + logic) |
| View | Presentational component |
| `@Observable` | The hook's state management |
| `@State private var viewModel` | `const vm = useMyViewModel()` |

**Where Logic Belongs:**

Present this decision framework:
- **Model**: Validation rules, computed properties derived from the model's own data, formatting that's universal (not screen-specific)
- **ViewModel**: Screen-specific state, data loading orchestration, filtering/sorting for a specific view, user action handling
- **View**: Layout, styling, navigation presentation, user input bindings
- **NOT in the View**: Network calls, JSON parsing, business logic, complex state machines

**Common Misconceptions:**
- Misconception: "Every view needs a ViewModel" → Clarify: "Simple views that just display passed-in data don't need one. A `BookRow` that takes a `Book` and renders it is fine on its own. ViewModels are for screens with state and logic."
- Misconception: "The ViewModel should know about SwiftUI" → Clarify: "ViewModels should not import SwiftUI. They use `@Observable` from the Observation framework. This keeps them testable without needing a SwiftUI rendering context."
- Misconception: "MVVM means I need a ViewModel for child views too" → Clarify: "One ViewModel per screen (or per significant feature area) is typical. Child views receive data through their initializer."

**Verification Questions:**
1. "In the refactored BookListView, what would you change to add a 'sort by author' feature? Which layer would the changes go in?"
2. Multiple choice: "Where should a computed property like `filteredBooks` live? A) The Model B) The ViewModel C) The View D) A utility function"
3. "Why does the ViewModel use `private(set)` for most of its properties?"
4. Design question: "You're building a screen that shows a user's profile with an edit button. What would the Model, ViewModel, and View look like?"

**Good answer indicators:**
- Sorting goes in the ViewModel (B for multiple choice)
- They understand `private(set)` enforces one-way data flow
- They can sketch a rough Model/ViewModel/View split for a new feature

**If they struggle:**
- "Think of the ViewModel as the brain of the screen. It decides what data to show and in what state. The View is the face — it just displays what the brain tells it to."
- If the @Observable pattern is confusing: "It works like @State but for classes. SwiftUI watches the properties and re-renders when they change."
- If they're not sure where logic goes: "Ask: 'Could this logic exist without a screen?' If yes, it might belong in the Model or Repository. If it's screen-specific, it's ViewModel."

---

### Section 3: The Data Layer

**Core Concept to Teach:**
The data layer abstracts where data comes from. The ViewModel asks for data; the repository provides it. Whether it comes from the network, a database, or a cache is the repository's concern. This uses the repository pattern with protocol-based abstractions.

**How to Explain:**
1. Show the problem: the ViewModel currently has URLSession code baked in
2. Extract a repository with a protocol interface
3. Show how this enables testing, caching, and swapping data sources

**The Repository Pattern:**

"Right now, `loadBooks()` in the ViewModel calls URLSession directly. This means:
- You can't test the ViewModel without a network connection
- You can't swap in a local cache
- The ViewModel knows about URL construction, JSON decoding — details it shouldn't care about"

Protocol:
```swift
protocol BookRepository {
    func fetchBooks() async throws -> [Book]
    func fetchBook(id: UUID) async throws -> Book
}
```

Real implementation:
```swift
struct RemoteBookRepository: BookRepository {
    private let baseURL = URL(string: "https://api.example.com")!
    private let decoder = JSONDecoder()

    func fetchBooks() async throws -> [Book] {
        let url = baseURL.appendingPathComponent("books")
        let (data, response) = try await URLSession.shared.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw BookError.requestFailed
        }

        return try decoder.decode([Book].self, from: data)
    }

    func fetchBook(id: UUID) async throws -> Book {
        let url = baseURL.appendingPathComponent("books/\(id)")
        let (data, response) = try await URLSession.shared.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw BookError.requestFailed
        }

        return try decoder.decode(Book.self, from: data)
    }
}

enum BookError: LocalizedError {
    case requestFailed
    case notFound

    var errorDescription: String? {
        switch self {
        case .requestFailed: return "Unable to load books. Check your connection."
        case .notFound: return "Book not found."
        }
    }
}
```

Updated ViewModel:
```swift
@Observable
class BookListViewModel {
    private let repository: BookRepository
    private(set) var books: [Book] = []
    private(set) var isLoading = false
    private(set) var errorMessage: String?
    var searchText = ""

    init(repository: BookRepository = RemoteBookRepository()) {
        self.repository = repository
    }

    var filteredBooks: [Book] {
        if searchText.isEmpty { return books }
        return books.filter {
            $0.title.localizedCaseInsensitiveContains(searchText)
        }
    }

    func loadBooks() async {
        isLoading = true
        errorMessage = nil
        do {
            books = try await repository.fetchBooks()
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }
}
```

**Walk Through:**
- "The ViewModel no longer knows about URLs, URLSession, or JSON decoding. It asks the repository for books and gets back `[Book]` or an error."
- "The protocol `BookRepository` defines the contract. Any implementation that conforms to this protocol can be swapped in."
- "The default parameter `RemoteBookRepository()` means the view still creates the ViewModel simply: `BookListViewModel()`. But for tests, you can inject a different repository."
- "In web terms: the repository is like an API service module that wraps `fetch()`. The ViewModel is like a React hook that calls the service."

**Networking with URLSession and Codable:**

"URLSession is Swift's built-in HTTP client. Combined with async/await and Codable, it handles most networking needs:"

```swift
// Basic GET request
let (data, response) = try await URLSession.shared.data(from: url)

// POST request with body
var request = URLRequest(url: url)
request.httpMethod = "POST"
request.setValue("application/json", forHTTPHeaderField: "Content-Type")
request.httpBody = try JSONEncoder().encode(newBook)
let (data, response) = try await URLSession.shared.data(for: request)

// Codable handles JSON ↔ Swift conversion automatically
let books = try JSONDecoder().decode([Book].self, from: data)
```

"Codable is like having a built-in version of Zod or io-ts that validates and transforms JSON. If your struct's properties match the JSON keys, it just works. For mismatched keys, use `CodingKeys`."

**Common Misconceptions:**
- Misconception: "The repository is just a wrapper around URLSession" → Clarify: "It's an abstraction over data access. The current implementation uses URLSession, but a future one could use Core Data, CloudKit, or a local cache — the ViewModel doesn't change."
- Misconception: "I need a generic network layer" → Clarify: "Start with concrete repositories per feature. Extract shared networking helpers only when you see real duplication. Don't build a generic HTTP client unless you need one."
- Misconception: "Protocols add unnecessary complexity" → Clarify: "The protocol is what makes this testable. Without it, you can't swap in a test double. One protocol per feature area is minimal overhead."

**Verification Questions:**
1. "What's the benefit of the ViewModel depending on a `BookRepository` protocol instead of `RemoteBookRepository` directly?"
2. "If you wanted to add caching — load from cache first, then refresh from network — where would that logic live?"
3. Code review: "Is there anything wrong with putting URLSession calls in the ViewModel?"

**Good answer indicators:**
- They understand protocol-based abstraction enables test doubles and alternative implementations
- Caching logic belongs in a repository implementation (or a caching repository that wraps the remote one)
- URLSession in ViewModel couples the view model to networking details

---

### Section 4: Dependency Injection

**Core Concept to Teach:**
Dependency injection means passing dependencies in from outside rather than creating them inside. In iOS/SwiftUI, this is done through initializer injection and `@Environment`. It makes code testable and flexible.

**How to Explain:**
1. Show the problem: hard-coded dependencies prevent testing
2. Demonstrate initializer injection (already started in Section 3)
3. Show @Environment for app-wide dependencies
4. Show how to write a test with injected dependencies

**Initializer Injection (What We Already Did):**
"When we changed the ViewModel to accept `repository: BookRepository` in its init, that was dependency injection. The ViewModel doesn't create its own data source — it receives one."

```swift
// Production: uses the default
let viewModel = BookListViewModel()

// Testing: inject a test double
let viewModel = BookListViewModel(repository: MockBookRepository())
```

**@Environment for App-Wide Dependencies:**

"For dependencies shared across many screens (authentication, settings, analytics), use @Environment:"

```swift
// Define an environment key
struct BookRepositoryKey: EnvironmentKey {
    static let defaultValue: BookRepository = RemoteBookRepository()
}

extension EnvironmentValues {
    var bookRepository: BookRepository {
        get { self[BookRepositoryKey.self] }
        set { self[BookRepositoryKey.self] = newValue }
    }
}

// Inject at the app root
@main
struct BookApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.bookRepository, RemoteBookRepository())
        }
    }
}

// Use in any view
struct BookListView: View {
    @Environment(\.bookRepository) private var repository

    var body: some View {
        // Use repository...
    }
}
```

"In React terms: @Environment is like React Context. You provide a value at the top of the tree, and any descendant can read it without prop drilling."

**Testing with Dependency Injection:**

```swift
struct MockBookRepository: BookRepository {
    var booksToReturn: [Book] = []
    var errorToThrow: Error?

    func fetchBooks() async throws -> [Book] {
        if let error = errorToThrow { throw error }
        return booksToReturn
    }

    func fetchBook(id: UUID) async throws -> Book {
        if let error = errorToThrow { throw error }
        guard let book = booksToReturn.first(where: { $0.id == id }) else {
            throw BookError.notFound
        }
        return book
    }
}

// In tests:
@Test func loadBooksSuccess() async {
    let testBooks = [
        Book(id: UUID(), title: "Swift Basics", author: "Apple", pageCount: 200)
    ]
    let viewModel = BookListViewModel(
        repository: MockBookRepository(booksToReturn: testBooks)
    )

    await viewModel.loadBooks()

    #expect(viewModel.books.count == 1)
    #expect(viewModel.books.first?.title == "Swift Basics")
    #expect(viewModel.isLoading == false)
    #expect(viewModel.errorMessage == nil)
}

@Test func loadBooksFailure() async {
    let viewModel = BookListViewModel(
        repository: MockBookRepository(errorToThrow: BookError.requestFailed)
    )

    await viewModel.loadBooks()

    #expect(viewModel.books.isEmpty)
    #expect(viewModel.errorMessage != nil)
}
```

**Walk Through:**
- "MockBookRepository conforms to the same protocol but returns controlled data. No network needed."
- "The tests verify ViewModel behavior in isolation — loading states, error handling, data mapping."
- "Swift Testing framework uses `@Test` and `#expect` — simpler syntax than XCTest."

**Common Misconceptions:**
- Misconception: "I need a dependency injection framework" → Clarify: "Swift's initializer injection and @Environment cover most cases. Frameworks like Swinject exist, but they add complexity you probably don't need."
- Misconception: "Every dependency needs to be injectable" → Clarify: "Inject things that change (data sources, services) or that prevent testing. Don't inject standard library types or utilities."
- Misconception: "Mocks should replicate the real implementation" → Clarify: "Mocks should be as simple as possible — return controlled data, optionally throw errors. They test your code's response to data, not the data source itself."

**Verification Questions:**
1. "What's the difference between creating a dependency inside a class vs receiving it through the initializer?"
2. "When would you use @Environment vs initializer injection?"
3. "Write a test that verifies the ViewModel's search filtering works correctly."

**Good answer indicators:**
- They understand that internal creation couples the class to a specific implementation
- @Environment is for broadly shared dependencies; initializer injection is for specific, per-instance dependencies
- They can write a test using a mock repository

---

### Section 5: Navigation Patterns

**Core Concept to Teach:**
In small apps, `NavigationLink` destinations inline in the view work fine. In larger apps, you want centralized navigation state — a single source of truth for the navigation stack that can be programmatically controlled.

**How to Explain:**
1. Show the simple inline approach and its limitations
2. Introduce NavigationStack with a navigation path
3. Show type-safe navigation with `navigationDestination`
4. Discuss coordinator-style patterns for complex flows

**NavigationStack with Path:**

```swift
@Observable
class NavigationRouter {
    var path = NavigationPath()

    func showBookDetail(_ book: Book) {
        path.append(book)
    }

    func showAuthor(_ author: Author) {
        path.append(author)
    }

    func popToRoot() {
        path = NavigationPath()
    }
}
```

```swift
struct BookListView: View {
    @State private var router = NavigationRouter()
    @State private var viewModel = BookListViewModel()

    var body: some View {
        NavigationStack(path: $router.path) {
            List(viewModel.filteredBooks) { book in
                Button(book.title) {
                    router.showBookDetail(book)
                }
            }
            .navigationDestination(for: Book.self) { book in
                BookDetailView(book: book)
            }
            .navigationDestination(for: Author.self) { author in
                AuthorDetailView(author: author)
            }
            .navigationTitle("Books")
        }
    }
}
```

**Walk Through:**
- "NavigationPath is a type-erased stack of destinations. You append values to push, and NavigationStack handles rendering the right destination."
- "`.navigationDestination(for:)` registers a mapping: 'when a Book is pushed, show BookDetailView.' Type-safe, no stringly-typed routes."
- "The router centralizes navigation actions. Any view model or view can call `router.showBookDetail(book)` to navigate."
- "In web terms: NavigationPath is like a history stack. The router is like React Router's `useNavigate()`. `navigationDestination` is like route definitions."

**For Complex Flows:**
"For flows like onboarding or checkout where screens must appear in a specific order with branching logic, consider keeping the flow state in a dedicated object:"

```swift
@Observable
class OnboardingFlow {
    var currentStep: OnboardingStep = .welcome

    enum OnboardingStep: Hashable {
        case welcome
        case nameEntry
        case preferences
        case complete
    }

    func advance() {
        switch currentStep {
        case .welcome: currentStep = .nameEntry
        case .nameEntry: currentStep = .preferences
        case .preferences: currentStep = .complete
        case .complete: break
        }
    }
}
```

**Common Misconceptions:**
- Misconception: "I need a full coordinator pattern for every app" → Clarify: "NavigationStack with a path handles most cases. Coordinators are for complex multi-step flows with branching logic. Don't add complexity you don't need."
- Misconception: "NavigationLink is bad" → Clarify: "NavigationLink with a value is fine and recommended. The path-based approach adds programmatic control when you need it."

**Verification Questions:**
1. "What's the advantage of using a NavigationPath over inline NavigationLink destinations?"
2. "How would you implement a 'pop to root' feature?"
3. "If you have a checkout flow with 4 steps, how would you structure the navigation?"

---

### Section 6: Error Handling and Loading States

**Core Concept to Teach:**
Every screen that loads data has three states: loading, loaded (success), and error. Representing these explicitly prevents bugs where the UI shows stale data, a spinner that never stops, or no feedback at all.

**How to Explain:**
1. Show the problem with separate boolean flags
2. Introduce an enum-based approach
3. Show how this simplifies view code

**The Problem with Booleans:**

```swift
// Fragile — easy to get into impossible states
var isLoading = false
var errorMessage: String?
var books: [Book] = []
// Can isLoading be true while errorMessage is non-nil? What does that mean?
```

**Enum-Based State:**

```swift
enum LoadingState<T> {
    case idle
    case loading
    case loaded(T)
    case error(String)
}
```

```swift
@Observable
class BookListViewModel {
    private let repository: BookRepository
    private(set) var state: LoadingState<[Book]> = .idle
    var searchText = ""

    init(repository: BookRepository = RemoteBookRepository()) {
        self.repository = repository
    }

    var filteredBooks: [Book] {
        guard case .loaded(let books) = state else { return [] }
        if searchText.isEmpty { return books }
        return books.filter {
            $0.title.localizedCaseInsensitiveContains(searchText)
        }
    }

    func loadBooks() async {
        state = .loading
        do {
            let books = try await repository.fetchBooks()
            state = .loaded(books)
        } catch {
            state = .error(error.localizedDescription)
        }
    }
}
```

```swift
struct BookListView: View {
    @State private var viewModel = BookListViewModel()

    var body: some View {
        NavigationStack {
            Group {
                switch viewModel.state {
                case .idle:
                    Color.clear
                case .loading:
                    ProgressView("Loading books...")
                case .loaded:
                    List(viewModel.filteredBooks) { book in
                        NavigationLink(book.title) {
                            BookDetailView(book: book)
                        }
                    }
                case .error(let message):
                    ContentUnavailableView {
                        Label("Error", systemImage: "exclamationmark.triangle")
                    } description: {
                        Text(message)
                    } actions: {
                        Button("Retry") {
                            Task { await viewModel.loadBooks() }
                        }
                    }
                }
            }
            .searchable(text: $viewModel.searchText)
            .navigationTitle("Books")
            .task {
                await viewModel.loadBooks()
            }
        }
    }
}
```

**Walk Through:**
- "The enum makes impossible states impossible. You can't be loading and have an error at the same time."
- "The view uses `switch` to handle every case — the compiler enforces exhaustiveness."
- "The error state includes a retry button — always give users a way to recover."
- "In React terms: this is like using a discriminated union type for fetch state instead of separate `loading`/`error`/`data` booleans."

**Common Misconceptions:**
- Misconception: "I need a third-party library for loading states" → Clarify: "A simple generic enum is all you need. Libraries exist but add unnecessary dependency for this."
- Misconception: "Every API call needs its own LoadingState" → Clarify: "One LoadingState per ViewModel is typical. If a screen has multiple independent data sources, you might have multiple, but start with one."

**Verification Questions:**
1. "What's wrong with using separate `isLoading`, `errorMessage`, and `data` properties?"
2. "How would you add a 'pull to refresh' that shows the existing data while refreshing?"
3. Multiple choice: "When should the loading state be `.idle`? A) After a successful load B) Before the first load has been triggered C) When there's no data D) While data is being fetched"

---

### Section 7: Project Organization

**Core Concept to Teach:**
As apps grow, code organization matters. The iOS community generally organizes by feature (all files for a feature together) rather than by layer (all views together, all view models together). Swift packages provide module boundaries for larger apps.

**How to Explain:**
1. Compare organization by feature vs by layer
2. Show a realistic project structure
3. Introduce Swift packages for modularization

**Organization by Feature:**

```
BookApp/
├── App/
│   ├── BookApp.swift
│   └── ContentView.swift
├── Books/
│   ├── Book.swift
│   ├── BookListView.swift
│   ├── BookListViewModel.swift
│   ├── BookDetailView.swift
│   ├── BookDetailViewModel.swift
│   └── BookRepository.swift
├── Authors/
│   ├── Author.swift
│   ├── AuthorListView.swift
│   ├── AuthorListViewModel.swift
│   └── AuthorRepository.swift
├── Shared/
│   ├── LoadingState.swift
│   ├── NavigationRouter.swift
│   └── ErrorView.swift
└── Resources/
    └── Assets.xcassets
```

"Compare this to organizing by layer (all views in a Views/ folder, all view models in ViewModels/ folder). Feature-based grouping keeps related code together — when you work on Books, everything you need is in one place."

**Swift Packages for Modularization:**

"For larger apps, Swift packages create hard module boundaries:"

```
BookApp/
├── BookApp/ (main target)
├── Packages/
│   ├── BookFeature/
│   │   ├── Sources/
│   │   │   ├── BookListView.swift
│   │   │   ├── BookListViewModel.swift
│   │   │   └── BookDetailView.swift
│   │   └── Tests/
│   │       └── BookListViewModelTests.swift
│   ├── Networking/
│   │   ├── Sources/
│   │   │   └── APIClient.swift
│   │   └── Tests/
│   │       └── APIClientTests.swift
│   └── SharedUI/
│       └── Sources/
│           ├── LoadingState.swift
│           └── ErrorView.swift
```

"Each package can have its own tests, dependencies, and access control. `internal` means accessible within the package but not outside it — a real module boundary, unlike file folders."

**When to Modularize:**
- "Start with folders. Move to packages when you feel pain — slow build times, unclear boundaries, multiple developers stepping on each other."
- "Don't create packages for a 5-screen app. Do consider them when you have 20+ screens or a team of 3+ developers."

**Common Misconceptions:**
- Misconception: "I should organize by layer (Views/, ViewModels/, Models/)" → Clarify: "This falls apart as apps grow. When you add a feature, you touch every folder. Feature-based organization keeps changes localized."
- Misconception: "Swift packages are only for shared libraries" → Clarify: "Local packages (in your project) are a lightweight way to enforce module boundaries. No need to publish them."

**Verification Questions:**
1. "Why is feature-based organization preferred over layer-based?"
2. "At what point would you consider using Swift packages instead of just folders?"
3. "What does `internal` access control give you in a Swift package that folders don't?"

---

### Section 8: Practice Project

**Project Introduction:**
"Let's put it all together. You're going to take a monolithic SwiftUI app — a simple book browser with search, detail views, and favorites — and refactor it into a well-structured MVVM architecture."

**Starting Code to Provide:**
Give the learner a monolithic version with everything in views — networking, state, filtering, favorites, navigation. About 100-150 lines in a single file.

**Requirements:**
1. Extract Model types (Book, Author)
2. Create a BookRepository protocol with a remote implementation
3. Create ViewModels for the list and detail screens
4. Use LoadingState enum for async state management
5. Wire up dependency injection so the ViewModel accepts a repository
6. Write at least two tests for the ViewModel using a mock repository

**Scaffolding Strategy:**
1. **If they want to try alone**: Give them the monolithic code and requirements. Check in at milestones.
2. **If they want guidance**: Work through it layer by layer — Models first, then Repository, then ViewModel, then update the View, then tests.
3. **If they're unsure**: Start with "Extract the Book struct into its own file" — the smallest step.

**Checkpoints During Project:**
- After Model extraction: "Clean separation? Any computed properties that belong on the model?"
- After Repository: "Does the protocol cover all the data access the app needs?"
- After ViewModel: "Is the view model testable without a network connection? Does it use private(set)?"
- After View update: "Does the view contain any business logic? Any URLSession calls?"
- After tests: "Do the tests verify meaningful behavior? Loading states, error handling, filtering?"

**Code Review Approach:**
1. Check that concerns are properly separated
2. Verify the ViewModel doesn't import SwiftUI
3. Make sure the repository protocol is minimal (only the methods actually needed)
4. Confirm tests use mock repositories, not real network calls
5. Check for proper async handling in the ViewModel

**If They Get Stuck:**
- On extraction: "Start with the data. What are the nouns? Those are your models. What operations do you perform on data? Those go in the repository."
- On ViewModel: "Copy the state and functions from the view into a new class. Mark it @Observable. Remove SwiftUI imports."
- On tests: "Create a MockBookRepository that returns hardcoded data. Create a ViewModel with that mock. Call the method. Check the state."

---

## Wrap-Up and Next Steps

**Review Key Takeaways:**
1. Architecture solves real problems — testability, maintainability, separation of concerns
2. MVVM maps naturally to SwiftUI: @Observable ViewModels hold state, Views render it
3. The repository pattern abstracts data access behind protocols
4. Dependency injection enables testing by letting you swap real implementations for test doubles
5. Enum-based loading states prevent impossible state combinations
6. Organize by feature, not by layer

**Assess Confidence:**
"On a scale of 1-10, how confident do you feel structuring a SwiftUI app now?"
- 1-4: Suggest re-reading the guide, focusing on the MVVM and repository sections. Walk through another example together.
- 5-7: Normal. Suggest building a small app from scratch using these patterns. Real practice solidifies the concepts.
- 8-10: They've got it. Suggest ios-data-persistence to add persistence to their architecture.

**Suggest Next Steps:**
- "iOS Data Persistence adds Core Data and other storage to your repository layer"
- "CloudKit Integration shows how to sync data across devices"
- "If you want to practice, try building a simple app (todo list, weather app, recipe browser) using the full architecture from scratch"

---

## Adaptive Teaching Strategies

### If Learner is Struggling
- Focus on MVVM first — get them comfortable with the three-layer split before adding repositories and DI
- Use a simple example (counter, todo list) instead of the book app
- Write the first refactoring step together, then let them continue
- If they're overwhelmed, skip navigation patterns and project organization — they can revisit those later

### If Learner is Excelling
- Discuss trade-offs: when is this architecture overkill?
- Explore advanced patterns: coordinator pattern, factory methods for dependency graphs
- Challenge them to add offline support (cache-first repository)
- Discuss how this architecture scales to team development

### If Learner Seems Disengaged
- Ask what they're building — use their app as the example
- If the patterns feel academic, jump to the practice project and learn by refactoring
- Focus on testing — if they've felt the pain of untestable code, DI becomes immediately compelling

### Different Learning Styles
- **Visual learners**: Draw dependency diagrams showing which layer depends on which
- **Hands-on learners**: Jump to the practice project early, learn patterns by refactoring
- **Conceptual learners**: Spend more time on the "why" — what problems does each pattern solve?
- **Example-driven learners**: Show multiple before/after comparisons at different scales

---

## Troubleshooting Common Issues

### Technical Problems
- **@Observable not found**: Make sure `import Observation` is present (not `import SwiftUI`)
- **ViewModel not updating the view**: Verify the ViewModel is stored in `@State` in the view, not created as a local variable
- **Tests not finding the ViewModel**: Make sure the ViewModel's access level is `internal` or `public` (default is `internal`, which works if tests are in the same package)
- **NavigationPath type errors**: Types appended to NavigationPath must conform to `Hashable`

### Concept-Specific Confusion
**If confused about where logic belongs:**
- "If it's about how something looks → View"
- "If it's about what data to show or how to respond to user actions → ViewModel"
- "If it's about where data comes from → Repository"
- "If it's about what the data is → Model"

**If confused about @Observable vs @State:**
- "@State is for value types (structs, enums) owned by a view"
- "@Observable is for classes that hold state — used with ViewModels"
- "The view uses @State to own the @Observable ViewModel instance"

**If confused about protocols for DI:**
- "The protocol is a contract. It says 'I need something that can fetch books.' It doesn't say how."
- "The real implementation fetches from the network. The mock returns hardcoded data. Both satisfy the contract."

---

## Teaching Notes

**Key Emphasis Points:**
- Motivation before pattern. Always show the problem first, then the solution.
- Don't over-architect. A simple app with good separation is better than a complex app with perfect abstraction layers.
- Testing is the payoff. The biggest win from all this architecture is testability — make sure they experience writing a test that works because of DI.
- Patterns serve the developer, not the other way around. If a pattern makes code harder to understand, it's being misapplied.

**Pacing Guidance:**
- Section 1-2 (Why + MVVM): Foundation. Take time here. If they get this, everything else follows.
- Section 3-4 (Data Layer + DI): Critical for testability. Make sure they write at least one test.
- Section 5 (Navigation): Can be brief unless they're building a complex app.
- Section 6 (Loading States): Quick concept, powerful impact. Don't over-teach it.
- Section 7 (Organization): Brief. Mention it, give the structure, move on.
- Section 8 (Practice): Allow the most time here. This is where it all clicks.

**Success Indicators:**
You'll know they've got it when they:
- Can explain why a piece of logic belongs in a specific layer
- Write a ViewModel that's testable without SwiftUI
- Create a test using a mock repository without guidance
- Look at a monolithic view and immediately identify what to extract
