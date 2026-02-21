---
title: iOS App Patterns
route_map: /routes/ios-app-patterns/map.md
paired_sherpa: /routes/ios-app-patterns/sherpa.md
prerequisites:
  - SwiftUI fundamentals
  - Swift language proficiency
topics:
  - MVVM
  - Architecture
  - Dependency Injection
  - SwiftUI Patterns
---

# iOS App Patterns

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/ios-app-patterns/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/ios-app-patterns/map.md` for the high-level overview.

## Overview

Building a single-screen demo is straightforward. Building a multi-screen app with shared state, network calls, data persistence, and testability requires architectural decisions. This guide covers the patterns the iOS community has settled on for structuring SwiftUI apps — MVVM, the repository pattern, dependency injection, and practical strategies for navigation, error handling, and project organization.

These aren't theoretical ideals. They're practical solutions to specific problems you'll hit as your app grows beyond a few screens.

## Learning Objectives

By the end of this guide, you will be able to:
- Identify when a SwiftUI view is doing too much and needs separation
- Structure a SwiftUI app using MVVM with @Observable view models
- Build a data layer with the repository pattern and protocol abstractions
- Make network requests with URLSession, async/await, and Codable
- Apply dependency injection for testability
- Manage navigation with NavigationStack and navigation paths
- Represent loading, success, and error states with enums
- Organize a project by feature using folders and Swift packages

## Prerequisites

Before starting, you should have:
- SwiftUI fundamentals — views, @State, @Binding, @Observable, @Environment, navigation (see swiftui-fundamentals route)
- Swift language proficiency — protocols, generics, closures, async/await, error handling (see swift-for-developers route)
- Basic understanding of REST APIs and JSON

## Setup

Create a new Xcode project:

1. File > New > Project
2. Choose **App** under iOS
3. Set Product Name to `BookBrowser`
4. Interface: **SwiftUI**, Language: **Swift**
5. Check **Include Tests**
6. Click Create

We'll build a book browsing app throughout this guide, progressively applying each pattern.

---

## Section 1: Why Architecture Matters

### The Problem

Here's a realistic SwiftUI view that does everything — fetches data, manages state, handles errors, filters, and renders:

```swift
struct BookListView: View {
    @State private var books: [Book] = []
    @State private var isLoading = false
    @State private var errorMessage: String?
    @State private var searchText = ""

    var filteredBooks: [Book] {
        if searchText.isEmpty { return books }
        return books.filter {
            $0.title.localizedCaseInsensitiveContains(searchText)
        }
    }

    var body: some View {
        NavigationStack {
            Group {
                if isLoading {
                    ProgressView()
                } else if let error = errorMessage {
                    Text(error)
                        .foregroundStyle(.red)
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

struct Book: Identifiable, Codable {
    let id: UUID
    let title: String
    let author: String
    let pageCount: Int
}
```

This works. For a demo or prototype, it's fine. But look at what this single view is responsible for:

1. **State management** — tracking books, loading, errors, search text
2. **Networking** — constructing URLs, calling URLSession, handling HTTP responses
3. **Data parsing** — JSON decoding with Codable
4. **Business logic** — filtering books by search text
5. **Error handling** — catching errors and converting to user-facing messages
6. **Rendering** — layout, navigation, conditional display

That's six jobs for one struct. As the app grows — adding favorites, sorting, pagination, offline support — this view becomes a mess.

### Why This Hurts

**Testability**: You can't test `loadBooks()` or `filteredBooks` without rendering the SwiftUI view. There's no way to inject test data or simulate errors.

**Reusability**: If another screen needs to fetch books (maybe a favorites screen), you'd duplicate the networking code.

**Maintainability**: Adding a feature means editing this one large view. Every change risks breaking something else.

### Separation of Concerns

The fix is splitting responsibilities into layers, each with a single job:

| Layer | Responsibility |
|-------|---------------|
| **Model** | Data structures, validation rules |
| **Repository** | Data access (network, database, cache) |
| **ViewModel** | Screen state, business logic, orchestration |
| **View** | Rendering, layout, user interaction |

By the end of this guide, the monolithic view above will be cleanly split across these layers.

### Checkpoint 1

Before moving on, make sure you understand:
- [ ] A view that handles state, networking, parsing, and rendering is doing too much
- [ ] This causes problems with testing, reusability, and maintainability
- [ ] The solution is separating concerns into layers with distinct responsibilities

---

## Section 2: MVVM in SwiftUI

### Model-View-ViewModel

MVVM splits a screen into three parts:

- **Model** — Data structures and business rules. Pure Swift, no UI framework dependency.
- **ViewModel** — Holds the screen's state and logic. Transforms models into what the view needs. Uses `@Observable` so SwiftUI reacts to changes.
- **View** — Renders the current state. Reads from the ViewModel, calls ViewModel methods for user actions. No business logic.

### The Model

Models are the nouns of your app. They're plain Swift types — structs for value types, enums for state:

```swift
struct Book: Identifiable, Codable {
    let id: UUID
    let title: String
    let author: String
    let pageCount: Int
    let coverURL: URL?

    var isLong: Bool { pageCount > 300 }
}
```

Models belong in their own files. They don't import SwiftUI, don't know about screens, and don't change based on how they're displayed.

### The ViewModel

The ViewModel owns the state for a screen and contains the logic that transforms models into what the view needs:

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

**Key details:**
- `import Observation`, not `import SwiftUI` — the ViewModel doesn't depend on the UI framework. This makes it testable without a rendering context.
- `@Observable` makes the class reactive. SwiftUI views that read its properties will re-render when those properties change.
- `private(set)` means only the ViewModel can modify `books`, `isLoading`, and `errorMessage`. The view can read them but not change them directly. This enforces a clear data flow direction.
- `searchText` is read-write because the view needs to bind to it (via `$viewModel.searchText`).

### The View

With the ViewModel handling state and logic, the view becomes pure rendering:

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

The view reads state from the ViewModel and calls ViewModel methods for actions. No networking, no JSON decoding, no filtering logic.

`@State private var viewModel = BookListViewModel()` — the view owns the ViewModel instance. Because `BookListViewModel` is `@Observable`, SwiftUI automatically tracks which properties the view reads and re-renders when they change.

### Where Logic Belongs

Use this decision framework when you're not sure where to put something:

| Question | Layer |
|----------|-------|
| What is the data? | Model |
| Where does the data come from? | Repository (next section) |
| What data should the screen show? | ViewModel |
| How should the data look on screen? | View |

More specifically:
- **Model**: Validation, computed properties from the model's own data (e.g., `isLong`), Codable conformance
- **ViewModel**: Screen state, data loading, filtering/sorting, user action handling, error transformation
- **View**: Layout, styling, navigation presentation, user input bindings

**Not every view needs a ViewModel.** A `BookRow` that takes a `Book` and renders it doesn't need one — it has no independent state or logic. ViewModels are for screens or features with state and business logic.

### Exercise 2.1: Extract a ViewModel

**Task:** Here's a monolithic profile view. Extract the state and logic into a `ProfileViewModel`:

```swift
struct ProfileView: View {
    @State private var user: User?
    @State private var isEditing = false
    @State private var editedName = ""
    @State private var isLoading = false

    var body: some View {
        Group {
            if isLoading {
                ProgressView()
            } else if let user {
                VStack {
                    Text(user.name).font(.title)
                    Text(user.email).foregroundStyle(.secondary)
                    Button("Edit") {
                        editedName = user.name
                        isEditing = true
                    }
                }
            }
        }
        .sheet(isPresented: $isEditing) {
            EditNameSheet(name: $editedName) {
                Task { await saveName() }
            }
        }
        .task { await loadProfile() }
    }

    func loadProfile() async {
        isLoading = true
        let url = URL(string: "https://api.example.com/profile")!
        let (data, _) = try! await URLSession.shared.data(from: url)
        user = try! JSONDecoder().decode(User.self, from: data)
        isLoading = false
    }

    func saveName() async {
        // Save logic...
    }
}
```

<details>
<summary>Hint 1: What moves to the ViewModel?</summary>

All `@State` properties except `isEditing` (which is purely about presentation). The `loadProfile()` and `saveName()` methods. The ViewModel should be `@Observable` and use `private(set)` for properties the view shouldn't modify directly.
</details>

<details>
<summary>Hint 2: What stays in the View?</summary>

The layout, the `.sheet` presentation, the `.task` modifier. The view reads from the ViewModel and calls its methods. `isEditing` can stay in the view since it's a UI concern (sheet presentation state).
</details>

<details>
<summary>Solution</summary>

```swift
import Observation

@Observable
class ProfileViewModel {
    private(set) var user: User?
    private(set) var isLoading = false
    var editedName = ""

    func loadProfile() async {
        isLoading = true
        do {
            let url = URL(string: "https://api.example.com/profile")!
            let (data, _) = try await URLSession.shared.data(from: url)
            user = try JSONDecoder().decode(User.self, from: data)
        } catch {
            // Handle error
        }
        isLoading = false
    }

    func startEditing() {
        guard let user else { return }
        editedName = user.name
    }

    func saveName() async {
        // Save logic using editedName...
    }
}

struct ProfileView: View {
    @State private var viewModel = ProfileViewModel()
    @State private var isEditing = false

    var body: some View {
        Group {
            if viewModel.isLoading {
                ProgressView()
            } else if let user = viewModel.user {
                VStack {
                    Text(user.name).font(.title)
                    Text(user.email).foregroundStyle(.secondary)
                    Button("Edit") {
                        viewModel.startEditing()
                        isEditing = true
                    }
                }
            }
        }
        .sheet(isPresented: $isEditing) {
            EditNameSheet(name: $viewModel.editedName) {
                Task { await viewModel.saveName() }
            }
        }
        .task { await viewModel.loadProfile() }
    }
}
```

The ViewModel handles data and logic. The view handles presentation. `isEditing` stays in the view because it's purely about sheet presentation.
</details>

### Checkpoint 2

Before moving on, make sure you understand:
- [ ] MVVM splits screens into Model (data), ViewModel (state + logic), View (rendering)
- [ ] ViewModels use `@Observable` and `private(set)` for controlled state
- [ ] Views own their ViewModel with `@State private var viewModel = ...`
- [ ] Not every view needs a ViewModel — only those with state and business logic

---

## Section 3: The Data Layer

### The Repository Pattern

The ViewModel in Section 2 still calls URLSession directly. This couples it to networking — you can't test it without a network connection, can't swap in a cache, and can't reuse the networking logic elsewhere.

The repository pattern fixes this by abstracting data access behind a protocol:

```swift
protocol BookRepository {
    func fetchBooks() async throws -> [Book]
    func fetchBook(id: UUID) async throws -> Book
}
```

The protocol defines what operations are available. It says nothing about how they work — that's the implementation's concern.

### Remote Implementation

```swift
struct RemoteBookRepository: BookRepository {
    private let baseURL = URL(string: "https://api.example.com")!
    private let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()

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
        case .requestFailed:
            return "Unable to load books. Please check your connection."
        case .notFound:
            return "Book not found."
        }
    }
}
```

**What's happening here:**
- `RemoteBookRepository` conforms to `BookRepository` — it implements the protocol's methods
- It owns the networking details: base URL, URLSession, JSON decoding, HTTP status checking
- Errors are domain-specific (`BookError`) rather than raw networking errors — the ViewModel gets meaningful error types
- `LocalizedError` conformance provides user-facing error descriptions via `errorDescription`

### Networking with URLSession and Codable

URLSession is Swift's built-in HTTP client. Combined with async/await, it's straightforward:

```swift
// GET request
let (data, response) = try await URLSession.shared.data(from: url)

// POST request with a JSON body
var request = URLRequest(url: url)
request.httpMethod = "POST"
request.setValue("application/json", forHTTPHeaderField: "Content-Type")
request.httpBody = try JSONEncoder().encode(newBook)
let (data, response) = try await URLSession.shared.data(for: request)
```

**Codable** handles JSON ↔ Swift conversion. If your struct's property names match the JSON keys, it works automatically:

```swift
// JSON: {"id": "...", "title": "Swift Basics", "author": "Apple", "page_count": 200}

struct Book: Codable {
    let id: UUID
    let title: String
    let author: String
    let pageCount: Int  // Doesn't match "page_count"

    enum CodingKeys: String, CodingKey {
        case id, title, author
        case pageCount = "page_count"  // Map the JSON key
    }
}
```

`CodingKeys` handles mismatched names. You can also configure `JSONDecoder().keyDecodingStrategy = .convertFromSnakeCase` to handle all snake_case → camelCase conversion automatically.

### Updating the ViewModel

With the repository, the ViewModel no longer knows about URLSession:

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

The ViewModel asks the repository for books and gets back `[Book]` or an error. It doesn't know (or care) whether the data comes from a server, a database, or a hardcoded list. The default parameter `RemoteBookRepository()` means the view still creates it simply, while tests can inject something different.

### Checkpoint 3

Before moving on, make sure you understand:
- [ ] The repository pattern abstracts data access behind a protocol
- [ ] The ViewModel depends on the protocol, not the concrete implementation
- [ ] URLSession + Codable handle HTTP requests and JSON parsing
- [ ] Domain-specific error types provide meaningful error messages

---

## Section 4: Dependency Injection

### What and Why

Dependency injection means passing dependencies in from outside rather than creating them inside. When the ViewModel in Section 3 accepts `repository: BookRepository` in its init, that's dependency injection. The ViewModel doesn't create its own data source — it receives one.

This matters for one reason above all: **testing**. If the ViewModel creates `RemoteBookRepository()` internally, every test hits the network. With injection, you swap in a test double that returns controlled data instantly.

### Initializer Injection

The simplest and most common form. We already did it:

```swift
// Production: uses the default
let viewModel = BookListViewModel()

// Testing: inject a test double
let viewModel = BookListViewModel(repository: MockBookRepository())
```

Default parameter values mean callers don't need to know about the dependency unless they want to override it.

### @Environment for Shared Dependencies

For dependencies used across many screens — an auth service, analytics, feature flags — passing through initializers gets tedious. `@Environment` provides app-wide injection:

```swift
// 1. Define an environment key
struct BookRepositoryKey: EnvironmentKey {
    static let defaultValue: BookRepository = RemoteBookRepository()
}

extension EnvironmentValues {
    var bookRepository: BookRepository {
        get { self[BookRepositoryKey.self] }
        set { self[BookRepositoryKey.self] = newValue }
    }
}

// 2. Inject at the app root
@main
struct BookApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.bookRepository, RemoteBookRepository())
        }
    }
}

// 3. Use in any descendant view
struct BookListView: View {
    @Environment(\.bookRepository) private var repository

    var body: some View {
        // Pass to ViewModel or use directly...
    }
}
```

This is conceptually identical to React Context — provide at the top, consume anywhere below without prop drilling.

**When to use which:**
- **Initializer injection**: For direct dependencies of a specific class (ViewModel's repository)
- **@Environment**: For dependencies shared broadly across the view hierarchy (auth state, analytics)

### Testing with Dependency Injection

Here's the payoff — testable ViewModels:

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
```

```swift
import Testing

@Test func loadBooksPopulatesList() async {
    let testBooks = [
        Book(id: UUID(), title: "Swift Basics", author: "Apple",
             pageCount: 200, coverURL: nil),
        Book(id: UUID(), title: "SwiftUI Guide", author: "Apple",
             pageCount: 350, coverURL: nil)
    ]
    let viewModel = BookListViewModel(
        repository: MockBookRepository(booksToReturn: testBooks)
    )

    await viewModel.loadBooks()

    #expect(viewModel.books.count == 2)
    #expect(viewModel.books.first?.title == "Swift Basics")
    #expect(viewModel.isLoading == false)
    #expect(viewModel.errorMessage == nil)
}

@Test func loadBooksHandlesError() async {
    let viewModel = BookListViewModel(
        repository: MockBookRepository(
            errorToThrow: BookError.requestFailed)
    )

    await viewModel.loadBooks()

    #expect(viewModel.books.isEmpty)
    #expect(viewModel.errorMessage != nil)
    #expect(viewModel.isLoading == false)
}

@Test func filteringMatchesByTitle() async {
    let testBooks = [
        Book(id: UUID(), title: "Swift Basics", author: "Apple",
             pageCount: 200, coverURL: nil),
        Book(id: UUID(), title: "Python Guide", author: "PSF",
             pageCount: 300, coverURL: nil)
    ]
    let viewModel = BookListViewModel(
        repository: MockBookRepository(booksToReturn: testBooks)
    )

    await viewModel.loadBooks()
    viewModel.searchText = "swift"

    #expect(viewModel.filteredBooks.count == 1)
    #expect(viewModel.filteredBooks.first?.title == "Swift Basics")
}
```

**What these tests verify:**
1. Successful load populates the books list and clears loading state
2. Failed load sets the error message and keeps the list empty
3. Search filtering works correctly

All without a network connection, running instantly. This is why architecture matters.

### Exercise 4.1: Write a Test

**Task:** Write a test that verifies the ViewModel starts in a loading state when `loadBooks()` is called, then transitions to a non-loading state when done.

<details>
<summary>Hint</summary>

You'll need to check `isLoading` during the async operation. One approach: check the state before awaiting (though this is tricky with async). A simpler approach: verify the final state is `isLoading == false`, and trust that the implementation sets it to `true` first (the other tests verify it works end-to-end).

For a more thorough test, you could create a repository that introduces a delay, but for this exercise, verifying the end state is sufficient.
</details>

<details>
<summary>Solution</summary>

```swift
@Test func loadBooksEndsInNonLoadingState() async {
    let viewModel = BookListViewModel(
        repository: MockBookRepository(booksToReturn: [
            Book(id: UUID(), title: "Test", author: "Author",
                 pageCount: 100, coverURL: nil)
        ])
    )

    // Before loading
    #expect(viewModel.isLoading == false)
    #expect(viewModel.books.isEmpty)

    await viewModel.loadBooks()

    // After loading
    #expect(viewModel.isLoading == false)
    #expect(viewModel.books.count == 1)
}
```
</details>

### Checkpoint 4

Before moving on, make sure you understand:
- [ ] Dependency injection means passing dependencies in, not creating them internally
- [ ] Initializer injection with default parameters is the simplest approach
- [ ] @Environment provides app-wide dependency injection (like React Context)
- [ ] Mock repositories enable fast, deterministic tests
- [ ] Swift Testing uses `@Test` and `#expect`

---

## Section 5: Navigation Patterns

### Inline vs Programmatic Navigation

For simple apps, inline `NavigationLink` with a destination view works fine:

```swift
NavigationLink("Book Title") {
    BookDetailView(book: book)
}
```

For larger apps, you want programmatic control — deep linking, "pop to root," conditional navigation. `NavigationStack` with a navigation path provides this.

### NavigationStack with Path

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

**How it works:**
- `NavigationPath` is a type-erased stack of destinations. Append values to push screens.
- `.navigationDestination(for:)` registers a mapping: "when a `Book` is pushed, show `BookDetailView`." The compiler ensures type safety.
- The router centralizes navigation logic. Any part of the app can call `router.showBookDetail(book)`.
- `popToRoot()` clears the entire stack — useful for "done" buttons in deep flows.

For types to work with NavigationPath, they must conform to `Hashable`:

```swift
struct Book: Identifiable, Codable, Hashable {
    let id: UUID
    let title: String
    let author: String
    let pageCount: Int
    let coverURL: URL?
}
```

### When You Need Programmatic Navigation

- **Deep linking**: An external URL needs to open a specific screen
- **Conditional flows**: After login, navigate to home or onboarding based on user state
- **Pop to root**: A "Done" button that returns to the top of the stack
- **Analytics**: Tracking which screens users visit

For apps with only forward navigation (tap item → see detail), inline `NavigationLink` is simpler and perfectly fine.

### Checkpoint 5

Before moving on, make sure you understand:
- [ ] `NavigationPath` provides a programmatic navigation stack
- [ ] `.navigationDestination(for:)` maps types to destination views
- [ ] Types appended to `NavigationPath` must conform to `Hashable`
- [ ] Inline `NavigationLink` is fine for simple forward navigation

---

## Section 6: Error Handling and Loading States

### The Problem with Booleans

The ViewModel currently uses separate properties for state:

```swift
var isLoading = false
var errorMessage: String?
var books: [Book] = []
```

This allows impossible combinations. Can `isLoading` be `true` while `errorMessage` is set? What does an empty `books` array mean — no results, or hasn't loaded yet?

### Enum-Based State

A generic enum makes impossible states impossible:

```swift
enum LoadingState<T> {
    case idle
    case loading
    case loaded(T)
    case error(String)
}
```

Four states, mutually exclusive. The compiler enforces that you handle every case.

### Applying to the ViewModel

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

The ViewModel has a single `state` property instead of three separate ones. `filteredBooks` safely extracts the books from `.loaded` or returns an empty array.

### Applying to the View

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
                        Label("Error",
                            systemImage: "exclamationmark.triangle")
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

The `switch` handles every case. The compiler warns if you add a new case to `LoadingState` and forget to handle it in the view. The error state includes a retry button — always give users a way to recover.

### Exercise 6.1: Add Pull-to-Refresh

**Task:** Add pull-to-refresh to the book list that shows the existing data while refreshing (not the loading spinner). The `.refreshable` modifier provides the pull-to-refresh gesture.

<details>
<summary>Hint 1</summary>

The `.refreshable` modifier takes an async closure. You need a way to reload without showing the loading state — maybe a separate method on the ViewModel, or a parameter on `loadBooks`.
</details>

<details>
<summary>Hint 2</summary>

One approach: Add a `refresh()` method that doesn't set `state = .loading`. It fetches new data and updates the `.loaded` state directly, or sets `.error` if it fails.
</details>

<details>
<summary>Solution</summary>

```swift
// In BookListViewModel, add:
func refresh() async {
    do {
        let books = try await repository.fetchBooks()
        state = .loaded(books)
    } catch {
        // On refresh failure, keep existing data visible
        // Optionally show a brief error toast instead
    }
}
```

```swift
// In the view, add .refreshable to the List:
case .loaded:
    List(viewModel.filteredBooks) { book in
        NavigationLink(book.title) {
            BookDetailView(book: book)
        }
    }
    .refreshable {
        await viewModel.refresh()
    }
```

The refresh doesn't replace the list with a spinner — it silently updates the data. SwiftUI handles the pull-to-refresh animation automatically.
</details>

### Checkpoint 6

Before moving on, make sure you understand:
- [ ] Separate booleans for loading/error/data allow impossible state combinations
- [ ] A generic `LoadingState<T>` enum enforces mutually exclusive states
- [ ] `switch` on the enum guarantees you handle every case
- [ ] Error states should include a retry mechanism

---

## Section 7: Project Organization

### By Feature, Not by Layer

As your project grows, organize files by feature (all files for a feature together) rather than by layer (all views together, all view models together):

```
BookBrowser/
├── App/
│   ├── BookBrowserApp.swift
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
│   └── NavigationRouter.swift
└── Resources/
    └── Assets.xcassets
```

**Why feature-based?** When you work on the Books feature, everything is in one folder. When you add a new feature, you add one folder. Layer-based organization (`Views/`, `ViewModels/`, `Models/`) means every feature change touches every folder.

### Swift Packages for Modularization

For larger apps, Swift packages create hard boundaries between modules:

```
BookBrowser/
├── BookBrowser/ (main app target)
├── Packages/
│   ├── BookFeature/
│   │   ├── Sources/
│   │   │   ├── BookListView.swift
│   │   │   ├── BookListViewModel.swift
│   │   │   └── BookRepository.swift
│   │   └── Tests/
│   │       └── BookListViewModelTests.swift
│   ├── Networking/
│   │   └── Sources/
│   │       └── APIClient.swift
│   └── SharedUI/
│       └── Sources/
│           └── LoadingState.swift
```

Each package enforces access control: `internal` members are visible within the package but not outside. This prevents unintended coupling between features — a real module boundary that file folders don't provide.

**When to switch from folders to packages:**
- Multiple developers working on separate features simultaneously
- Build times becoming painful (packages build independently)
- You want to enforce that Feature A can't directly access Feature B's internals

For a solo developer or a small app, folders are sufficient.

### Checkpoint 7

Before moving on, make sure you understand:
- [ ] Feature-based organization keeps related files together
- [ ] Layer-based organization (Views/, ViewModels/) becomes painful as apps grow
- [ ] Swift packages provide real module boundaries with access control
- [ ] Start with folders; move to packages when you need the benefits

---

## Practice Project

### Project Description

Refactor a monolithic book browsing app into a well-structured MVVM architecture. You'll take the single-file starting code and split it into proper layers with testable components.

### Starting Code

Create a new file and paste this — it's the monolithic version we'll refactor:

```swift
import SwiftUI

struct Book: Identifiable, Codable, Hashable {
    let id: UUID
    let title: String
    let author: String
    let pageCount: Int
}

struct ContentView: View {
    @State private var books: [Book] = []
    @State private var isLoading = false
    @State private var errorMessage: String?
    @State private var searchText = ""
    @State private var favorites: Set<UUID> = []

    var filteredBooks: [Book] {
        let base = searchText.isEmpty ? books : books.filter {
            $0.title.localizedCaseInsensitiveContains(searchText)
        }
        return base
    }

    var body: some View {
        NavigationStack {
            Group {
                if isLoading {
                    ProgressView()
                } else if let error = errorMessage {
                    VStack {
                        Text(error)
                        Button("Retry") {
                            Task { await loadBooks() }
                        }
                    }
                } else {
                    List(filteredBooks) { book in
                        NavigationLink(value: book) {
                            HStack {
                                VStack(alignment: .leading) {
                                    Text(book.title).font(.headline)
                                    Text(book.author).foregroundStyle(.secondary)
                                }
                                Spacer()
                                if favorites.contains(book.id) {
                                    Image(systemName: "star.fill")
                                        .foregroundStyle(.yellow)
                                }
                            }
                        }
                        .swipeActions {
                            Button {
                                toggleFavorite(book)
                            } label: {
                                Image(systemName: favorites.contains(book.id)
                                    ? "star.slash" : "star.fill")
                            }
                            .tint(.yellow)
                        }
                    }
                }
            }
            .searchable(text: $searchText)
            .navigationTitle("Books")
            .navigationDestination(for: Book.self) { book in
                BookDetailView(book: book,
                    isFavorite: favorites.contains(book.id)) {
                    toggleFavorite(book)
                }
            }
            .task {
                await loadBooks()
            }
        }
    }

    func loadBooks() async {
        isLoading = true
        errorMessage = nil
        // Simulated network call
        do {
            try await Task.sleep(for: .seconds(1))
            books = [
                Book(id: UUID(), title: "The Swift Programming Language",
                     author: "Apple", pageCount: 500),
                Book(id: UUID(), title: "SwiftUI Essentials",
                     author: "Apple", pageCount: 300),
                Book(id: UUID(), title: "iOS Development with Swift",
                     author: "Craig Clayton", pageCount: 450),
                Book(id: UUID(), title: "Combine: Asynchronous Programming",
                     author: "Raywenderlich", pageCount: 350)
            ]
        } catch {
            errorMessage = "Failed to load books"
        }
        isLoading = false
    }

    func toggleFavorite(_ book: Book) {
        if favorites.contains(book.id) {
            favorites.remove(book.id)
        } else {
            favorites.insert(book.id)
        }
    }
}

struct BookDetailView: View {
    let book: Book
    let isFavorite: Bool
    let onToggleFavorite: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            Text(book.title).font(.title)
            Text("by \(book.author)").foregroundStyle(.secondary)
            Text("\(book.pageCount) pages")
            Button {
                onToggleFavorite()
            } label: {
                Label(isFavorite ? "Remove Favorite" : "Add Favorite",
                      systemImage: isFavorite ? "star.slash" : "star.fill")
            }
            .buttonStyle(.borderedProminent)
        }
        .navigationTitle(book.title)
    }
}
```

### Requirements

Refactor this into:

1. **Model** — `Book` struct in its own file
2. **Repository** — `BookRepository` protocol with a mock implementation (the simulated data)
3. **ViewModel** — `BookListViewModel` with state, filtering, and favorites logic
4. **LoadingState** — Enum-based loading state instead of separate booleans
5. **Tests** — At least 3 tests: loading, filtering, and favorites toggling

### Step-by-Step Guide

**Step 1: Extract the Model**

Move `Book` into `Book.swift`. No changes needed — it's already a clean data struct.

**Step 2: Create the Repository**

Create `BookRepository.swift` with the protocol and a mock implementation:

```swift
protocol BookRepository {
    func fetchBooks() async throws -> [Book]
}

struct MockBookRepository: BookRepository {
    func fetchBooks() async throws -> [Book] {
        // Simulate network delay
        try await Task.sleep(for: .seconds(1))
        return [
            Book(id: UUID(), title: "The Swift Programming Language",
                 author: "Apple", pageCount: 500),
            // ... more books
        ]
    }
}
```

**Step 3: Create LoadingState**

Create `LoadingState.swift`:

```swift
enum LoadingState<T> {
    case idle
    case loading
    case loaded(T)
    case error(String)
}
```

**Step 4: Create the ViewModel**

Create `BookListViewModel.swift` that uses the repository and LoadingState. Include favorites logic.

**Step 5: Update the View**

Simplify `ContentView` to read from the ViewModel and call its methods.

**Step 6: Write Tests**

In your test target, create `BookListViewModelTests.swift`. Use a test-specific mock repository that returns instant, controlled data.

### Hints and Tips

<details>
<summary>If you're not sure about favorites in the ViewModel</summary>

The ViewModel owns the `favorites: Set<UUID>` and a `toggleFavorite(_:)` method. It also exposes `isFavorite(_:) -> Bool` for the view to check.
</details>

<details>
<summary>If tests are failing with concurrency issues</summary>

Make sure your test functions are `async`. Use `await` when calling ViewModel methods that are async. The mock repository for tests should return immediately (no `Task.sleep`).
</details>

### Example Solution

<details>
<summary>Click to see one possible solution</summary>

**Book.swift:**
```swift
struct Book: Identifiable, Codable, Hashable {
    let id: UUID
    let title: String
    let author: String
    let pageCount: Int
}
```

**LoadingState.swift:**
```swift
enum LoadingState<T> {
    case idle
    case loading
    case loaded(T)
    case error(String)
}
```

**BookRepository.swift:**
```swift
protocol BookRepository {
    func fetchBooks() async throws -> [Book]
}

struct SimulatedBookRepository: BookRepository {
    func fetchBooks() async throws -> [Book] {
        try await Task.sleep(for: .seconds(1))
        return [
            Book(id: UUID(), title: "The Swift Programming Language",
                 author: "Apple", pageCount: 500),
            Book(id: UUID(), title: "SwiftUI Essentials",
                 author: "Apple", pageCount: 300),
            Book(id: UUID(), title: "iOS Development with Swift",
                 author: "Craig Clayton", pageCount: 450),
            Book(id: UUID(), title: "Combine: Asynchronous Programming",
                 author: "Raywenderlich", pageCount: 350)
        ]
    }
}
```

**BookListViewModel.swift:**
```swift
import Observation

@Observable
class BookListViewModel {
    private let repository: BookRepository
    private(set) var state: LoadingState<[Book]> = .idle
    private(set) var favorites: Set<UUID> = []
    var searchText = ""

    init(repository: BookRepository = SimulatedBookRepository()) {
        self.repository = repository
    }

    var filteredBooks: [Book] {
        guard case .loaded(let books) = state else { return [] }
        if searchText.isEmpty { return books }
        return books.filter {
            $0.title.localizedCaseInsensitiveContains(searchText)
        }
    }

    func isFavorite(_ book: Book) -> Bool {
        favorites.contains(book.id)
    }

    func toggleFavorite(_ book: Book) {
        if favorites.contains(book.id) {
            favorites.remove(book.id)
        } else {
            favorites.insert(book.id)
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

**ContentView.swift:**
```swift
import SwiftUI

struct ContentView: View {
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
                    bookList
                case .error(let message):
                    ContentUnavailableView {
                        Label("Error",
                            systemImage: "exclamationmark.triangle")
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
            .navigationDestination(for: Book.self) { book in
                BookDetailView(
                    book: book,
                    isFavorite: viewModel.isFavorite(book)
                ) {
                    viewModel.toggleFavorite(book)
                }
            }
            .task {
                await viewModel.loadBooks()
            }
        }
    }

    private var bookList: some View {
        List(viewModel.filteredBooks) { book in
            NavigationLink(value: book) {
                HStack {
                    VStack(alignment: .leading) {
                        Text(book.title).font(.headline)
                        Text(book.author).foregroundStyle(.secondary)
                    }
                    Spacer()
                    if viewModel.isFavorite(book) {
                        Image(systemName: "star.fill")
                            .foregroundStyle(.yellow)
                    }
                }
            }
            .swipeActions {
                Button {
                    viewModel.toggleFavorite(book)
                } label: {
                    Image(systemName: viewModel.isFavorite(book)
                        ? "star.slash" : "star.fill")
                }
                .tint(.yellow)
            }
        }
    }
}
```

**BookListViewModelTests.swift:**
```swift
import Testing

struct TestBookRepository: BookRepository {
    var books: [Book] = []
    var error: Error?

    func fetchBooks() async throws -> [Book] {
        if let error { throw error }
        return books
    }
}

@Test func loadBooksSuccess() async {
    let books = [
        Book(id: UUID(), title: "Test Book", author: "Author",
             pageCount: 200)
    ]
    let viewModel = BookListViewModel(
        repository: TestBookRepository(books: books))

    await viewModel.loadBooks()

    guard case .loaded(let loaded) = viewModel.state else {
        #expect(Bool(false), "Expected loaded state")
        return
    }
    #expect(loaded.count == 1)
    #expect(loaded.first?.title == "Test Book")
}

@Test func loadBooksError() async {
    let viewModel = BookListViewModel(
        repository: TestBookRepository(
            error: BookError.requestFailed))

    await viewModel.loadBooks()

    guard case .error = viewModel.state else {
        #expect(Bool(false), "Expected error state")
        return
    }
}

@Test func filterBySearchText() async {
    let books = [
        Book(id: UUID(), title: "Swift Guide", author: "Apple",
             pageCount: 200),
        Book(id: UUID(), title: "Python Guide", author: "PSF",
             pageCount: 300)
    ]
    let viewModel = BookListViewModel(
        repository: TestBookRepository(books: books))

    await viewModel.loadBooks()
    viewModel.searchText = "swift"

    #expect(viewModel.filteredBooks.count == 1)
    #expect(viewModel.filteredBooks.first?.title == "Swift Guide")
}

@Test func toggleFavorite() async {
    let book = Book(id: UUID(), title: "Test", author: "Author",
                    pageCount: 100)
    let viewModel = BookListViewModel(
        repository: TestBookRepository(books: [book]))

    await viewModel.loadBooks()

    #expect(viewModel.isFavorite(book) == false)
    viewModel.toggleFavorite(book)
    #expect(viewModel.isFavorite(book) == true)
    viewModel.toggleFavorite(book)
    #expect(viewModel.isFavorite(book) == false)
}
```
</details>

---

## Summary

### Key Takeaways

- **Separation of concerns** prevents views from becoming bloated and untestable
- **MVVM** maps naturally to SwiftUI: @Observable ViewModels hold state, Views render it
- **Repository pattern** abstracts data access behind protocols, enabling testability and flexibility
- **Dependency injection** lets you swap implementations — real services in production, mocks in tests
- **Enum-based loading states** prevent impossible state combinations
- **Feature-based organization** keeps related code together as your app grows

### Skills You've Gained

You can now:
- Identify when a view needs architectural separation
- Structure an app with Model, ViewModel, Repository, and View layers
- Write testable ViewModels using dependency injection
- Handle async states cleanly with enums
- Organize projects by feature for maintainability

### Self-Assessment

Take a moment to reflect:
- Can you look at a monolithic view and identify what should be extracted?
- Could you write a test for a ViewModel using a mock repository?
- Do you understand where each piece of logic belongs (Model vs ViewModel vs Repository vs View)?

---

## Next Steps

### Continue Learning

**Build on this topic:**
- Practice by building a small app from scratch using these patterns (weather app, todo list, recipe browser)

**Explore related routes:**
- [iOS Data Persistence](/routes/ios-data-persistence/map.md) — Add local storage to your repository layer
- [CloudKit Integration](/routes/cloudkit-integration/map.md) — Sync data across devices via iCloud
- [App Store Publishing](/routes/app-store-publishing/map.md) — Ship your well-structured app

### Additional Resources

**Documentation:**
- Apple's "Adopting Observation in SwiftUI" article
- Swift Testing documentation

**Practice Ideas:**
- Build a weather app: fetch from OpenWeatherMap API, display current conditions and forecast
- Build a recipe browser: list/detail navigation, search, favorites persistence
- Add offline support to the book browser: cache-first repository that falls back to network
