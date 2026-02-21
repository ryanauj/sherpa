---
title: iOS Data Persistence
route_map: /routes/ios-data-persistence/map.md
paired_sherpa: /routes/ios-data-persistence/sherpa.md
prerequisites:
  - Swift language proficiency
  - SwiftUI fundamentals
topics:
  - SwiftData
  - Core Data
  - UserDefaults
  - Keychain
  - Local Storage
---

# iOS Data Persistence

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/ios-data-persistence/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/ios-data-persistence/map.md` for the high-level overview.

## Overview

Every app needs to remember things between launches — user preferences, authentication tokens, the user's actual data. iOS provides several persistence options, from simple key-value storage to full database frameworks. Picking the right one matters: you wouldn't use a database to store a theme preference, and you wouldn't use a plist to store thousands of searchable records.

This guide covers the full spectrum of iOS persistence, with a focus on SwiftData (Apple's modern framework for structured data) and the simpler alternatives for when a database is overkill.

## Learning Objectives

By the end of this guide, you will be able to:
- Choose the right persistence mechanism for different types of data
- Store preferences with UserDefaults and @AppStorage
- Securely store credentials with Keychain
- Model and persist structured data with SwiftData
- Query, filter, and sort persisted data with #Predicate and SortDescriptor
- Integrate persistence with SwiftUI using @Query and ModelContainer
- Recognize Core Data patterns in existing code

## Prerequisites

Before starting, you should have:
- Swift language proficiency — classes, protocols, closures, async/await (see swift-for-developers route)
- SwiftUI fundamentals — views, state management, @Environment (see swiftui-fundamentals route)
- Helpful: MVVM patterns from the ios-app-patterns route (for structuring data access)

## Setup

Create a new Xcode project:

1. File > New > Project
2. Choose **App** under iOS
3. Set Product Name to `NoteTaker`
4. Interface: **SwiftUI**, Language: **Swift**, Storage: **None** (we'll add SwiftData manually)
5. Click Create

We'll build a note-taking app throughout this guide, adding persistence features section by section.

---

## Section 1: Choosing a Persistence Strategy

Before writing any code, understand which tool fits which job. iOS offers several persistence options, each designed for different data types and access patterns.

### The Decision Framework

Ask these questions about each piece of data your app needs to persist:

1. **Is it sensitive?** (passwords, tokens, credentials) → **Keychain**
2. **Is it a simple preference or setting?** (theme, sort order, feature toggles) → **UserDefaults / @AppStorage**
3. **Is it a large file?** (images, PDFs, videos) → **File system**
4. **Is it structured data you need to query?** (notes, tasks, contacts) → **SwiftData**
5. **Is it temporary cache data?** → **File system (cache directory)**

### Quick Reference

| Data Type | Best Option | Why |
|-----------|------------|-----|
| Theme preference | UserDefaults / @AppStorage | Simple key-value, reactive in SwiftUI |
| Auth tokens | Keychain | Encrypted, persists across reinstalls |
| User's notes, tasks, records | SwiftData | Queryable, relationships, SwiftUI integration |
| Downloaded images | File system | Binary data, too large for database |
| API response cache | File system + URLCache | Temporary, OS can purge |

### A Real Example

Building a note-taking app? Here's where each piece of data goes:

- **Sort order preference** (newest first, alphabetical) → UserDefaults
- **Login token** → Keychain
- **Notes with titles, content, tags** → SwiftData
- **Attached images** → File system (store the file path in SwiftData)

### What NOT to Do

- **Don't store passwords in UserDefaults.** UserDefaults is a plaintext plist file. Anyone with filesystem access can read it.
- **Don't store large binary data in SwiftData.** Images, PDFs, and videos belong on the file system. Store the file path in SwiftData.
- **Don't use SwiftData for simple settings.** A full database for a boolean preference is overkill.

### Checkpoint 1

Before moving on, make sure you understand:
- [ ] Different data types need different persistence solutions
- [ ] Sensitive data goes in Keychain, preferences in UserDefaults, structured data in SwiftData
- [ ] Large binary files go on the file system, not in a database

---

## Section 2: UserDefaults and @AppStorage

### UserDefaults Basics

UserDefaults is iOS's simplest persistence — a key-value store backed by a plist file. It's for small amounts of non-sensitive data: preferences, settings, flags.

```swift
// Save values
UserDefaults.standard.set("dark", forKey: "theme")
UserDefaults.standard.set(true, forKey: "showNotifications")
UserDefaults.standard.set(14, forKey: "fontSize")

// Read values
let theme = UserDefaults.standard.string(forKey: "theme") ?? "light"
let showNotifications = UserDefaults.standard.bool(forKey: "showNotifications")
let fontSize = UserDefaults.standard.integer(forKey: "fontSize")

// Remove a value
UserDefaults.standard.removeObject(forKey: "theme")
```

Supported types: `Bool`, `Int`, `Double`, `String`, `URL`, `Data`, and arrays/dictionaries of these types.

### @AppStorage in SwiftUI

`@AppStorage` is a property wrapper that reads and writes UserDefaults values reactively in SwiftUI. When the value changes, the view re-renders — just like `@State`.

```swift
struct SettingsView: View {
    @AppStorage("theme") private var theme = "light"
    @AppStorage("fontSize") private var fontSize = 14
    @AppStorage("showCompletedTasks") private var showCompleted = true

    var body: some View {
        Form {
            Section("Appearance") {
                Picker("Theme", selection: $theme) {
                    Text("Light").tag("light")
                    Text("Dark").tag("dark")
                    Text("System").tag("system")
                }

                Stepper("Font Size: \(fontSize)", value: $fontSize, in: 10...24)
            }

            Section("Behavior") {
                Toggle("Show Completed Tasks", isOn: $showCompleted)
            }
        }
        .navigationTitle("Settings")
    }
}
```

The string parameter (`"theme"`, `"fontSize"`) is the UserDefaults key. The value you assign (`"light"`, `14`) is the default when no saved value exists.

**Reading @AppStorage from other views:**

```swift
struct NoteListView: View {
    @AppStorage("showCompletedTasks") private var showCompleted = true

    // The same key reads the same value — changes in SettingsView
    // are reflected here immediately
}
```

Any view using the same `@AppStorage` key stays in sync automatically.

### Limitations

- **Not encrypted.** Don't store sensitive data.
- **Loads entirely into memory.** Don't store large amounts of data.
- **String keys.** Easy to typo. Consider constants:

```swift
enum StorageKeys {
    static let theme = "theme"
    static let fontSize = "fontSize"
    static let showCompleted = "showCompletedTasks"
}

@AppStorage(StorageKeys.theme) private var theme = "light"
```

### Exercise 2.1: Build a Settings Screen

**Task:** Create a settings screen for the NoteTaker app with these preferences:
- Sort order: by date or by title (use a Picker)
- Default note pinned state: on or off (use a Toggle)
- Notes per page: a number between 10 and 50 (use a Stepper)

Use `@AppStorage` for all three.

<details>
<summary>Hint</summary>

For the sort order Picker, use string tags like `"date"` and `"title"`. @AppStorage works with String values out of the box.
</details>

<details>
<summary>Solution</summary>

```swift
struct SettingsView: View {
    @AppStorage("sortOrder") private var sortOrder = "date"
    @AppStorage("defaultPinned") private var defaultPinned = false
    @AppStorage("notesPerPage") private var notesPerPage = 20

    var body: some View {
        Form {
            Section("Display") {
                Picker("Sort By", selection: $sortOrder) {
                    Text("Date").tag("date")
                    Text("Title").tag("title")
                }

                Stepper("Notes per page: \(notesPerPage)",
                        value: $notesPerPage, in: 10...50, step: 5)
            }

            Section("Defaults") {
                Toggle("Pin new notes by default", isOn: $defaultPinned)
            }
        }
        .navigationTitle("Settings")
    }
}
```
</details>

### Checkpoint 2

Before moving on, make sure you understand:
- [ ] UserDefaults stores simple key-value data in a plist file
- [ ] @AppStorage makes UserDefaults values reactive in SwiftUI
- [ ] Multiple views using the same @AppStorage key stay in sync
- [ ] UserDefaults is not encrypted — never store secrets in it

---

## Section 3: Keychain Basics

### Why Keychain?

Keychain is iOS's secure storage, backed by hardware encryption. Use it for:
- Authentication tokens (OAuth, JWT)
- Passwords and credentials
- API keys that shouldn't be in plaintext
- Any data that would be a security risk if exposed

Key difference from UserDefaults: Keychain data is encrypted and persists even when the user deletes and reinstalls the app.

### A Simple Keychain Helper

The raw Keychain API (Security framework) is verbose and uses Core Foundation types. Here's a practical wrapper:

```swift
import Security

enum KeychainHelper {
    static func save(_ data: Data, for key: String) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key
        ]

        // Remove existing item (update = delete + add)
        SecItemDelete(query as CFDictionary)

        var addQuery = query
        addQuery[kSecValueData as String] = data

        let status = SecItemAdd(addQuery as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw KeychainError.saveFailed(status)
        }
    }

    static func read(for key: String) throws -> Data? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        var result: AnyObject?
        let status = SecCopyMatching(query as CFDictionary, &result)

        if status == errSecItemNotFound { return nil }
        guard status == errSecSuccess else {
            throw KeychainError.readFailed(status)
        }
        return result as? Data
    }

    static func delete(for key: String) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key
        ]
        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.deleteFailed(status)
        }
    }
}

enum KeychainError: LocalizedError {
    case saveFailed(OSStatus)
    case readFailed(OSStatus)
    case deleteFailed(OSStatus)

    var errorDescription: String? {
        switch self {
        case .saveFailed(let s): return "Keychain save failed: \(s)"
        case .readFailed(let s): return "Keychain read failed: \(s)"
        case .deleteFailed(let s): return "Keychain delete failed: \(s)"
        }
    }
}
```

### Using the Helper

```swift
// Save a token after login
func saveAuthToken(_ token: String) throws {
    guard let data = token.data(using: .utf8) else { return }
    try KeychainHelper.save(data, for: "authToken")
}

// Read the token at app launch
func getAuthToken() throws -> String? {
    guard let data = try KeychainHelper.read(for: "authToken") else {
        return nil
    }
    return String(data: data, encoding: .utf8)
}

// Delete on logout
func clearAuthToken() throws {
    try KeychainHelper.delete(for: "authToken")
}
```

### When to Use Keychain vs UserDefaults

| Question | UserDefaults | Keychain |
|----------|-------------|---------|
| Is the data sensitive? | No | Yes |
| Should it survive app reinstall? | No | Yes |
| Is encryption required? | No | Yes |
| Is it a simple preference? | Yes | No |
| Is it a credential or token? | No | Yes |

### Checkpoint 3

Before moving on, make sure you understand:
- [ ] Keychain provides hardware-encrypted storage for sensitive data
- [ ] Keychain data persists across app reinstalls (UserDefaults does not)
- [ ] The raw API is verbose — a helper wrapper makes it practical
- [ ] Auth tokens, passwords, and API keys belong in Keychain, not UserDefaults

---

## Section 4: SwiftData Fundamentals

### What is SwiftData?

SwiftData is Apple's modern framework for persisting structured data. It uses SQLite under the hood but exposes a Swift-native API with macros, property wrappers, and SwiftUI integration. If you've used an ORM (ActiveRecord, Prisma, SQLAlchemy), the concepts will feel familiar.

### Defining Models with @Model

The `@Model` macro turns a Swift class into a persistent data model:

```swift
import SwiftData

@Model
class Note {
    var title: String
    var content: String
    var createdAt: Date
    var isPinned: Bool

    init(title: String, content: String, isPinned: Bool = false) {
        self.title = title
        self.content = content
        self.createdAt = .now
        self.isPinned = isPinned
    }
}
```

SwiftData generates the database schema from this class definition. Properties become columns. No separate schema file, no migration tool for simple changes.

**Key points:**
- **Must be a class, not a struct.** SwiftData needs reference semantics for identity tracking and change observation.
- **Properties are automatically persisted.** Use `@Transient` for properties you don't want stored.
- **SwiftData handles identity.** You don't need an explicit `id` property (but you can add one).
- **Use standard Swift types.** String, Int, Double, Bool, Date, Data, URL, UUID, arrays, and optionals are all supported.

### ModelContainer and ModelContext

**ModelContainer** is the database. You configure it once at the app level:

```swift
@main
struct NoteTakerApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .modelContainer(for: Note.self)
    }
}
```

`.modelContainer(for:)` creates a SQLite database, generates the schema from your @Model classes, and injects a `ModelContext` into the SwiftUI environment.

**ModelContext** is your connection to the database. Use it for all data operations:

```swift
@Environment(\.modelContext) private var context
```

Think of it this way: ModelContainer is the database file. ModelContext is your open connection to it.

### CRUD Operations

**Create:**
```swift
let note = Note(title: "Shopping List", content: "Milk, eggs, bread")
context.insert(note)
```

**Read:**
```swift
let descriptor = FetchDescriptor<Note>()
let allNotes = try context.fetch(descriptor)
```

**Update — just modify the object:**
```swift
note.title = "Updated Shopping List"
note.content = "Milk, eggs, bread, cheese"
// SwiftData tracks changes automatically — no save call needed
```

**Delete:**
```swift
context.delete(note)
```

SwiftData auto-saves changes at appropriate times. You can force a save with `try context.save()`, but it's rarely necessary.

### Relationships

SwiftData infers relationships from property types:

```swift
@Model
class Note {
    var title: String
    var content: String
    var createdAt: Date
    var isPinned: Bool
    var tags: [Tag]

    init(title: String, content: String, isPinned: Bool = false) {
        self.title = title
        self.content = content
        self.createdAt = .now
        self.isPinned = isPinned
        self.tags = []
    }
}

@Model
class Tag {
    var name: String
    var notes: [Note]

    init(name: String) {
        self.name = name
        self.notes = []
    }
}
```

`Note` has `tags: [Tag]` and `Tag` has `notes: [Note]` — SwiftData recognizes this as a many-to-many relationship and creates the join table automatically. Assigning a tag to a note updates both sides:

```swift
let tag = Tag(name: "Work")
context.insert(tag)

note.tags.append(tag)
// tag.notes now contains this note automatically
```

For one-to-many relationships, use an optional on the "one" side and an array on the "many" side:

```swift
@Model
class Folder {
    var name: String
    var notes: [Note]
    // ...
}

@Model
class Note {
    var title: String
    var folder: Folder?
    // ...
}
```

### Exercise 4.1: Define Models

**Task:** Define SwiftData models for a task tracker:
- A `TaskItem` with title, notes, due date (optional), priority (high/medium/low), and completion status
- A `Category` with a name and a one-to-many relationship with tasks

<details>
<summary>Hint: Priority as an enum</summary>

You can use an enum for priority. Make it `Codable` and SwiftData will store it:
```swift
enum Priority: String, Codable, CaseIterable {
    case low, medium, high
}
```
</details>

<details>
<summary>Solution</summary>

```swift
import SwiftData

enum Priority: String, Codable, CaseIterable {
    case low, medium, high
}

@Model
class TaskItem {
    var title: String
    var notes: String
    var dueDate: Date?
    var priority: Priority
    var isCompleted: Bool
    var createdAt: Date
    var category: Category?

    init(title: String, notes: String = "", dueDate: Date? = nil,
         priority: Priority = .medium, isCompleted: Bool = false) {
        self.title = title
        self.notes = notes
        self.dueDate = dueDate
        self.priority = priority
        self.isCompleted = isCompleted
        self.createdAt = .now
    }
}

@Model
class Category {
    var name: String
    var tasks: [TaskItem]

    init(name: String) {
        self.name = name
        self.tasks = []
    }
}
```
</details>

### Checkpoint 4

Before moving on, make sure you understand:
- [ ] @Model turns a Swift class into a persistent data model
- [ ] ModelContainer is the database; ModelContext is the connection
- [ ] CRUD: insert(), modify properties directly, delete()
- [ ] Relationships are inferred from property types (arrays = to-many, optional = to-one)
- [ ] SwiftData auto-saves — explicit save() is rarely needed

---

## Section 5: Querying with SwiftData

### #Predicate for Type-Safe Queries

`#Predicate` is a macro for building type-safe database queries:

```swift
// Find pinned notes
let pinned = #Predicate<Note> { note in
    note.isPinned == true
}

// Search by title or content
let search = #Predicate<Note> { note in
    note.title.localizedStandardContains("shopping") ||
    note.content.localizedStandardContains("shopping")
}

// Notes created after a specific date
let recent = #Predicate<Note> { note in
    note.createdAt >= cutoffDate
}

// Combine conditions
let pinnedAndRecent = #Predicate<Note> { note in
    note.isPinned == true && note.createdAt >= cutoffDate
}
```

The compiler checks property names and types — no stringly-typed queries. If you rename a property, the predicate gets a compile error (not a runtime crash).

### SortDescriptor for Ordering

```swift
// Single sort
let byDate = SortDescriptor(\Note.createdAt, order: .reverse)

// Multiple sort criteria: pinned first, then by date
let sorts = [
    SortDescriptor(\Note.isPinned, order: .reverse),  // true before false
    SortDescriptor(\Note.createdAt, order: .reverse)   // newest first
]
```

### FetchDescriptor

`FetchDescriptor` combines predicates, sorts, and limits:

```swift
var descriptor = FetchDescriptor<Note>(
    predicate: #Predicate { $0.isPinned },
    sortBy: [SortDescriptor(\.createdAt, order: .reverse)]
)
descriptor.fetchLimit = 10

let results = try context.fetch(descriptor)
```

### @Query in SwiftUI Views

`@Query` is the SwiftUI integration. It fetches data and keeps it live — when records change anywhere in the app, the array updates automatically:

```swift
struct NoteListView: View {
    @Query(sort: \Note.createdAt, order: .reverse)
    private var notes: [Note]

    var body: some View {
        List(notes) { note in
            Text(note.title)
        }
    }
}
```

You can add filters:

```swift
@Query(
    filter: #Predicate<Note> { $0.isPinned },
    sort: \Note.createdAt,
    order: .reverse
)
private var pinnedNotes: [Note]
```

Multiple @Query properties work independently:

```swift
struct NoteListView: View {
    @Query(
        filter: #Predicate<Note> { $0.isPinned },
        sort: \Note.createdAt, order: .reverse
    ) private var pinnedNotes: [Note]

    @Query(
        sort: \Note.createdAt, order: .reverse
    ) private var allNotes: [Note]

    var body: some View {
        List {
            if !pinnedNotes.isEmpty {
                Section("Pinned") {
                    ForEach(pinnedNotes) { note in
                        NoteRow(note: note)
                    }
                }
            }
            Section("All Notes") {
                ForEach(allNotes) { note in
                    NoteRow(note: note)
                }
            }
        }
    }
}
```

### Dynamic Queries for Search

When the predicate depends on user input (like search text), configure @Query in the view's init:

```swift
struct SearchableNoteList: View {
    @Query private var notes: [Note]

    init(searchText: String) {
        if searchText.isEmpty {
            _notes = Query(sort: \Note.createdAt, order: .reverse)
        } else {
            _notes = Query(
                filter: #Predicate<Note> {
                    $0.title.localizedStandardContains(searchText)
                },
                sort: \Note.createdAt,
                order: .reverse
            )
        }
    }

    var body: some View {
        List(notes) { note in
            NoteRow(note: note)
        }
    }
}

// Parent view drives the search
struct NoteListView: View {
    @State private var searchText = ""

    var body: some View {
        NavigationStack {
            SearchableNoteList(searchText: searchText)
                .searchable(text: $searchText)
                .navigationTitle("Notes")
        }
    }
}
```

The parent passes `searchText` to the child view, which reconfigures @Query. Each time `searchText` changes, SwiftUI recreates `SearchableNoteList` with the new query.

### Exercise 5.1: Build a Filtered List

**Task:** Create a view that displays notes filtered by tag. Accept a `Tag` parameter and use @Query with a predicate to show only notes that contain that tag.

<details>
<summary>Hint</summary>

#Predicate can filter on relationships. For a note that has a tag in its tags array, use:
```swift
#Predicate<Note> { note in
    note.tags.contains(where: { $0.name == tagName })
}
```
Note: Check if this specific predicate works with your SwiftData version — relationship predicates can have limitations. An alternative is fetching the tag and reading its `notes` array directly.
</details>

<details>
<summary>Solution</summary>

```swift
struct TaggedNotesView: View {
    let tag: Tag

    var body: some View {
        List(tag.notes) { note in
            NoteRow(note: note)
        }
        .navigationTitle(tag.name)
    }
}
```

For this case, accessing the relationship directly (`tag.notes`) is simpler than a predicate-based query. SwiftData keeps the relationship live.

If you need sorting or additional filtering, use @Query with a predicate:
```swift
struct TaggedNotesView: View {
    @Query private var notes: [Note]

    init(tagName: String) {
        _notes = Query(
            filter: #Predicate<Note> {
                $0.tags.contains(where: { $0.name == tagName })
            },
            sort: \Note.createdAt,
            order: .reverse
        )
    }

    var body: some View {
        List(notes) { note in
            NoteRow(note: note)
        }
    }
}
```
</details>

### Checkpoint 5

Before moving on, make sure you understand:
- [ ] #Predicate creates type-safe, compiler-checked queries
- [ ] SortDescriptor orders results by one or more properties
- [ ] @Query provides live, auto-updating data in SwiftUI views
- [ ] Dynamic queries use init-based @Query configuration

---

## Section 6: SwiftData and SwiftUI Integration

This section ties everything together — a complete app with create, read, update, and delete operations, all wired through SwiftUI.

### App Setup

Register all model types with the model container:

```swift
@main
struct NoteTakerApp: App {
    var body: some Scene {
        WindowGroup {
            NoteListView()
        }
        .modelContainer(for: [Note.self, Tag.self])
    }
}
```

### Complete Note List View

```swift
struct NoteListView: View {
    @Environment(\.modelContext) private var context
    @Query(sort: \Note.createdAt, order: .reverse) private var notes: [Note]
    @State private var searchText = ""

    var body: some View {
        NavigationStack {
            List {
                ForEach(notes) { note in
                    NavigationLink {
                        NoteDetailView(note: note)
                    } label: {
                        NoteRow(note: note)
                    }
                }
                .onDelete(perform: deleteNotes)
            }
            .searchable(text: $searchText)
            .navigationTitle("Notes")
            .toolbar {
                Button("New Note", systemImage: "plus") {
                    createNote()
                }
            }
            .overlay {
                if notes.isEmpty {
                    ContentUnavailableView(
                        "No Notes",
                        systemImage: "note.text",
                        description: Text("Tap + to create your first note"))
                }
            }
        }
    }

    private func createNote() {
        let note = Note(title: "New Note", content: "")
        context.insert(note)
    }

    private func deleteNotes(at offsets: IndexSet) {
        for index in offsets {
            context.delete(notes[index])
        }
    }
}

struct NoteRow: View {
    let note: Note

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                if note.isPinned {
                    Image(systemName: "pin.fill")
                        .foregroundStyle(.orange)
                        .font(.caption)
                }
                Text(note.title)
                    .font(.headline)
            }
            Text(note.content)
                .lineLimit(2)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Text(note.createdAt, style: .date)
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
    }
}
```

**What's happening:**
- `@Query` provides a live array of all notes, sorted by creation date
- `context.insert()` creates a new note — the @Query array updates immediately
- `.onDelete` with `context.delete()` handles swipe-to-delete
- `ContentUnavailableView` shows an empty state when there are no notes

### Editing with @Bindable

`@Bindable` creates two-way bindings to @Model properties. Changes write directly to the model, and SwiftData persists them automatically:

```swift
struct NoteDetailView: View {
    @Bindable var note: Note

    var body: some View {
        Form {
            Section {
                TextField("Title", text: $note.title)
                TextEditor(text: $note.content)
                    .frame(minHeight: 200)
            }

            Section {
                Toggle("Pinned", isOn: $note.isPinned)
                LabeledContent("Created") {
                    Text(note.createdAt, style: .date)
                }
            }

            if !note.tags.isEmpty {
                Section("Tags") {
                    ForEach(note.tags) { tag in
                        Text(tag.name)
                    }
                }
            }
        }
        .navigationTitle(note.title)
    }
}
```

When the user types in the title `TextField`, it writes directly to `note.title`. SwiftData observes the change and persists it. No save button, no manual sync.

### Tag Management

```swift
struct TagListView: View {
    @Environment(\.modelContext) private var context
    @Query(sort: \Tag.name) private var tags: [Tag]
    @State private var newTagName = ""

    var body: some View {
        List {
            Section {
                HStack {
                    TextField("New tag name", text: $newTagName)
                    Button("Add") {
                        guard !newTagName.isEmpty else { return }
                        let tag = Tag(name: newTagName)
                        context.insert(tag)
                        newTagName = ""
                    }
                }
            }

            Section("Tags") {
                ForEach(tags) { tag in
                    HStack {
                        Text(tag.name)
                        Spacer()
                        Text("\(tag.notes.count) notes")
                            .foregroundStyle(.secondary)
                    }
                }
                .onDelete { offsets in
                    for index in offsets {
                        context.delete(tags[index])
                    }
                }
            }
        }
        .navigationTitle("Tags")
    }
}
```

### Exercise 6.1: Add Tag Assignment

**Task:** Add the ability to assign tags to a note from the NoteDetailView. Show a multi-select list of all tags, with checkmarks on tags already assigned to the note.

<details>
<summary>Hint 1</summary>

Use @Query to get all tags in the detail view. For each tag, check if `note.tags.contains(tag)`. On tap, append or remove the tag.
</details>

<details>
<summary>Hint 2</summary>

Since Tag is a class (@Model), you can compare by identity. But for `contains`, you may need Tag to conform to Equatable (which @Model provides).
</details>

<details>
<summary>Solution</summary>

Add a tag assignment section to NoteDetailView:

```swift
struct NoteDetailView: View {
    @Bindable var note: Note
    @Query(sort: \Tag.name) private var allTags: [Tag]

    var body: some View {
        Form {
            Section {
                TextField("Title", text: $note.title)
                TextEditor(text: $note.content)
                    .frame(minHeight: 200)
            }

            Section {
                Toggle("Pinned", isOn: $note.isPinned)
            }

            Section("Tags") {
                ForEach(allTags) { tag in
                    Button {
                        toggleTag(tag)
                    } label: {
                        HStack {
                            Text(tag.name)
                            Spacer()
                            if note.tags.contains(tag) {
                                Image(systemName: "checkmark")
                                    .foregroundStyle(.blue)
                            }
                        }
                    }
                    .tint(.primary)
                }
            }
        }
        .navigationTitle(note.title)
    }

    private func toggleTag(_ tag: Tag) {
        if let index = note.tags.firstIndex(of: tag) {
            note.tags.remove(at: index)
        } else {
            note.tags.append(tag)
        }
    }
}
```
</details>

### Checkpoint 6

Before moving on, make sure you understand:
- [ ] `.modelContainer(for:)` sets up the database and injects ModelContext
- [ ] @Query provides live data that updates when the database changes
- [ ] @Bindable creates two-way bindings to @Model properties for editing
- [ ] context.insert() and context.delete() handle create and delete
- [ ] Relationship changes (like assigning tags) are tracked automatically

---

## Section 7: Core Data Overview

### What is Core Data?

Core Data is SwiftData's predecessor — Apple's original persistence framework, shipping since 2005 (Mac) and 2009 (iOS). SwiftData was introduced in 2023 and is built on top of Core Data. You'll encounter Core Data in:

- Existing app codebases
- Older tutorials and Stack Overflow answers
- Third-party libraries
- Apple's own sample code (pre-2023)

You don't need to learn Core Data for new projects, but recognizing it helps when you encounter it.

### Key Differences

| Aspect | SwiftData | Core Data |
|--------|-----------|-----------|
| Model definition | `@Model` macro on a Swift class | `.xcdatamodeld` visual editor + generated classes |
| Schema | Inferred from code | Visual model editor in Xcode |
| Query syntax | `#Predicate` (type-safe Swift) | `NSPredicate` (string-based) |
| SwiftUI integration | `@Query`, `@Bindable` | `@FetchRequest` |
| Setup | `.modelContainer()` | `NSPersistentContainer` + boilerplate |
| Concurrency | Swift concurrency native | Complex thread/context rules |

### Recognizing Core Data Code

If you see any of these, you're looking at Core Data:

```swift
// Core Data imports and types
import CoreData
NSManagedObject
NSManagedObjectContext
NSPersistentContainer

// Core Data model files
.xcdatamodeld

// Core Data fetch
let request: NSFetchRequest<CDNote> = CDNote.fetchRequest()
request.predicate = NSPredicate(format: "isPinned == YES")
request.sortDescriptors = [
    NSSortDescriptor(key: "createdAt", ascending: false)
]
let notes = try context.fetch(request)

// Core Data in SwiftUI
@FetchRequest(
    sortDescriptors: [NSSortDescriptor(keyPath: \CDNote.createdAt, ascending: false)]
) var notes: FetchedResults<CDNote>
```

Compare to the SwiftData equivalent:

```swift
// SwiftData query
let descriptor = FetchDescriptor<Note>(
    predicate: #Predicate { $0.isPinned },
    sortBy: [SortDescriptor(\.createdAt, order: .reverse)]
)
let notes = try context.fetch(descriptor)

// SwiftData in SwiftUI
@Query(sort: \Note.createdAt, order: .reverse) var notes: [Note]
```

The key improvement: type safety. `NSPredicate(format: "isPinned == YES")` is a string — typos compile fine but crash at runtime. `#Predicate { $0.isPinned }` is checked by the compiler.

### Migration Path

If you're working on a Core Data project that wants to adopt SwiftData:
- SwiftData can read existing Core Data SQLite stores
- Both frameworks can coexist in the same app
- Models can be migrated one at a time
- Apple provides official migration documentation

Core Data is not deprecated — Apple still maintains it. But SwiftData is recommended for new projects.

### Checkpoint 7

Before moving on, make sure you understand:
- [ ] Core Data is SwiftData's predecessor, still found in existing code
- [ ] Key indicators: NSManagedObject, NSPredicate, .xcdatamodeld, @FetchRequest
- [ ] SwiftData's main advantage is type safety (compile-time checked queries)
- [ ] Both can coexist in the same app for gradual migration

---

## Practice Project

### Project Description

Build a note-taking app that combines multiple persistence strategies: SwiftData for notes and tags, @AppStorage for user preferences, and the Keychain helper for a bonus PIN lock feature.

### Requirements

1. **SwiftData models**: Note (title, content, created date, pinned status) and Tag (name), with many-to-many relationship
2. **Note list**: @Query with sorted, live-updating data. Create and swipe-to-delete.
3. **Note editing**: @Bindable for inline editing of title, content, and pinned state
4. **Tag management**: Create tags, assign tags to notes, delete tags
5. **Preferences**: @AppStorage for sort order (date/title) and a boolean "show pinned first" setting
6. **Dynamic sort**: The note list should respect the sort preference from settings

### Bonus

- Add a simple PIN lock: store a 4-digit PIN in Keychain, show a PIN entry screen on launch

### Getting Started

**Step 1:** Define your @Model classes (Note and Tag with relationships)

**Step 2:** Set up .modelContainer in the app entry point

**Step 3:** Build the note list with @Query, create, and delete

**Step 4:** Build the detail view with @Bindable editing

**Step 5:** Add tag management

**Step 6:** Add a settings screen with @AppStorage

**Step 7:** Wire the sort preference into the note list query

### Hints

<details>
<summary>If the sort preference isn't affecting the list</summary>

You'll need to use the init-based @Query approach (like the search example) to dynamically configure the sort based on the @AppStorage value. Pass the sort preference to a child view that configures @Query in its init.
</details>

<details>
<summary>If relationships aren't working</summary>

Make sure both sides of the relationship are defined. For many-to-many: Note has `tags: [Tag]` and Tag has `notes: [Note]`. Register both types with `.modelContainer(for: [Note.self, Tag.self])`.
</details>

### Example Solution

<details>
<summary>Click to see the key files</summary>

**Models:**
```swift
import SwiftData

@Model
class Note {
    var title: String
    var content: String
    var createdAt: Date
    var isPinned: Bool
    var tags: [Tag]

    init(title: String, content: String, isPinned: Bool = false) {
        self.title = title
        self.content = content
        self.createdAt = .now
        self.isPinned = isPinned
        self.tags = []
    }
}

@Model
class Tag {
    var name: String
    var notes: [Note]

    init(name: String) {
        self.name = name
        self.notes = []
    }
}
```

**App entry point:**
```swift
@main
struct NoteTakerApp: App {
    var body: some Scene {
        WindowGroup {
            NoteListView()
        }
        .modelContainer(for: [Note.self, Tag.self])
    }
}
```

**Note list with dynamic sort:**
```swift
struct NoteListView: View {
    @AppStorage("sortOrder") private var sortOrder = "date"
    @AppStorage("showPinnedFirst") private var showPinnedFirst = true
    @State private var searchText = ""

    var body: some View {
        NavigationStack {
            SortedNoteList(
                sortOrder: sortOrder,
                showPinnedFirst: showPinnedFirst,
                searchText: searchText
            )
            .searchable(text: $searchText)
            .navigationTitle("Notes")
            .toolbar {
                NavigationLink("Settings", destination: SettingsView())
            }
        }
    }
}

struct SortedNoteList: View {
    @Environment(\.modelContext) private var context
    @Query private var notes: [Note]

    init(sortOrder: String, showPinnedFirst: Bool, searchText: String) {
        var sorts: [SortDescriptor<Note>] = []
        if showPinnedFirst {
            sorts.append(SortDescriptor(\.isPinned, order: .reverse))
        }
        switch sortOrder {
        case "title":
            sorts.append(SortDescriptor(\.title))
        default:
            sorts.append(SortDescriptor(\.createdAt, order: .reverse))
        }

        if searchText.isEmpty {
            _notes = Query(sort: sorts)
        } else {
            _notes = Query(
                filter: #Predicate<Note> {
                    $0.title.localizedStandardContains(searchText)
                },
                sort: sorts
            )
        }
    }

    var body: some View {
        List {
            ForEach(notes) { note in
                NavigationLink {
                    NoteDetailView(note: note)
                } label: {
                    NoteRow(note: note)
                }
            }
            .onDelete { offsets in
                for index in offsets {
                    context.delete(notes[index])
                }
            }
        }
        .overlay {
            if notes.isEmpty {
                ContentUnavailableView(
                    "No Notes",
                    systemImage: "note.text",
                    description: Text("Tap + to create a note"))
            }
        }
        .toolbar {
            Button("New Note", systemImage: "plus") {
                let note = Note(title: "New Note", content: "")
                context.insert(note)
            }
        }
    }
}
```

**Settings:**
```swift
struct SettingsView: View {
    @AppStorage("sortOrder") private var sortOrder = "date"
    @AppStorage("showPinnedFirst") private var showPinnedFirst = true

    var body: some View {
        Form {
            Section("Display") {
                Picker("Sort By", selection: $sortOrder) {
                    Text("Date Created").tag("date")
                    Text("Title").tag("title")
                }
                Toggle("Show Pinned First", isOn: $showPinnedFirst)
            }
        }
        .navigationTitle("Settings")
    }
}
```
</details>

---

## Summary

### Key Takeaways

- **Choose the right tool**: UserDefaults for preferences, Keychain for secrets, SwiftData for structured data, file system for large files
- **@AppStorage** makes UserDefaults reactive in SwiftUI
- **Keychain** encrypts sensitive data with hardware-backed security
- **SwiftData** provides a Swift-native ORM with @Model, #Predicate, and @Query
- **@Query** delivers live, auto-updating data from SwiftData to SwiftUI views
- **Core Data** is the predecessor — recognize it, but use SwiftData for new projects

### Skills You've Gained

You can now:
- Choose the appropriate persistence strategy for any data type
- Build reactive settings screens with @AppStorage
- Securely store credentials with Keychain
- Model, query, and persist structured data with SwiftData
- Integrate persistence seamlessly with SwiftUI views

### Self-Assessment

Take a moment to reflect:
- Could you set up SwiftData in a new project from scratch?
- Do you know when to use Keychain vs UserDefaults without hesitation?
- Can you build a @Query-powered list with search and dynamic sorting?

---

## Next Steps

### Continue Learning

**Build on this topic:**
- Add persistence to the book browser app from the ios-app-patterns route
- Practice with more complex SwiftData schemas (multiple relationships, cascading deletes)

**Explore related routes:**
- [CloudKit Integration](/routes/cloudkit-integration/map.md) — Sync SwiftData across devices via iCloud
- [App Store Publishing](/routes/app-store-publishing/map.md) — Ship your data-driven app

### Additional Resources

**Documentation:**
- Apple's SwiftData documentation
- Apple's "Build an app with SwiftData" WWDC tutorial
- Apple's Keychain Services documentation
