---
title: iOS Data Persistence
route_map: /routes/ios-data-persistence/map.md
paired_guide: /routes/ios-data-persistence/guide.md
topics:
  - SwiftData
  - Core Data
  - UserDefaults
  - Keychain
  - Local Storage
---

# iOS Data Persistence - Sherpa (AI Teaching Guide)

**Purpose**: This sherpa guide helps AI assistants teach iOS data persistence to developers building SwiftUI apps. It covers the full spectrum — from UserDefaults for simple settings to SwiftData for structured data — with a focus on choosing the right tool and integrating persistence with SwiftUI.

**Route Map**: See `/routes/ios-data-persistence/map.md` for the high-level overview of this route.
**Paired Guide**: The human-focused content is at `/routes/ios-data-persistence/guide.md`.

---

## Teaching Overview

### Learning Objectives
By the end of this session, the learner should be able to:
- Choose the right persistence mechanism for different types of data
- Store preferences with UserDefaults and @AppStorage
- Securely store credentials with Keychain
- Model structured data with SwiftData's @Model macro
- Perform CRUD operations and build queries with #Predicate and SortDescriptor
- Integrate SwiftData with SwiftUI using @Query and ModelContainer
- Recognize Core Data patterns in existing code and understand the migration path to SwiftData
- Build an app that combines multiple persistence strategies

### Prior Sessions
Before starting, check `.sessions/index.md` and `.sessions/ios-data-persistence/` for prior session history. If the learner has completed previous sessions on this route, review the summaries to understand what they've covered and pick up where they left off.

### Prerequisites to Verify
Before starting, verify the learner has:
- Swift language proficiency (classes, structs, protocols, async/await)
- SwiftUI fundamentals (views, state management, @Environment)
- Ideally, familiarity with MVVM from the ios-app-patterns route

**If prerequisites are missing**: If SwiftUI is weak, suggest swiftui-fundamentals. If they haven't covered app patterns, the persistence concepts still work but they may put data access in views rather than repositories — that's fine for learning, note they can refactor later.

### Learner Preferences Configuration

Learners can configure their preferred learning style by creating a `.sherpa-config.yml` file in the repository root (gitignored by default). See swiftui-fundamentals sherpa for the full configuration spec.

If no configuration file exists, use defaults (objective tone, mixed assessments, balanced pacing).

### Assessment Strategies

**Multiple Choice Questions:**
- Good for "which persistence option?" decision-making questions
- Example: "Where should you store a user's authentication token? A) UserDefaults B) Keychain C) SwiftData D) A plist file"

**Code Reading Questions:**
- Show SwiftData model definitions and ask what the database schema looks like
- Present @Query usage and ask what data the view displays

**Design Questions:**
- Give an app description and ask which persistence strategies to use for each data type
- These assess the decision framework, not just individual APIs

---

## Persistence Decision Reference

Use this table throughout the session:

| Data Type | Best Option | Why |
|-----------|------------|-----|
| App preferences (theme, sort order) | UserDefaults / @AppStorage | Simple key-value, small data, reactive in SwiftUI |
| User settings (notification prefs) | UserDefaults | Simple key-value |
| Auth tokens, passwords | Keychain | Encrypted, persists across app reinstalls |
| API keys embedded in app | Keychain or config file | Keychain for runtime secrets |
| Structured app data (tasks, notes) | SwiftData | Queryable, relationships, SwiftUI integration |
| Large files (images, PDFs) | File system | Not suited for database storage |
| Cache data | File system + URLCache | Temporary, can be purged by OS |
| Offline API responses | SwiftData or file system | Depends on query needs |

---

## Teaching Flow

### Introduction

**What to Cover:**
- iOS apps have several persistence options, each suited to different kinds of data
- The biggest mistake is using one solution for everything — UserDefaults for complex data, or a database for simple settings
- This route teaches when to use each option and how to implement them
- Primary focus is SwiftData (Apple's modern framework), with coverage of simpler alternatives

**Opening Questions to Assess Level:**
1. "Have you stored data locally in an app before? What did you use?"
2. "Are you familiar with any ORM or database framework (ActiveRecord, SQLAlchemy, Prisma, etc.)?"
3. "What kind of data does your app (or planned app) need to persist?"

**Adapt based on responses:**
- If they've used Core Data: Acknowledge their experience, focus on how SwiftData simplifies things
- If they know ORMs from web: Bridge SwiftData concepts to what they know
- If completely new to persistence: Start from first principles, take time with the decision framework
- If they have a specific app: Use their data types as examples throughout

**Opening framing:**
"Every app needs to remember things — user preferences, authentication tokens, the user's actual data. iOS gives you several tools for this, and picking the right one matters. You wouldn't use a database to store a theme preference, and you wouldn't use UserDefaults to store a thousand notes with tags and search. We'll start with the simplest options and work up to SwiftData for structured data."

---

### Section 1: Choosing a Persistence Strategy

**Core Concept to Teach:**
Different data types need different storage solutions. The key factors are: data complexity, security requirements, query needs, and data size.

**How to Explain:**
1. Present the decision framework as a flowchart
2. Walk through real examples for each category
3. Emphasize that using the wrong tool creates real problems

**Decision Framework:**

"Ask these questions about each piece of data:
1. Is it sensitive (passwords, tokens)? → **Keychain**
2. Is it a simple preference or setting? → **UserDefaults / @AppStorage**
3. Is it a large file (image, PDF, video)? → **File system**
4. Is it structured data you need to query, filter, or relate to other data? → **SwiftData**
5. Is it temporary cache data? → **File system with cache directory**"

**Real Examples:**

"Let's say you're building a note-taking app:
- Theme preference (light/dark) → UserDefaults
- Sort order (newest first, alphabetical) → UserDefaults
- User's login token → Keychain
- The notes themselves (with tags, dates, content) → SwiftData
- Attached images → File system (store the file path in SwiftData)
- Downloaded web content for offline reading → File system + cache"

**Common Mistakes to Warn About:**
- "Don't store passwords in UserDefaults — it's a plaintext plist file. Anyone with device access can read it."
- "Don't store large binary data (images, files) in SwiftData/Core Data. Store the file on disk and keep the path in the database."
- "Don't use SwiftData for simple key-value settings. It's overkill and adds complexity."

**Verification Questions:**
1. "You're building a fitness app. Where would you store: the user's preferred measurement unit, their workout history, their login credentials?"
2. Multiple choice: "Where should you store a user's 'remember me' authentication token? A) UserDefaults B) Keychain C) SwiftData D) A file in the documents directory"

**Good answer indicators:**
- Measurement unit → UserDefaults, workout history → SwiftData, credentials → Keychain (B for multiple choice)
- They understand the reasoning, not just the answer

---

### Section 2: UserDefaults and @AppStorage

**Core Concept to Teach:**
UserDefaults is iOS's simplest persistence — a key-value store for small amounts of non-sensitive data. @AppStorage is a SwiftUI property wrapper that makes UserDefaults values reactive.

**How to Explain:**
1. Show basic UserDefaults API
2. Introduce @AppStorage as the SwiftUI-native approach
3. Cover supported types and limitations
4. Emphasize what NOT to store in UserDefaults

**UserDefaults API:**

```swift
// Write
UserDefaults.standard.set("dark", forKey: "theme")
UserDefaults.standard.set(true, forKey: "showNotifications")
UserDefaults.standard.set(14, forKey: "fontSize")

// Read
let theme = UserDefaults.standard.string(forKey: "theme") ?? "light"
let showNotifications = UserDefaults.standard.bool(forKey: "showNotifications")
let fontSize = UserDefaults.standard.integer(forKey: "fontSize")
```

"Simple key-value storage. Strings, booleans, numbers, dates, arrays of those types, and Data."

**@AppStorage in SwiftUI:**

```swift
struct SettingsView: View {
    @AppStorage("theme") private var theme = "light"
    @AppStorage("fontSize") private var fontSize = 14
    @AppStorage("showNotifications") private var showNotifications = true

    var body: some View {
        Form {
            Picker("Theme", selection: $theme) {
                Text("Light").tag("light")
                Text("Dark").tag("dark")
                Text("System").tag("system")
            }

            Stepper("Font Size: \(fontSize)", value: $fontSize, in: 10...24)

            Toggle("Notifications", isOn: $showNotifications)
        }
    }
}
```

"@AppStorage reads from and writes to UserDefaults automatically. The view re-renders when the value changes, just like @State. The string parameter is the UserDefaults key."

**Web Comparison:**
"@AppStorage is like localStorage in the browser — simple key-value pairs that persist between sessions. The API is similar too: set a key, get a value."

**Limitations and Warnings:**
- "UserDefaults is a plist file. Not encrypted. Don't store secrets."
- "Don't store large amounts of data. It loads entirely into memory at launch."
- "Supported types: Bool, Int, Double, String, URL, Data, arrays/dictionaries of these. No custom types directly (encode to Data first with Codable if needed, but at that point consider SwiftData)."
- "Keys are strings — easy to typo. Consider using an enum or constants for keys."

**Exercise 2.1:**
"Create a settings screen with @AppStorage for: theme choice (light/dark/system), a boolean for haptic feedback, and a number for list page size."

---

### Section 3: Keychain Basics

**Core Concept to Teach:**
Keychain is iOS's secure storage for sensitive data — passwords, authentication tokens, API keys. It's encrypted, persists across app reinstalls, and can be shared between apps from the same developer.

**How to Explain:**
1. Explain why Keychain exists (UserDefaults isn't secure)
2. Show the raw API (verbose) then a simple wrapper
3. Keep it practical — most apps only need save/read/delete for tokens

**Why Keychain:**
"UserDefaults stores a plaintext plist. If someone gets access to the device filesystem, they can read everything. Keychain encrypts data with hardware-backed keys. Even if someone extracts the device storage, they can't read Keychain items without the device passcode."

**The Raw API Is Verbose:**

```swift
// Save to Keychain
let tokenData = "abc123token".data(using: .utf8)!
let query: [String: Any] = [
    kSecClass as String: kSecClassGenericPassword,
    kSecAttrAccount as String: "authToken",
    kSecValueData as String: tokenData
]
SecItemAdd(query as CFDictionary, nil)
```

"The raw Security framework API uses dictionaries with Core Foundation keys. It works, but it's not pleasant to use."

**A Simple Wrapper:**

```swift
enum KeychainHelper {
    static func save(_ data: Data, for key: String) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key
        ]

        // Delete existing item first
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

        guard status == errSecSuccess else {
            if status == errSecItemNotFound { return nil }
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

enum KeychainError: Error {
    case saveFailed(OSStatus)
    case readFailed(OSStatus)
    case deleteFailed(OSStatus)
}
```

**Usage:**
```swift
// Save a token
let token = "eyJhbGciOi..."
try KeychainHelper.save(
    token.data(using: .utf8)!,
    for: "authToken"
)

// Read it back
if let data = try KeychainHelper.read(for: "authToken"),
   let token = String(data: data, encoding: .utf8) {
    print("Token: \(token)")
}

// Delete on logout
try KeychainHelper.delete(for: "authToken")
```

"In practice, many teams use a small Keychain wrapper like this, or a third-party library. The important thing is knowing when to use Keychain (secrets) vs UserDefaults (preferences)."

**Common Misconceptions:**
- Misconception: "UserDefaults is fine for tokens if you encode them" → Clarify: "Encoding doesn't encrypt. The data is still readable. Use Keychain for anything sensitive."
- Misconception: "Keychain is slow" → Clarify: "It's slightly slower than UserDefaults, but you're reading a single token — the difference is imperceptible."

**Verification Questions:**
1. "What's the security difference between UserDefaults and Keychain?"
2. "What happens to Keychain data when the user deletes and reinstalls the app?"
3. Multiple choice: "Which data should go in Keychain? A) User's favorite color B) OAuth refresh token C) The user's name D) App launch count"

**Good answer indicators:**
- They know UserDefaults is plaintext, Keychain is encrypted (B for multiple choice)
- Keychain data persists across reinstalls (UserDefaults does not)

---

### Section 4: SwiftData Fundamentals

**Core Concept to Teach:**
SwiftData is Apple's modern persistence framework for structured data. You define models with the @Model macro, and SwiftData handles the database (SQLite under the hood). It's conceptually similar to ORMs like ActiveRecord, Prisma, or SQLAlchemy.

**How to Explain:**
1. Start with @Model — defining a data model
2. Explain ModelContainer (the database) and ModelContext (the connection)
3. Walk through CRUD operations
4. Show relationships between models
5. Compare to ORMs they may know

**@Model Macro:**

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

"@Model turns a Swift class into a persistent data model. SwiftData generates the database schema from the class definition — the properties become columns. No separate schema file, no migration boilerplate for simple changes."

**Key differences from regular classes:**
- "Must be a class, not a struct — SwiftData needs reference semantics for identity tracking"
- "Properties are automatically persisted. Mark properties with `@Transient` if you don't want them stored."
- "SwiftData handles the database identity — you don't need an explicit `id` property (though you can add one if needed)"

**ModelContainer and ModelContext:**

```swift
// ModelContainer: the database itself
// Set up once, usually at the app level
@main
struct NotesApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .modelContainer(for: Note.self)
    }
}
```

"`.modelContainer(for:)` creates the database and injects a ModelContext into the SwiftUI environment. Every descendant view can access it."

```swift
// ModelContext: your connection to the database
// Use it for CRUD operations
@Environment(\.modelContext) private var context
```

"In ORM terms: ModelContainer is the database connection pool. ModelContext is a unit of work / session."

**CRUD Operations:**

```swift
// CREATE
let note = Note(title: "Shopping List", content: "Milk, eggs, bread")
context.insert(note)

// READ (simple — we'll cover queries in the next section)
let descriptor = FetchDescriptor<Note>()
let allNotes = try context.fetch(descriptor)

// UPDATE — just modify the object
note.title = "Updated Shopping List"
// SwiftData tracks changes automatically

// DELETE
context.delete(note)
```

"Changes are tracked automatically. SwiftData saves them at the right time (usually when the context deems appropriate or when you explicitly call `try context.save()`). You don't need to call save after every change."

**Relationships:**

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

"SwiftData infers the relationship from the types. `Note` has a `tags` array of `Tag` objects, and `Tag` has a `notes` array of `Note` objects — SwiftData sees this as a many-to-many relationship and creates the join table automatically."

**ORM Comparison:**

| Concept | SwiftData | ActiveRecord | Prisma |
|---------|-----------|-------------|--------|
| Model definition | @Model class | class with schema | model in schema.prisma |
| Database setup | .modelContainer(for:) | database.yml + migrate | prisma migrate |
| Connection | ModelContext | implicit | PrismaClient |
| Create | context.insert(object) | Model.create() | client.model.create() |
| Query | FetchDescriptor + #Predicate | Model.where() | client.model.findMany() |
| Update | Modify properties directly | model.update() | client.model.update() |
| Delete | context.delete(object) | model.destroy() | client.model.delete() |
| Relationships | Type annotations | has_many/belongs_to | relation fields |

**Common Misconceptions:**
- Misconception: "SwiftData models should be structs like SwiftUI views" → Clarify: "@Model requires classes. SwiftData needs reference semantics to track identity and changes."
- Misconception: "I need to call save() after every change" → Clarify: "SwiftData auto-saves at appropriate times. You can call save() explicitly if you want to force a save point, but it's usually unnecessary."
- Misconception: "I need to set up the database schema separately" → Clarify: "The @Model macro IS the schema. SwiftData generates the database structure from your Swift class."

**Verification Questions:**
1. "What's the difference between ModelContainer and ModelContext?"
2. "How does SwiftData know to create a many-to-many relationship between Note and Tag?"
3. Code reading: Show a @Model class and ask what the database table looks like
4. "Why does @Model require a class instead of a struct?"

---

### Section 5: Querying with SwiftData

**Core Concept to Teach:**
SwiftData uses #Predicate for type-safe queries and SortDescriptor for ordering. @Query in SwiftUI views provides live, auto-updating query results.

**How to Explain:**
1. Start with FetchDescriptor for manual queries
2. Introduce #Predicate for filtering
3. Show SortDescriptor for ordering
4. Present @Query as the SwiftUI integration

**#Predicate:**

```swift
// All pinned notes
let pinnedNotes = #Predicate<Note> { note in
    note.isPinned == true
}

// Notes matching a search term
let searchPredicate = #Predicate<Note> { note in
    note.title.localizedStandardContains(searchTerm) ||
    note.content.localizedStandardContains(searchTerm)
}

// Notes created today
let today = Calendar.current.startOfDay(for: .now)
let todayNotes = #Predicate<Note> { note in
    note.createdAt >= today
}
```

"#Predicate is a macro that creates type-safe queries. The compiler checks your property names and types — no stringly-typed queries. In ORM terms, it's like Prisma's where clause but enforced at compile time."

**SortDescriptor:**

```swift
// Sort by creation date, newest first
let sort = SortDescriptor(\Note.createdAt, order: .reverse)

// Multiple sort criteria — pinned first, then by date
let sorts = [
    SortDescriptor(\Note.isPinned, order: .reverse),
    SortDescriptor(\Note.createdAt, order: .reverse)
]
```

**FetchDescriptor (Putting It Together):**

```swift
var descriptor = FetchDescriptor<Note>(
    predicate: #Predicate { $0.isPinned == true },
    sortBy: [SortDescriptor(\.createdAt, order: .reverse)]
)
descriptor.fetchLimit = 10  // Pagination

let pinnedNotes = try context.fetch(descriptor)
```

**@Query in SwiftUI:**

```swift
struct NoteListView: View {
    @Query(
        filter: #Predicate<Note> { $0.isPinned == true },
        sort: \Note.createdAt,
        order: .reverse
    ) private var pinnedNotes: [Note]

    @Query(
        sort: \Note.createdAt,
        order: .reverse
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

"@Query is the magic. It fetches data from SwiftData and keeps it live — when you insert, update, or delete notes anywhere in the app, these arrays update automatically, and the view re-renders. It's like having a live database subscription built into the view."

**Dynamic Queries:**

"For search, where the predicate changes based on user input, use init-based @Query configuration:"

```swift
struct SearchableNoteList: View {
    @Query private var notes: [Note]

    init(searchText: String) {
        if searchText.isEmpty {
            _notes = Query(
                sort: \Note.createdAt,
                order: .reverse
            )
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
```

**Common Misconceptions:**
- Misconception: "#Predicate supports any Swift expression" → Clarify: "#Predicate supports a subset of Swift expressions — property access, comparisons, logical operators, string methods. Complex closures or function calls won't work."
- Misconception: "@Query fetches all data into memory" → Clarify: "SwiftData handles batching and lazy loading behind the scenes. For large datasets, use fetchLimit on FetchDescriptor."

**Verification Questions:**
1. "Write a #Predicate that finds notes created in the last 7 days"
2. "What happens to a @Query array when you insert a new note elsewhere in the app?"
3. "How would you implement search that filters notes as the user types?"

---

### Section 6: SwiftData and SwiftUI Integration

**Core Concept to Teach:**
SwiftData integrates deeply with SwiftUI through ModelContainer (at the app level), @Query (in views), and @Environment(\.modelContext) for mutations. This section ties everything together.

**How to Explain:**
1. Show the full setup from app to view
2. Demonstrate creating, editing, and deleting in response to user actions
3. Show how @Query keeps everything in sync

**Full App Setup:**

```swift
@main
struct NotesApp: App {
    var body: some Scene {
        WindowGroup {
            NoteListView()
        }
        .modelContainer(for: [Note.self, Tag.self])
    }
}
```

"`.modelContainer(for:)` with an array registers multiple model types. This creates the database and injects the ModelContext into the environment."

**Complete List View with Create and Delete:**

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
                        VStack(alignment: .leading) {
                            HStack {
                                if note.isPinned {
                                    Image(systemName: "pin.fill")
                                        .foregroundStyle(.orange)
                                }
                                Text(note.title).font(.headline)
                            }
                            Text(note.content)
                                .lineLimit(2)
                                .foregroundStyle(.secondary)
                        }
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
```

**Walk Through:**
- "@Query provides the notes array — automatically sorted and live-updating"
- "context.insert() adds a new note — the @Query array updates immediately"
- "context.delete() removes notes — onDelete provides the swipe-to-delete gesture"
- "No save call needed — SwiftData handles persistence automatically"

**Editing:**

```swift
struct NoteDetailView: View {
    @Bindable var note: Note

    var body: some View {
        Form {
            TextField("Title", text: $note.title)
            TextEditor(text: $note.content)
            Toggle("Pinned", isOn: $note.isPinned)
        }
        .navigationTitle(note.title)
    }
}
```

"@Bindable creates bindings to @Model properties. When the user types in the TextField, it writes directly to the model, and SwiftData persists the change. No save button needed."

**Exercise 6.1:**
"Add a tag management feature: a screen that shows all tags, lets you create new tags, and delete tags. Use @Query for the tag list and ModelContext for mutations."

**Verification Questions:**
1. "How does @Query know when to refresh its results?"
2. "Why don't you need to call context.save() after inserting or deleting?"
3. "What does @Bindable do, and when do you use it instead of @Binding?"

---

### Section 7: Core Data Overview

**Core Concept to Teach:**
Core Data is SwiftData's predecessor. The learner will encounter it in existing projects, older tutorials, and Stack Overflow answers. They don't need to learn it in depth, but they should recognize the patterns.

**How to Explain:**
1. Brief history — Core Data shipped in 2005, SwiftData in 2023
2. Key differences
3. What to recognize when reading Core Data code
4. Migration path

**Key Differences:**

| Aspect | SwiftData | Core Data |
|--------|-----------|-----------|
| Model definition | @Model macro on a Swift class | .xcdatamodeld file + generated classes |
| Schema | Inferred from code | Visual editor in Xcode |
| Query syntax | #Predicate (Swift) | NSPredicate (string-based) |
| SwiftUI integration | @Query, @Bindable | @FetchRequest, manual bindings |
| Setup | .modelContainer() | NSPersistentContainer boilerplate |
| Concurrency | Swift concurrency native | Complex context/thread rules |

**Recognizing Core Data Code:**

"If you see these, you're looking at Core Data:
- `NSManagedObject` or `NSManagedObjectContext`
- `.xcdatamodeld` files (data model editor)
- `NSFetchRequest` and `NSPredicate`
- `@FetchRequest` in SwiftUI
- `NSPersistentContainer` or `NSPersistentCloudKitContainer`"

```swift
// Core Data fetch — for recognition, not memorization
let request: NSFetchRequest<CDNote> = CDNote.fetchRequest()
request.predicate = NSPredicate(format: "isPinned == YES")
request.sortDescriptors = [NSSortDescriptor(key: "createdAt", ascending: false)]
let notes = try context.fetch(request)
```

"Compare to SwiftData:"
```swift
let descriptor = FetchDescriptor<Note>(
    predicate: #Predicate { $0.isPinned },
    sortBy: [SortDescriptor(\.createdAt, order: .reverse)]
)
let notes = try context.fetch(descriptor)
```

"Type-safe vs string-based. That's the core improvement."

**Migration Path:**
"If working on a Core Data project that wants to adopt SwiftData:
1. SwiftData can read existing Core Data stores (same underlying SQLite format)
2. You can migrate models one at a time
3. Both can coexist in the same app
4. Apple provides migration guides in their documentation"

**Common Misconceptions:**
- Misconception: "Core Data is deprecated" → Clarify: "It's not deprecated. Apple still maintains it. SwiftData is built on top of Core Data. But for new projects, SwiftData is the recommended choice."
- Misconception: "I need to learn Core Data before SwiftData" → Clarify: "No. SwiftData is designed to be learned independently. You only need Core Data knowledge if you're working on an existing Core Data codebase."

**Verification Questions:**
1. "If you see `NSManagedObjectContext` in code, what framework is it using?"
2. "What's the main advantage of #Predicate over NSPredicate?"
3. "Can SwiftData and Core Data coexist in the same app?"

---

### Section 8: Practice Project

**Project Introduction:**
"Build a note-taking app that combines multiple persistence strategies: SwiftData for notes and tags, UserDefaults/@AppStorage for preferences, and optionally Keychain for a simple PIN lock."

**Requirements:**
1. Notes with title, content, creation date, and pinned status (SwiftData)
2. Tags with a many-to-many relationship to notes (SwiftData)
3. App preferences: sort order and default view (@AppStorage)
4. List view with @Query, search, create, and swipe-to-delete
5. Detail view with @Bindable for editing
6. Tag management (create, assign to notes, delete)

**Optional Extension:**
- Add a simple PIN lock using Keychain to store the PIN

**Scaffolding Strategy:**
1. **If they want to try alone**: Provide requirements and let them build. Check in at milestones.
2. **If they want guidance**: Build layer by layer — models first, then container setup, then list view, then detail view, then tags, then preferences.
3. **If they're unsure**: Start with a single Note model and list view, then add features incrementally.

**Checkpoints:**
- After models: "Do Note and Tag have a proper many-to-many relationship?"
- After list view: "Does @Query keep the list live when you add or delete notes?"
- After detail view: "Can you edit a note's title and content and see the changes persist?"
- After preferences: "Does changing the sort order in settings affect the list immediately?"

---

## Wrap-Up and Next Steps

**Review Key Takeaways:**
1. Choose the right persistence tool for each data type
2. UserDefaults/@AppStorage for simple preferences
3. Keychain for secrets and credentials
4. SwiftData for structured, queryable data
5. @Query provides live, auto-updating data in SwiftUI views
6. Core Data is the predecessor — recognize it, don't learn it for new projects

**Suggest Next Steps:**
- "CloudKit Integration shows how to sync SwiftData across devices using iCloud"
- "If you want to practice, add persistence to the book browser app from ios-app-patterns"

---

## Adaptive Teaching Strategies

### If Learner is Struggling
- Focus on UserDefaults and SwiftData only — skip Keychain and Core Data
- Use the simplest possible SwiftData model (one class, no relationships)
- Build the @Query list view first to show the payoff, then explain the underlying concepts
- Keep examples concrete — a real app scenario they care about

### If Learner is Excelling
- Dive into SwiftData schema migrations
- Explore #Predicate limitations and workarounds
- Discuss performance optimization (fetch limits, batch operations)
- Introduce ModelActor for background data operations
- Show how to add CloudKit syncing to SwiftData

### If Learner Seems Disengaged
- Ask what kind of app they're building — use their data model as the example
- Jump to the practice project and learn by building
- Focus on @Query — the "magic" of live-updating data is compelling

---

## Teaching Notes

**Key Emphasis Points:**
- The decision framework (Section 1) is the most important conceptual section. Every other section is "how to use the tool you chose."
- @Query is the biggest payoff. Make sure they see data appearing live in their views.
- Don't let them put secrets in UserDefaults. This is a real security issue worth emphasizing.
- SwiftData's simplicity is a feature — don't make it seem more complex than it is.

**Pacing Guidance:**
- Section 1 (Decision Framework): Important conceptual foundation. Don't rush.
- Section 2 (UserDefaults): Quick. They probably already know key-value storage.
- Section 3 (Keychain): Brief. Show the wrapper, emphasize when to use it.
- Section 4-5 (SwiftData + Queries): Core content. Take time here.
- Section 6 (SwiftUI Integration): Where everything clicks. Allow experimentation.
- Section 7 (Core Data): Brief overview. Don't teach Core Data in depth.
- Section 8 (Practice): Maximum time here.

**Success Indicators:**
- They can choose the right persistence tool without hesitation
- They can define SwiftData models and build queries
- @Query feels natural — they understand it provides live data
- They never suggest storing tokens in UserDefaults
