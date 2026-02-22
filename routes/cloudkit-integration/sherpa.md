---
title: CloudKit Integration
route_map: /routes/cloudkit-integration/map.md
paired_guide: /routes/cloudkit-integration/guide.md
topics:
  - CloudKit
  - iCloud
  - Cloud Sync
  - CKSyncEngine
  - Subscriptions
---

# CloudKit Integration - Sherpa (AI Teaching Guide)

**Purpose**: This sherpa guide helps AI assistants teach CloudKit integration to iOS developers. It covers the full path from local data to cloud-synced data — setting up CloudKit, CRUD with records, syncing with CKSyncEngine, subscriptions, sharing, and error handling.

**Route Map**: See `/routes/cloudkit-integration/map.md` for the high-level overview of this route.
**Paired Guide**: The human-focused content is at `/routes/cloudkit-integration/guide.md`.

---

## Teaching Overview

### Learning Objectives
By the end of this session, the learner should be able to:
- Explain CloudKit's architecture (containers, databases, zones, records)
- Set up CloudKit capabilities and containers in an Xcode project
- Perform CRUD operations with CKRecord and CKDatabase
- Query data with CKQuery and NSPredicate
- Sync local data with CloudKit using CKSyncEngine
- Subscribe to remote changes and handle push notifications
- Share records between users with CKShare
- Handle errors, conflicts, and offline scenarios
- Test CloudKit in development and production environments

### Prior Sessions
Before starting, check `.sessions/index.md` and `.sessions/cloudkit-integration/` for prior session history.

### Prerequisites to Verify
Before starting, verify the learner has:
- Swift proficiency (async/await, error handling, protocols)
- SwiftUI fundamentals
- iOS data persistence (especially SwiftData — this route builds on it)
- An Apple Developer Program membership (paid, required for CloudKit)

**If prerequisites are missing**: If they haven't covered data persistence, suggest ios-data-persistence first — CloudKit builds directly on local persistence. Without a paid developer account, they can follow along conceptually but can't run CloudKit code.

### Learner Preferences Configuration

See swiftui-fundamentals sherpa for the full `.sherpa-config.yml` spec. Use defaults if no config exists.

### Assessment Strategies

**Multiple Choice**: Good for architecture questions (which database for which use case, container vs zone vs record).

**Design Questions**: "Your app has user profiles (private), shared shopping lists, and a public recipe database. Which CloudKit databases and record types would you use?"

**Code Review**: Show CloudKit error handling and ask the learner to identify missing cases.

---

## CloudKit Architecture Reference

| Concept | CloudKit | Firebase Equivalent | Description |
|---------|----------|-------------------|-------------|
| Container | CKContainer | Firebase project | Top-level namespace for your app's data |
| Public database | CKDatabase (.public) | Firestore public collection | Readable by all users, writable by authenticated users |
| Private database | CKDatabase (.private) | Firestore user-scoped doc | User's own data, counts against their iCloud quota |
| Shared database | CKDatabase (.shared) | Firestore shared collection | Data shared between specific users |
| Record | CKRecord | Firestore document | A single data object (like a row in a database) |
| Record type | String identifier | Collection name | Schema category (like a table name) |
| Zone | CKRecordZone | N/A | Grouping within a database for sync boundaries |
| Reference | CKRecord.Reference | Document reference | Link between records (foreign key) |
| Asset | CKAsset | Cloud Storage file | Binary file attached to a record |
| Subscription | CKSubscription | Firestore listener | Push notification when data changes |

---

## Teaching Flow

### Introduction

**What to Cover:**
- CloudKit is Apple's cloud backend built into every iCloud account
- No servers to manage, no database to administer, no authentication to build — iCloud handles it
- Free tier is generous (and private data counts against the user's iCloud quota, not yours)
- Best for: syncing user data across their own devices, sharing data between users, public content databases

**Opening Questions to Assess Level:**
1. "Have you used any backend-as-a-service before? (Firebase, Supabase, AWS Amplify)"
2. "What kind of data does your app need to sync? Between one user's devices, or between different users?"
3. "Have you worked with the CloudKit Dashboard at all?"

**Adapt based on responses:**
- If they know Firebase: Bridge concepts heavily — container ≈ project, record ≈ document, etc.
- If no backend experience: Spend more time on the mental model. Explain client-server concepts as needed.
- If they have a specific app: Use their data model as the running example.

**Opening framing:**
"CloudKit gives you a cloud database, file storage, user authentication (via iCloud), and push notifications — all built into Apple's ecosystem. You don't deploy servers or manage infrastructure. The trade-off: it only works on Apple platforms. If that's your target, it's the path of least resistance for cloud sync."

---

### Section 1: CloudKit Concepts

**Core Concept to Teach:**
CloudKit's architecture: containers hold databases (public, private, shared), databases hold zones, zones hold records. Understanding this hierarchy is essential before writing any code.

**How to Explain:**

"Think of it as a filing system:
- **Container** — your app's dedicated storage space in iCloud (like a Firebase project)
- **Databases** — three per container:
  - **Public** — readable by everyone, writable by authenticated users. Good for shared content (recipes, landmarks, announcements).
  - **Private** — each user's personal data. Only they can access it. Counts against their iCloud quota.
  - **Shared** — data explicitly shared between users via CKShare. Lives in the owner's private database but is accessible to invited participants.
- **Zones** — subdivisions within a database. The default zone works for simple apps. Custom zones enable sync features (fetching changes, atomic commits).
- **Records** — individual data objects. Like rows in a table or documents in Firestore. Each has a record type (like a table name) and fields (like columns).
- **References** — links between records (foreign keys). A note can reference its folder.
- **Assets** — binary files (images, documents) attached to records."

**Firebase Comparison (for learners who know it):**
"If you know Firebase:
- CKContainer ≈ Firebase project
- CKDatabase (public) ≈ Firestore collection with public read rules
- CKDatabase (private) ≈ Firestore docs scoped to a user
- CKRecord ≈ Firestore document
- Record type ≈ Collection name
- CKAsset ≈ Cloud Storage file
- CKSubscription ≈ Firestore real-time listener (but via push notifications)"

**Key Difference from Traditional Backends:**
"You don't define a schema in code and deploy it. You design record types in the CloudKit Dashboard (a web UI), or let them be created implicitly when you first save a record. Records are loosely typed — more like NoSQL documents than SQL rows."

**Common Misconceptions:**
- Misconception: "CloudKit is like iCloud Drive" → Clarify: "iCloud Drive is file-based storage. CloudKit is a structured database service. They're different services that both live under iCloud."
- Misconception: "Private database data costs me money" → Clarify: "Private database data counts against the user's iCloud storage quota, not yours. You pay for public database usage, but the free tier is generous."
- Misconception: "CloudKit works on Android/Web" → Clarify: "CloudKit is Apple-only. There's a CloudKit JS library for web, but it's limited. If you need cross-platform, consider Firebase or Supabase instead."

**Verification Questions:**
1. "What are the three types of databases in a CloudKit container, and when would you use each?"
2. "Where does private database data count against storage — the developer's quota or the user's iCloud quota?"
3. "How is a CKRecord similar to and different from a Firestore document?"

---

### Section 2: Project Setup

**Core Concept to Teach:**
Setting up CloudKit in an Xcode project: enabling the capability, creating a container, and understanding the CloudKit Dashboard.

**How to Explain:**
Walk through the setup steps, showing Xcode screenshots or describing the UI precisely.

**Step-by-step:**
1. Open project settings → Signing & Capabilities → + Capability → "iCloud"
2. Check "CloudKit" under iCloud services
3. Click "+" to create a new CloudKit container (name: `iCloud.com.yourcompany.yourapp`)
4. The container appears in the CloudKit Dashboard at https://icloud.developer.apple.com

**The CloudKit Dashboard:**
"The Dashboard is a web UI for managing your CloudKit data:
- **Schema**: Define record types and their fields
- **Records**: Browse, create, edit, and delete records directly
- **Logs**: See API calls, errors, and performance metrics
- **Telemetry**: Usage statistics

Think of it as your database admin panel."

**Development vs Production:**
"CloudKit has two environments:
- **Development**: Where you work and test. Schema changes are flexible.
- **Production**: Where real users' data lives. Schema changes are additive only (can't remove fields).

You deploy schema from Development to Production when ready. You can't go backwards — production schema changes are permanent."

**Important Setup Note:**
"You need a paid Apple Developer account. The free tier lets you build and test locally, but CloudKit requires a paid membership for full functionality."

**Verification Questions:**
1. "What's the difference between the CloudKit development and production environments?"
2. "Where do you manage CloudKit record types and browse data?"

---

### Section 3: Working with Records

**Core Concept to Teach:**
CKRecord is the fundamental data unit. You create records, set fields, save them to a database, and fetch them back. Records are identified by a CKRecord.ID (record type + unique ID).

**How to Explain:**
1. Show basic CRUD operations
2. Explain supported field types
3. Demonstrate references and assets
4. Show batch operations

**Creating and Saving:**

```swift
import CloudKit

let container = CKContainer.default()
let database = container.privateCloudDatabase

// Create a record
let noteRecord = CKRecord(recordType: "Note")
noteRecord["title"] = "Shopping List"
noteRecord["content"] = "Milk, eggs, bread"
noteRecord["isPinned"] = false
noteRecord["createdAt"] = Date.now

// Save to iCloud
do {
    let savedRecord = try await database.save(noteRecord)
    print("Saved: \(savedRecord.recordID)")
} catch {
    print("Save failed: \(error)")
}
```

**Walk Through:**
- "`CKRecord(recordType:)` creates a record. The record type is a string — like a table name."
- "Fields are set with subscript syntax. CloudKit infers the schema from the data you save."
- "`database.save()` is async — it makes a network call to iCloud."
- "The returned record has a `recordID` — CloudKit's unique identifier."

**Reading:**

```swift
// Fetch a specific record by ID
let recordID = CKRecord.ID(recordName: "unique-id-here")
let record = try await database.record(for: recordID)
let title = record["title"] as? String
```

**Updating:**

```swift
// Fetch, modify, save
let record = try await database.record(for: recordID)
record["title"] = "Updated Shopping List"
try await database.save(record)
```

**Deleting:**

```swift
try await database.deleteRecord(withID: recordID)
```

**Supported Field Types:**
"CKRecord fields support: String, Int, Double, Date, Data, Bool, [String], CLLocation, CKRecord.Reference, CKAsset, and arrays of most of these."

**CKRecord.Reference (Relationships):**

```swift
// Create a folder
let folder = CKRecord(recordType: "Folder")
folder["name"] = "Work"
let savedFolder = try await database.save(folder)

// Create a note that references the folder
let note = CKRecord(recordType: "Note")
note["title"] = "Meeting Notes"
note["folder"] = CKRecord.Reference(
    recordID: savedFolder.recordID,
    action: .deleteSelf  // Delete note when folder is deleted
)
```

"The `.action` parameter controls cascade behavior: `.deleteSelf` means deleting the parent deletes this record. `.none` means the reference becomes dangling."

**CKAsset (Files):**

```swift
let imageURL = // ... local file URL
let asset = CKAsset(fileURL: imageURL)
noteRecord["attachment"] = asset
try await database.save(noteRecord)
```

"CKAsset stores binary files (images, documents). Attach them to records. CloudKit handles upload, download, and CDN distribution."

**Batch Operations:**

```swift
let operation = CKModifyRecordsOperation(
    recordsToSave: [record1, record2, record3],
    recordIDsToDelete: [deleteID1]
)
operation.savePolicy = .changedKeys  // Only upload modified fields
database.add(operation)
```

"Use batch operations when saving or deleting multiple records. More efficient than individual calls."

**Common Misconceptions:**
- Misconception: "CKRecord is strongly typed" → Clarify: "CKRecord uses dynamic key-value storage — fields are accessed by string key and cast to the expected type. It's more like a dictionary than a struct."
- Misconception: "I should save after every field change" → Clarify: "Set all fields, then save once. Each save is a network call."

**Verification Questions:**
1. "What does `CKRecord.Reference` action `.deleteSelf` do?"
2. "What types of data can you store in a CKRecord field?"
3. "When would you use CKAsset vs storing data directly in a field?"

---

### Section 4: Querying Data

**Core Concept to Teach:**
CKQuery filters records by type and predicate. NSPredicate provides the query language (similar to Core Data). Cursors handle pagination for large result sets.

**How to Explain:**

**Basic Query:**

```swift
let predicate = NSPredicate(format: "isPinned == %@", NSNumber(value: true))
let query = CKQuery(recordType: "Note", predicate: predicate)
query.sortDescriptors = [
    NSSortDescriptor(key: "createdAt", ascending: false)
]

let (results, _) = try await database.records(matching: query)
let notes = results.compactMap { try? $0.1.get() }
```

**Common Predicates:**

```swift
// All records of a type
NSPredicate(value: true)

// Equality
NSPredicate(format: "title == %@", "Shopping List")

// Contains (case-insensitive search)
NSPredicate(format: "self contains %@", searchTerm)

// Comparison
NSPredicate(format: "createdAt > %@", cutoffDate as NSDate)

// Compound
NSPredicate(format: "isPinned == YES AND createdAt > %@", cutoffDate as NSDate)

// Reference matching
NSPredicate(format: "folder == %@", folderReference)
```

"NSPredicate uses string-based format strings — like Core Data's query language. Not as type-safe as SwiftData's #Predicate, but functional."

**Pagination with Cursors:**

```swift
var allNotes: [CKRecord] = []
var cursor: CKQueryOperation.Cursor?

repeat {
    let results: (
        matchResults: [(CKRecord.ID, Result<CKRecord, Error>)],
        queryCursor: CKQueryOperation.Cursor?
    )

    if let cursor {
        results = try await database.records(continuingMatchFrom: cursor)
    } else {
        let query = CKQuery(recordType: "Note",
                           predicate: NSPredicate(value: true))
        results = try await database.records(matching: query)
    }

    let records = results.matchResults.compactMap { try? $0.1.get() }
    allNotes.append(contentsOf: records)
    cursor = results.queryCursor
} while cursor != nil
```

"CloudKit returns results in batches. If there's more data, you get a cursor to continue fetching. Loop until the cursor is nil."

**Verification Questions:**
1. "Write an NSPredicate that finds notes containing a search term created in the last 7 days"
2. "Why does CloudKit use cursors instead of offset-based pagination?"
3. "What does `NSPredicate(value: true)` match?"

---

### Section 5: Syncing with CKSyncEngine

**Core Concept to Teach:**
CKSyncEngine automates the sync between a local data store and CloudKit. It handles sending local changes, fetching remote changes, managing sync state, and conflict resolution. Introduced in iOS 17, it replaces manual sync logic that was previously error-prone.

**How to Explain:**
1. Explain what CKSyncEngine handles (and what you're responsible for)
2. Walk through the setup and delegate
3. Cover conflict resolution
4. Show integration with SwiftData

**What CKSyncEngine Handles:**
- "Tracking which local changes need to be uploaded"
- "Batching and scheduling uploads efficiently"
- "Fetching remote changes since last sync"
- "Managing sync tokens (bookmarks of where you left off)"
- "Handling network availability — queuing when offline"

**What You Handle:**
- "Converting between your local model and CKRecord"
- "Applying remote changes to your local store"
- "Resolving conflicts when the same record is modified on multiple devices"

**Basic Setup:**

```swift
import CloudKit

class SyncManager: CKSyncEngineDelegate {
    let syncEngine: CKSyncEngine

    init() {
        let configuration = CKSyncEngine.Configuration(
            database: CKContainer.default().privateCloudDatabase,
            stateSerialization: loadSyncState(),
            delegate: self
        )
        syncEngine = CKSyncEngine(configuration)
    }

    // Tell the engine about local changes to upload
    func scheduleSend(_ recordID: CKRecord.ID) {
        syncEngine.state.add(
            pendingRecordZoneChanges: [
                .saveRecord(recordID)
            ]
        )
    }

    // DELEGATE: Engine needs records to upload
    func nextRecordZoneChangeBatch(
        _ context: CKSyncEngine.SendChangesContext
    ) async -> CKSyncEngine.RecordZoneChangeBatch? {
        let pendingChanges = syncEngine.state.pendingRecordZoneChanges
        // Convert local models to CKRecords and return as a batch
        // ...
    }

    // DELEGATE: Engine fetched remote changes
    func handleEvent(_ event: CKSyncEngine.Event) async {
        switch event {
        case .stateUpdate(let update):
            saveSyncState(update.stateSerialization)

        case .fetchedRecordZoneChanges(let changes):
            for modification in changes.modifications {
                // Apply remote change to local store
                applyRemoteChange(modification.record)
            }
            for deletion in changes.deletions {
                // Remove from local store
                applyRemoteDeletion(deletion.recordID)
            }

        case .sentRecordZoneChanges(let sent):
            for saved in sent.savedRecords {
                // Update local record with server version
                updateLocalRecord(with: saved)
            }
            for failed in sent.failedRecordSaves {
                handleSaveFailure(failed)
            }

        default:
            break
        }
    }
}
```

**Walk Through:**
- "The sync engine runs automatically once configured. It fetches remote changes and notifies you via the delegate."
- "`scheduleSend` tells the engine 'this record needs uploading.' The engine batches and sends when appropriate."
- "`handleEvent` is where you react to sync events — applying remote changes locally, handling upload results."
- "State serialization persists the sync position between app launches."

**Conflict Resolution:**

"When the same record is modified on two devices, CloudKit detects the conflict. Your delegate receives the server record and the conflict, and you decide how to resolve it."

"Common strategies:
1. **Server wins**: Accept the server version, discard local changes
2. **Client wins**: Overwrite the server with local changes
3. **Merge**: Combine changes field by field (most complex but best UX)
4. **Last write wins**: Based on timestamp (simple but can lose data)"

```swift
func handleSaveFailure(_ failure: CKSyncEngine.RecordZoneChangeBatch.FailedRecordSave) {
    if case .serverRecordChanged(let serverRecord) = failure.error {
        // Server has a newer version — merge or choose a winner
        let resolved = resolveConflict(
            local: failure.record,
            server: serverRecord
        )
        // Save the resolved record
        scheduleSend(resolved.recordID)
    }
}
```

**SwiftData Integration:**
"If using SwiftData, the sync manager converts between @Model objects and CKRecords. SwiftData can also use NSPersistentCloudKitContainer under the hood for automatic sync — but CKSyncEngine gives you more control."

"For automatic SwiftData + CloudKit sync (less control but simpler setup):"
```swift
// This enables automatic CloudKit sync with SwiftData
.modelContainer(for: Note.self,
                isAutosaveEnabled: true,
                isUndoEnabled: true,
                cloudKitDatabase: .private("iCloud.com.yourapp"))
```

"This is the simplest path — SwiftData handles sync automatically. Use CKSyncEngine when you need custom conflict resolution or more control over what syncs and when."

**Common Misconceptions:**
- Misconception: "CKSyncEngine syncs everything automatically" → Clarify: "It handles the network and scheduling, but you provide the records and apply remote changes. It's a framework, not magic."
- Misconception: "SwiftData CloudKit sync and CKSyncEngine are the same" → Clarify: "SwiftData's CloudKit sync uses NSPersistentCloudKitContainer under the hood. CKSyncEngine is a lower-level API for custom sync logic. Choose based on control vs simplicity."

**Verification Questions:**
1. "What are you responsible for when using CKSyncEngine vs what does the engine handle?"
2. "Describe three strategies for resolving a conflict when two devices modify the same record"
3. "When would you use CKSyncEngine vs SwiftData's built-in CloudKit sync?"

---

### Section 6: Subscriptions and Notifications

**Core Concept to Teach:**
CloudKit subscriptions notify your app when data changes on the server. Combined with silent push notifications, they keep the UI current when changes happen on another device.

**How to Explain:**

**Database Subscription:**

```swift
let subscription = CKDatabaseSubscription(subscriptionID: "all-changes")
subscription.notificationInfo = CKSubscription.NotificationInfo(
    shouldSendContentAvailable: true  // Silent push
)

try await database.save(subscription)
```

"This subscribes to all changes in the database. When any record changes (from any device), your app receives a silent push notification."

**Handling Notifications:**

```swift
// In your App delegate or app lifecycle
func handleRemoteNotification(_ userInfo: [AnyHashable: Any]) {
    let notification = CKNotification(fromRemoteNotificationDictionary: userInfo)
    if notification?.notificationType == .database {
        // Trigger a sync
        syncManager.fetchChanges()
    }
}
```

"Silent push notifications wake your app in the background to sync. The user doesn't see an alert — the data just appears updated."

**Note:** "If using CKSyncEngine, it handles subscriptions and notifications automatically. You only need manual subscription management if you're building custom sync logic."

**Verification Questions:**
1. "What's the difference between a silent push and a visible push notification?"
2. "Why are subscriptions important for keeping multiple devices in sync?"

---

### Section 7: Sharing

**Core Concept to Teach:**
CKShare enables sharing records between iCloud users. The owner creates a share, invites participants, and participants access the shared records through their shared database.

**How to Explain:**

**Creating a Share:**

```swift
let share = CKShare(rootRecord: noteRecord)
share[CKShare.SystemFieldKey.title] = "Shared Note"
share.publicPermission = .readOnly

let operation = CKModifyRecordsOperation(
    recordsToSave: [noteRecord, share]
)
try await database.modifyRecords(saving: [noteRecord, share], deleting: [])
```

**Sharing UI:**

```swift
import UIKit
import SwiftUI

struct CloudSharingView: UIViewControllerRepresentable {
    let share: CKShare
    let container: CKContainer

    func makeUIViewController(context: Context) -> UICloudSharingController {
        let controller = UICloudSharingController(share: share, container: container)
        controller.delegate = context.coordinator
        return controller
    }

    func updateUIViewController(_ uiViewController: UICloudSharingController,
                                 context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    class Coordinator: NSObject, UICloudSharingControllerDelegate {
        func cloudSharingController(_ csc: UICloudSharingController,
                                     failedToSaveShareWithError error: Error) {
            print("Share failed: \(error)")
        }

        func itemTitle(for csc: UICloudSharingController) -> String? {
            return "Shared Note"
        }
    }
}
```

"UICloudSharingController provides Apple's standard sharing UI — invite via Messages, Mail, or link. It handles permissions, participant management, and invitation delivery."

**Accessing Shared Data:**

```swift
let sharedDatabase = container.sharedCloudDatabase

let query = CKQuery(recordType: "Note", predicate: NSPredicate(value: true))
let (results, _) = try await sharedDatabase.records(matching: query)
```

"Shared records appear in the participant's shared database. The API is the same — query, read, modify (if they have write permission)."

**Common Misconceptions:**
- Misconception: "Sharing copies the data to the other user" → Clarify: "Sharing grants access to the original record. Changes by any participant are visible to all."
- Misconception: "Public database is the same as sharing" → Clarify: "Public database is anonymous — everyone sees everything. Sharing is targeted — specific users get access to specific records."

**Verification Questions:**
1. "What's the difference between public database access and CKShare?"
2. "Where do shared records appear for the participant — private or shared database?"
3. "What does UICloudSharingController handle for you?"

---

### Section 8: Error Handling and Edge Cases

**Core Concept to Teach:**
CloudKit operations fail for many reasons — network issues, rate limiting, account changes, quota limits. Robust error handling is essential for a good user experience.

**How to Explain:**

**Common Errors:**

```swift
do {
    try await database.save(record)
} catch let error as CKError {
    switch error.code {
    case .networkUnavailable, .networkFailure:
        // No internet — queue for retry
        queueForRetry(record)

    case .serverRecordChanged:
        // Conflict — another device modified this record
        if let serverRecord = error.serverRecord {
            resolveConflict(local: record, server: serverRecord)
        }

    case .quotaExceeded:
        // User's iCloud storage is full
        showAlert("iCloud storage is full. Free up space in Settings.")

    case .notAuthenticated:
        // User is not signed into iCloud
        showAlert("Please sign in to iCloud in Settings.")

    case .requestRateLimited:
        // Too many requests — wait and retry
        let retryAfter = error.retryAfterSeconds ?? 3.0
        try await Task.sleep(for: .seconds(retryAfter))
        try await database.save(record)

    case .zoneNotFound:
        // Zone doesn't exist — create it first
        try await createZone()
        try await database.save(record)

    case .partialFailure:
        // Batch operation — some succeeded, some failed
        if let partialErrors = error.partialErrorsByItemID {
            for (id, itemError) in partialErrors {
                handleItemError(id: id, error: itemError)
            }
        }

    default:
        print("CloudKit error: \(error.localizedDescription)")
    }
}
```

**Walk Through:**
- "Every CloudKit call can fail. Network issues are the most common — always handle offline gracefully."
- "`serverRecordChanged` is the conflict error — handle it with your conflict resolution strategy."
- "`requestRateLimited` includes a `retryAfterSeconds` — respect it to avoid being throttled further."
- "`partialFailure` means some items in a batch succeeded and others failed — check each one."

**Offline Strategy:**
"Your app should work offline. Store data locally (SwiftData), sync when connectivity returns. CKSyncEngine handles this automatically — it queues changes when offline and sends them when the network comes back."

**Account Changes:**
"Users can sign out of iCloud, switch accounts, or disable iCloud for your app. Handle these cases:
- Subscribe to `CKAccountChanged` notification
- Check account status on launch
- Clear local caches if the account changes"

```swift
NotificationCenter.default.addObserver(
    forName: .CKAccountChanged,
    object: nil, queue: .main
) { _ in
    Task {
        let status = try await CKContainer.default().accountStatus()
        switch status {
        case .available: resumeSync()
        case .noAccount: pauseSyncAndShowSignIn()
        case .restricted: showRestrictionMessage()
        default: break
        }
    }
}
```

**Common Misconceptions:**
- Misconception: "CloudKit operations always succeed if the code is right" → Clarify: "Network operations always fail sometimes. Design for failure — handle every error case."
- Misconception: "I can ignore rate limiting" → Clarify: "CloudKit will throttle your app. Respect retryAfterSeconds. Batch operations reduce your request count."

**Verification Questions:**
1. "What should your app do when it gets a `.networkUnavailable` error?"
2. "How do you handle `requestRateLimited`?"
3. "What happens to your app's data if the user signs out of iCloud?"

---

### Section 9: Practice Project

**Project Introduction:**
"Add CloudKit sync to the note-taking app from the ios-data-persistence route. Notes sync across devices, and users can share individual notes with other iCloud users."

**Requirements:**
1. CloudKit capability and container configured
2. Notes sync between the user's devices via private database
3. Basic conflict resolution (server wins or last-write-wins)
4. Offline support — app works without network, syncs when online
5. A share button on notes that uses UICloudSharingController

**Scaffolding Strategy:**
1. **If they want to try alone**: Provide requirements, check in at milestones.
2. **If they want guidance**: Start with setup, then manual CRUD, then CKSyncEngine, then sharing.
3. **If they want the simplest path**: Use SwiftData's built-in CloudKit sync (`.modelContainer` with `cloudKitDatabase` parameter).

**Checkpoints:**
- After setup: "Can you see your container in the CloudKit Dashboard?"
- After CRUD: "Can you create a note and see it in the Dashboard?"
- After sync: "Does a note created on one device appear on another?"
- After error handling: "What happens when you toggle airplane mode?"
- After sharing: "Can you share a note with another iCloud account?"

---

## Wrap-Up and Next Steps

**Review Key Takeaways:**
1. CloudKit provides cloud database, sync, and sharing without managing servers
2. Three databases: public (everyone), private (per user), shared (invited users)
3. CKRecord is the fundamental data unit — loosely typed, dictionary-style access
4. CKSyncEngine automates sync; SwiftData has built-in CloudKit support for simpler cases
5. Subscriptions + push notifications keep devices in sync
6. CKShare enables user-to-user sharing with Apple's built-in UI
7. Always handle network errors, conflicts, account changes, and offline scenarios

**Suggest Next Steps:**
- "App Store Publishing covers shipping your CloudKit-enabled app"
- "iOS CI/CD with GitHub Actions automates your build and deployment pipeline"

---

## Adaptive Teaching Strategies

### If Learner is Struggling
- Focus on SwiftData's built-in CloudKit sync — it's dramatically simpler than manual CKSyncEngine
- Skip sharing (Section 7) — it's advanced and not needed for basic sync
- Use the CloudKit Dashboard to visualize what's happening — seeing records appear makes concepts concrete
- Simplify the practice project to just "sync notes between devices"

### If Learner is Excelling
- Dive into custom zones and zone-based sync
- Explore CKSyncEngine conflict resolution in depth
- Discuss CloudKit performance optimization (batch operations, denormalization)
- Challenge them to build a real-time collaborative feature

### If Learner Seems Disengaged
- Ask what they're building and focus on the specific CloudKit features they need
- Jump to a working demo — seeing data appear on another device is motivating
- Focus on the "no server" aspect — compare effort to setting up Firebase or a custom backend

---

## Teaching Notes

**Key Emphasis Points:**
- CloudKit setup requires a paid developer account — verify this before starting
- The CloudKit Dashboard is essential for debugging — teach them to use it early
- Error handling is not optional with CloudKit — network operations always fail sometimes
- SwiftData's built-in CloudKit sync is the simplest path for most apps

**Pacing Guidance:**
- Sections 1-2 (Concepts + Setup): Foundation. Don't rush setup — errors here block everything.
- Section 3-4 (Records + Queries): Core operations. Ensure they can CRUD before moving on.
- Section 5 (CKSyncEngine): Most complex section. Take time or offer the SwiftData shortcut.
- Section 6 (Subscriptions): Brief if using CKSyncEngine (it handles this).
- Section 7 (Sharing): Optional for many apps. Cover based on interest.
- Section 8 (Errors): Critical. Don't skip.
- Section 9 (Practice): Scale to available time.

**Success Indicators:**
- They can set up CloudKit and see records in the Dashboard
- They understand which database to use for different data types
- They can implement sync (either CKSyncEngine or SwiftData's built-in)
- They handle errors gracefully, especially offline scenarios
