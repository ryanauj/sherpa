---
title: CloudKit Integration
route_map: /routes/cloudkit-integration/map.md
paired_sherpa: /routes/cloudkit-integration/sherpa.md
prerequisites:
  - Swift language proficiency
  - SwiftUI fundamentals
  - iOS data persistence
  - Apple Developer Program membership
topics:
  - CloudKit
  - iCloud
  - Cloud Sync
  - CKSyncEngine
  - Subscriptions
---

# CloudKit Integration

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/cloudkit-integration/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/cloudkit-integration/map.md` for the high-level overview.

## Overview

CloudKit is Apple's cloud backend service built into every iCloud account. It gives your app cloud storage, database sync, user authentication, and sharing — all without running your own servers. This guide covers the full path from local data to cloud-synced, shareable data.

CloudKit is Apple-platform-only. If you need cross-platform sync, consider Firebase or Supabase. If your app targets Apple devices, CloudKit is the path of least resistance.

## Learning Objectives

By the end of this guide, you will be able to:
- Explain CloudKit's architecture and choose the right database for each use case
- Set up CloudKit capabilities and containers in Xcode
- Perform CRUD operations with CKRecord
- Query data with CKQuery and NSPredicate
- Sync data between devices using CKSyncEngine or SwiftData's built-in sync
- Handle subscriptions and push notifications for real-time updates
- Share records between users with CKShare
- Handle errors, conflicts, and offline scenarios

## Prerequisites

Before starting, you should have:
- Swift language proficiency — async/await, error handling, protocols (see swift-for-developers route)
- SwiftUI fundamentals — views, state management, @Environment (see swiftui-fundamentals route)
- iOS data persistence — especially SwiftData (see ios-data-persistence route)
- A paid Apple Developer Program membership (required for CloudKit)

## Setup

You'll need an existing project with local data persistence (from the ios-data-persistence route) or a new project. CloudKit-specific setup is covered in Section 2.

**Important**: CloudKit requires a paid Apple Developer account ($99/year). Without it, you can follow the concepts but can't run CloudKit code on a device or simulator.

---

## Section 1: CloudKit Concepts

### The Architecture

CloudKit organizes data in a hierarchy:

```
CKContainer (your app's space in iCloud)
├── Public Database (shared by all users)
├── Private Database (per-user data)
│   ├── Default Zone
│   └── Custom Zones (for sync)
└── Shared Database (data shared between specific users)
```

**Container** — Your app's dedicated space in iCloud. Created when you enable CloudKit in Xcode. Named like `iCloud.com.yourcompany.yourapp`.

**Databases** — Three per container:

| Database | Who can read | Who can write | Storage quota | Use for |
|----------|-------------|--------------|--------------|---------|
| Public | Everyone | Authenticated users | Your developer quota | Shared content (recipes, landmarks) |
| Private | Only the user | Only the user | User's iCloud quota | Personal data (notes, settings) |
| Shared | Invited participants | Based on permissions | Owner's iCloud quota | Collaboration (shared lists, documents) |

**Zones** — Subdivisions within a database. Every database has a default zone. Custom zones enable advanced sync features (fetching changes since last sync, atomic multi-record commits). Most apps use custom zones for the private database.

**Records** — Individual data objects, identified by a record type (like a table name) and a unique ID. Fields are set by key-value — more like a NoSQL document than a SQL row.

**References** — Links between records (like foreign keys). A note can reference its folder.

**Assets** — Binary files (images, documents) attached to records. CloudKit handles upload, download, and CDN distribution.

### Comparison to Firebase

If you've used Firebase, the mapping is:

| CloudKit | Firebase |
|----------|---------|
| CKContainer | Firebase project |
| Public database | Firestore with public read rules |
| Private database | Firestore user-scoped documents |
| CKRecord | Firestore document |
| Record type | Collection name |
| CKAsset | Cloud Storage file |
| CKSubscription | Firestore real-time listener |
| CKShare | Firestore security rules + sharing logic |

The biggest difference: CloudKit's private database data counts against the user's iCloud quota, not yours. Firebase charges you for all storage and bandwidth.

### Checkpoint 1

Before moving on, make sure you understand:
- [ ] CloudKit has three databases: public, private, shared — each for different use cases
- [ ] Private data counts against the user's iCloud quota, not the developer's
- [ ] Records are loosely typed (key-value), not strongly typed like SwiftData models
- [ ] CloudKit is Apple-only — no Android or cross-platform support

---

## Section 2: Project Setup

### Enabling CloudKit in Xcode

1. Open your project in Xcode
2. Select your app target → **Signing & Capabilities** tab
3. Click **+ Capability** → search for "iCloud" → add it
4. Under iCloud, check **CloudKit**
5. Click **+** under Containers to create a new container
6. Name it `iCloud.com.yourteamid.yourappname`

Xcode creates the container in Apple's systems and adds the necessary entitlements to your project.

### The CloudKit Dashboard

Visit [icloud.developer.apple.com](https://icloud.developer.apple.com) to access the CloudKit Dashboard. This is your admin panel for:

- **Schema**: View and edit record types and their fields
- **Records**: Browse, create, edit, and delete records directly (useful for debugging)
- **Logs**: See API calls, errors, and timing
- **Telemetry**: Usage statistics

The Dashboard is essential for debugging. When something isn't working, check the Dashboard to see what records exist and what errors occurred.

### Development vs Production

CloudKit has two environments:

| Aspect | Development | Production |
|--------|------------|-----------|
| Who uses it | You and testers | Real users |
| Schema changes | Flexible — add, modify, remove | Additive only — can't remove fields |
| Data | Test data | Real user data |
| Reset | Can reset the schema | Cannot reset |

You work in Development, then deploy the schema to Production when ready. Production schema changes are permanent — you can add fields and record types, but you can't remove them.

**Tip**: Design your schema carefully before deploying to production. Test thoroughly in Development first.

### Verifying Setup

After enabling CloudKit, verify it works:

```swift
import CloudKit

func checkCloudKit() async {
    do {
        let status = try await CKContainer.default().accountStatus()
        switch status {
        case .available:
            print("iCloud is available")
        case .noAccount:
            print("No iCloud account — sign in via Settings")
        case .restricted:
            print("iCloud is restricted on this device")
        default:
            print("Unknown status")
        }
    } catch {
        print("CloudKit check failed: \(error)")
    }
}
```

Call this from a `.task` modifier in your root view. If you see "iCloud is available," your setup is correct.

### Checkpoint 2

Before moving on, make sure you understand:
- [ ] CloudKit capability is added in Xcode's Signing & Capabilities
- [ ] The CloudKit Dashboard lets you manage schema and browse records
- [ ] Development environment allows flexible schema changes; Production doesn't
- [ ] Always verify iCloud account status before making CloudKit calls

---

## Section 3: Working with Records

### CKRecord Basics

`CKRecord` is CloudKit's fundamental data unit — a dictionary of fields identified by a record type and unique ID:

```swift
import CloudKit

let container = CKContainer.default()
let database = container.privateCloudDatabase
```

**Create and save:**
```swift
let note = CKRecord(recordType: "Note")
note["title"] = "Shopping List"
note["content"] = "Milk, eggs, bread"
note["isPinned"] = false
note["createdAt"] = Date.now

do {
    let saved = try await database.save(note)
    print("Saved with ID: \(saved.recordID)")
} catch {
    print("Save failed: \(error)")
}
```

Fields are set with subscript syntax. CloudKit infers field types from the values you save. The record type ("Note") is like a table name.

**Fetch by ID:**
```swift
let recordID = CKRecord.ID(recordName: "unique-id-here")
let record = try await database.record(for: recordID)
let title = record["title"] as? String ?? ""
```

**Update:**
```swift
let record = try await database.record(for: recordID)
record["title"] = "Updated Shopping List"
try await database.save(record)
```

Fetch the record, modify fields, save it back. CloudKit tracks versions to detect conflicts.

**Delete:**
```swift
try await database.deleteRecord(withID: recordID)
```

### Supported Field Types

CKRecord fields support: `String`, `Int`, `Double`, `Bool`, `Date`, `Data`, `[String]`, `[Int]`, `CLLocation`, `CKRecord.Reference`, `CKAsset`.

### References (Relationships)

`CKRecord.Reference` links records together, like a foreign key:

```swift
// Create a folder
let folder = CKRecord(recordType: "Folder")
folder["name"] = "Work Notes"
let savedFolder = try await database.save(folder)

// Create a note that belongs to the folder
let note = CKRecord(recordType: "Note")
note["title"] = "Meeting Notes"
note["folder"] = CKRecord.Reference(
    recordID: savedFolder.recordID,
    action: .deleteSelf
)
try await database.save(note)
```

The `action` parameter controls cascading:
- `.deleteSelf` — when the referenced record is deleted, delete this record too
- `.none` — the reference becomes dangling (no cascade)

### Assets (Files)

`CKAsset` attaches binary files to records:

```swift
// Save an image
let imageURL = // ... local file URL on disk
let asset = CKAsset(fileURL: imageURL)
note["photo"] = asset
try await database.save(note)

// Read an image
if let asset = record["photo"] as? CKAsset,
   let fileURL = asset.fileURL {
    let imageData = try Data(contentsOf: fileURL)
    // Use imageData...
}
```

CloudKit handles upload, download, and CDN distribution. Store the asset on the record — don't try to store binary data directly in a field.

### Batch Operations

When saving or deleting multiple records, use batch operations for efficiency:

```swift
let operation = CKModifyRecordsOperation(
    recordsToSave: [note1, note2, note3],
    recordIDsToDelete: [oldNoteID]
)
operation.savePolicy = .changedKeys
operation.modifyRecordsResultBlock = { result in
    switch result {
    case .success: print("Batch complete")
    case .failure(let error): print("Batch failed: \(error)")
    }
}
database.add(operation)
```

`.changedKeys` only uploads modified fields, reducing bandwidth.

### Exercise 3.1: CRUD with CloudKit

**Task:** Write functions to create, read, update, and delete a "Task" record type in CloudKit's private database. Fields: title (String), isCompleted (Bool), dueDate (Date, optional).

<details>
<summary>Hint</summary>

Follow the same pattern as the note examples. For optional fields, only set them if they have a value. When reading, cast with `as?` since fields are dynamic.
</details>

<details>
<summary>Solution</summary>

```swift
import CloudKit

let database = CKContainer.default().privateCloudDatabase

func createTask(title: String, dueDate: Date? = nil) async throws -> CKRecord {
    let task = CKRecord(recordType: "Task")
    task["title"] = title
    task["isCompleted"] = false
    if let dueDate {
        task["dueDate"] = dueDate
    }
    return try await database.save(task)
}

func fetchTask(id: CKRecord.ID) async throws -> CKRecord {
    return try await database.record(for: id)
}

func completeTask(id: CKRecord.ID) async throws {
    let task = try await database.record(for: id)
    task["isCompleted"] = true
    try await database.save(task)
}

func deleteTask(id: CKRecord.ID) async throws {
    try await database.deleteRecord(withID: id)
}
```
</details>

### Checkpoint 3

Before moving on, make sure you understand:
- [ ] CKRecord uses subscript syntax for field access
- [ ] `database.save()` handles both create and update
- [ ] CKRecord.Reference creates relationships between records
- [ ] CKAsset stores binary files attached to records
- [ ] Batch operations are more efficient for multiple records

---

## Section 4: Querying Data

### CKQuery and NSPredicate

`CKQuery` fetches records matching a filter. NSPredicate provides the query language:

```swift
// Fetch all notes
let query = CKQuery(recordType: "Note",
                    predicate: NSPredicate(value: true))
let (results, _) = try await database.records(matching: query)
let notes = results.compactMap { try? $0.1.get() }
```

The results come as an array of `(CKRecord.ID, Result<CKRecord, Error>)` tuples. Use `compactMap` to extract the successful records.

### Common Predicates

```swift
// Match a specific value
NSPredicate(format: "title == %@", "Shopping List")

// Text search (tokenized, not substring)
NSPredicate(format: "self contains %@", "shopping")

// Date comparison
NSPredicate(format: "createdAt > %@", cutoffDate as NSDate)

// Boolean
NSPredicate(format: "isPinned == YES")

// Compound conditions
NSPredicate(format: "isPinned == YES AND createdAt > %@",
            cutoffDate as NSDate)

// Reference matching (notes in a specific folder)
NSPredicate(format: "folder == %@", folderReference)

// All records (no filter)
NSPredicate(value: true)
```

**Note**: NSPredicate uses string-based format strings — not type-safe like SwiftData's #Predicate. Typos in field names compile fine but fail at runtime.

### Sorting

```swift
let query = CKQuery(recordType: "Note",
                    predicate: NSPredicate(value: true))
query.sortDescriptors = [
    NSSortDescriptor(key: "createdAt", ascending: false)
]
```

### Pagination with Cursors

CloudKit returns results in batches. For large datasets, use cursors to fetch additional pages:

```swift
func fetchAllNotes() async throws -> [CKRecord] {
    var allNotes: [CKRecord] = []
    let query = CKQuery(recordType: "Note",
                        predicate: NSPredicate(value: true))
    query.sortDescriptors = [
        NSSortDescriptor(key: "createdAt", ascending: false)
    ]

    var cursor: CKQueryOperation.Cursor?

    // First fetch
    let firstResult = try await database.records(matching: query)
    allNotes += firstResult.matchResults.compactMap { try? $0.1.get() }
    cursor = firstResult.queryCursor

    // Continue fetching while there's more data
    while let activeCursor = cursor {
        let nextResult = try await database.records(
            continuingMatchFrom: activeCursor)
        allNotes += nextResult.matchResults.compactMap { try? $0.1.get() }
        cursor = nextResult.queryCursor
    }

    return allNotes
}
```

### Exercise 4.1: Build a Query

**Task:** Write a function that fetches all uncompleted tasks due within the next 7 days, sorted by due date (earliest first).

<details>
<summary>Hint</summary>

Combine two conditions in the predicate: `isCompleted == NO` and `dueDate` between now and 7 days from now. Use `NSCompoundPredicate` or a format string with `AND`.
</details>

<details>
<summary>Solution</summary>

```swift
func fetchUpcomingTasks() async throws -> [CKRecord] {
    let now = Date.now
    let sevenDaysLater = Calendar.current.date(
        byAdding: .day, value: 7, to: now)!

    let predicate = NSPredicate(
        format: "isCompleted == NO AND dueDate >= %@ AND dueDate <= %@",
        now as NSDate, sevenDaysLater as NSDate
    )

    let query = CKQuery(recordType: "Task", predicate: predicate)
    query.sortDescriptors = [
        NSSortDescriptor(key: "dueDate", ascending: true)
    ]

    let (results, _) = try await database.records(matching: query)
    return results.compactMap { try? $0.1.get() }
}
```
</details>

### Checkpoint 4

Before moving on, make sure you understand:
- [ ] CKQuery combines a record type with an NSPredicate filter
- [ ] NSPredicate uses string-based format (not type-safe)
- [ ] Results come as `(CKRecord.ID, Result<CKRecord, Error>)` tuples
- [ ] Cursors handle pagination for large result sets

---

## Section 5: Syncing with CKSyncEngine

### Two Paths to Sync

There are two ways to sync local data with CloudKit:

**1. SwiftData's built-in CloudKit sync (simpler):**
```swift
.modelContainer(for: Note.self,
                cloudKitDatabase: .private("iCloud.com.yourapp"))
```

This enables automatic sync with minimal code. SwiftData handles everything — record conversion, change tracking, conflict resolution. Use this when you want sync with minimal control.

**2. CKSyncEngine (more control):**

CKSyncEngine (iOS 17+) gives you explicit control over what syncs, how conflicts are resolved, and how remote changes are applied. Use this when you need custom behavior.

### SwiftData + CloudKit (The Simple Path)

If you're using SwiftData from the ios-data-persistence route, adding CloudKit sync can be as simple as:

```swift
@main
struct NoteTakerApp: App {
    var body: some Scene {
        WindowGroup {
            NoteListView()
        }
        .modelContainer(for: [Note.self, Tag.self],
                        cloudKitDatabase: .private("iCloud.com.yourapp"))
    }
}
```

Your @Model classes sync automatically. Changes on one device appear on others. @Query results update live.

**Limitations of the simple path:**
- Conflict resolution is automatic (last write wins)
- No control over what syncs or when
- Limited error visibility
- Requires custom zones (created automatically)

### CKSyncEngine (The Controlled Path)

CKSyncEngine separates concerns: the engine handles network scheduling, batching, and state tracking. You handle record conversion and applying changes.

```swift
import CloudKit

class SyncManager: CKSyncEngineDelegate {
    let syncEngine: CKSyncEngine
    let database = CKContainer.default().privateCloudDatabase

    init() {
        let state = Self.loadSyncState()
        let config = CKSyncEngine.Configuration(
            database: database,
            stateSerialization: state,
            delegate: self
        )
        syncEngine = CKSyncEngine(config)
    }

    // Tell the engine about local changes
    func noteWasModified(_ note: Note) {
        let recordID = CKRecord.ID(recordName: note.id.uuidString)
        syncEngine.state.add(
            pendingRecordZoneChanges: [.saveRecord(recordID)]
        )
    }

    func noteWasDeleted(_ noteID: UUID) {
        let recordID = CKRecord.ID(recordName: noteID.uuidString)
        syncEngine.state.add(
            pendingRecordZoneChanges: [.deleteRecord(recordID)]
        )
    }
}
```

**Providing records to upload:**

```swift
extension SyncManager {
    func nextRecordZoneChangeBatch(
        _ context: CKSyncEngine.SendChangesContext
    ) async -> CKSyncEngine.RecordZoneChangeBatch? {
        let pending = syncEngine.state.pendingRecordZoneChanges

        let batch = await CKSyncEngine.RecordZoneChangeBatch(
            pendingChanges: pending
        ) { recordID in
            // Convert local model to CKRecord
            guard let note = findLocalNote(id: recordID.recordName) else {
                return nil
            }
            let record = CKRecord(recordType: "Note",
                                  recordID: recordID)
            record["title"] = note.title
            record["content"] = note.content
            record["isPinned"] = note.isPinned
            record["createdAt"] = note.createdAt
            return record
        }

        return batch
    }
}
```

**Handling remote changes:**

```swift
extension SyncManager {
    func handleEvent(_ event: CKSyncEngine.Event) async {
        switch event {
        case .stateUpdate(let update):
            Self.saveSyncState(update.stateSerialization)

        case .fetchedRecordZoneChanges(let changes):
            for modification in changes.modifications {
                applyRemoteChange(modification.record)
            }
            for deletion in changes.deletions {
                removeLocalRecord(deletion.recordID)
            }

        case .sentRecordZoneChanges(let sent):
            for failed in sent.failedRecordSaves {
                handleConflict(failed)
            }

        default:
            break
        }
    }

    private func applyRemoteChange(_ record: CKRecord) {
        // Find or create local model, update from record fields
        let title = record["title"] as? String ?? ""
        let content = record["content"] as? String ?? ""
        let isPinned = record["isPinned"] as? Bool ?? false
        // ... update local store
    }
}
```

### Conflict Resolution

When two devices modify the same record, the second save fails with a conflict. You decide how to resolve it:

```swift
private func handleConflict(
    _ failure: CKSyncEngine.RecordZoneChangeBatch.FailedRecordSave
) {
    guard let serverRecord = failure.error.serverRecord else { return }
    let clientRecord = failure.record

    // Strategy 1: Server wins (simplest)
    applyRemoteChange(serverRecord)

    // Strategy 2: Client wins
    // Copy client values onto server record and re-save
    // serverRecord["title"] = clientRecord["title"]
    // scheduleSend(serverRecord.recordID)

    // Strategy 3: Field-level merge (most complex, best UX)
    // Compare each field and pick the latest change
}
```

For most apps, "server wins" or "last write wins" is sufficient. Field-level merging is better but significantly more complex.

### Checkpoint 5

Before moving on, make sure you understand:
- [ ] SwiftData's built-in CloudKit sync is the simplest path (one line of code)
- [ ] CKSyncEngine gives more control over sync behavior and conflict resolution
- [ ] You provide records to upload and apply remote changes locally
- [ ] Conflict resolution is your responsibility — choose a strategy that fits your app

---

## Section 6: Subscriptions and Notifications

### Keeping Devices in Sync

When data changes on the server (from another device), your app needs to know. CloudKit uses subscriptions and push notifications for this.

**If using CKSyncEngine or SwiftData CloudKit sync**, this is handled automatically. You only need manual subscriptions for custom sync logic.

### Database Subscription

Subscribe to all changes in a database:

```swift
let subscription = CKDatabaseSubscription(subscriptionID: "private-changes")
subscription.notificationInfo = CKSubscription.NotificationInfo(
    shouldSendContentAvailable: true  // Silent push — no visible alert
)

do {
    try await database.save(subscription)
    print("Subscribed to changes")
} catch {
    print("Subscription failed: \(error)")
}
```

### Handling Push Notifications

When data changes, your app receives a silent push notification:

```swift
// In your App struct or app delegate
class AppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        didReceiveRemoteNotification userInfo: [AnyHashable: Any]
    ) async -> UIBackgroundFetchResult {
        let notification = CKNotification(
            fromRemoteNotificationDictionary: userInfo)

        if notification?.notificationType == .database {
            // Fetch changes from CloudKit
            await syncManager.fetchRemoteChanges()
            return .newData
        }
        return .noData
    }
}
```

Don't forget to register for remote notifications:

```swift
// At app launch
UIApplication.shared.registerForRemoteNotifications()
```

Silent push notifications wake your app in the background briefly to sync data. The user doesn't see a notification — their data just appears updated.

### Checkpoint 6

Before moving on, make sure you understand:
- [ ] Subscriptions trigger push notifications when data changes on the server
- [ ] Silent push notifications sync data without visible alerts
- [ ] CKSyncEngine and SwiftData CloudKit sync handle this automatically
- [ ] Manual subscriptions are only needed for custom sync logic

---

## Section 7: Sharing

### CKShare

CloudKit sharing lets users share records with other iCloud users. The owner creates a share, invites participants, and participants access shared records through their shared database.

**Creating a Share:**

```swift
// The record to share
let noteRecord = try await database.record(for: noteID)

// Create a share
let share = CKShare(rootRecord: noteRecord)
share[CKShare.SystemFieldKey.title] = "Shared Note"
share.publicPermission = .readOnly  // Anyone with the link can read

// Save both the record and the share
try await database.modifyRecords(
    saving: [noteRecord, share],
    deleting: []
)
```

### Sharing UI with UICloudSharingController

Apple provides a standard sharing UI that handles invitations via Messages, Mail, or link:

```swift
import UIKit
import SwiftUI

struct CloudSharingView: UIViewControllerRepresentable {
    let share: CKShare
    let container: CKContainer

    func makeUIViewController(context: Context)
        -> UICloudSharingController {
        let controller = UICloudSharingController(
            share: share, container: container)
        controller.delegate = context.coordinator
        controller.availablePermissions = [.allowReadOnly, .allowReadWrite]
        return controller
    }

    func updateUIViewController(
        _ uiViewController: UICloudSharingController,
        context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    class Coordinator: NSObject, UICloudSharingControllerDelegate {
        func cloudSharingController(
            _ csc: UICloudSharingController,
            failedToSaveShareWithError error: Error
        ) {
            print("Share failed: \(error)")
        }

        func itemTitle(
            for csc: UICloudSharingController
        ) -> String? {
            return "Shared Note"
        }
    }
}
```

Present it in SwiftUI with a sheet:

```swift
.sheet(isPresented: $showingShareSheet) {
    if let share = currentShare {
        CloudSharingView(share: share,
                        container: CKContainer.default())
    }
}
```

### Accessing Shared Data

Shared records appear in the participant's shared database:

```swift
let sharedDB = CKContainer.default().sharedCloudDatabase

let query = CKQuery(recordType: "Note",
                    predicate: NSPredicate(value: true))
let (results, _) = try await sharedDB.records(matching: query)
```

Participants can read and (if permitted) write shared records using the same CKRecord API.

### Key Points About Sharing

- Sharing grants access to the original record, not a copy — changes are visible to all participants
- Public database is anonymous access for everyone. Sharing is targeted access for specific users.
- Shared data lives in the owner's private database but is accessible through participants' shared database
- `UICloudSharingController` handles the invitation UI, permission management, and participant list

### Checkpoint 7

Before moving on, make sure you understand:
- [ ] CKShare enables targeted sharing between specific iCloud users
- [ ] UICloudSharingController provides Apple's standard sharing UI
- [ ] Shared records appear in the participant's shared database
- [ ] Sharing grants access to the original record, not a copy

---

## Section 8: Error Handling and Edge Cases

### CloudKit Errors

Every CloudKit operation can fail. Network issues are the most common reason, but there are many others. Handle them all:

```swift
do {
    try await database.save(record)
} catch let error as CKError {
    switch error.code {
    case .networkUnavailable, .networkFailure:
        // No internet — queue for retry when connectivity returns
        queueForRetry(record)

    case .serverRecordChanged:
        // Conflict — another device changed this record
        if let serverRecord = error.serverRecord {
            resolveConflict(local: record, server: serverRecord)
        }

    case .quotaExceeded:
        // User's iCloud storage is full
        showUserAlert(
            "iCloud storage is full. "
            + "Free up space in Settings > iCloud.")

    case .notAuthenticated:
        // User isn't signed into iCloud
        showUserAlert("Sign in to iCloud in Settings to sync.")

    case .requestRateLimited:
        // Too many requests — CloudKit includes a wait time
        let retryAfter = error.retryAfterSeconds ?? 3.0
        try await Task.sleep(for: .seconds(retryAfter))
        try await database.save(record)

    case .zoneNotFound:
        // Custom zone doesn't exist yet — create it
        try await createCustomZone()
        try await database.save(record)

    case .partialFailure:
        // Batch operation — some items failed
        if let partialErrors = error.partialErrorsByItemID {
            for (id, itemError) in partialErrors {
                print("Failed for \(id): \(itemError)")
            }
        }

    default:
        print("CloudKit error: \(error.localizedDescription)")
    }
} catch {
    print("Unexpected error: \(error)")
}
```

### Offline Strategy

Your app should work without a network connection:

1. **Store data locally** (SwiftData, UserDefaults)
2. **Queue changes** when offline
3. **Sync when connectivity returns**
4. **Show sync status** to the user (synced, pending, error)

CKSyncEngine and SwiftData's CloudKit sync both handle offline queuing automatically. If you're doing manual sync, you need to track pending changes yourself.

### Account Changes

Users can sign out of iCloud, switch accounts, or disable iCloud for your app. Monitor account status:

```swift
NotificationCenter.default.addObserver(
    forName: .CKAccountChanged,
    object: nil, queue: .main
) { _ in
    Task {
        let status = try await CKContainer.default().accountStatus()
        switch status {
        case .available:
            // Resume syncing
            startSync()
        case .noAccount:
            // Pause sync, show sign-in prompt
            pauseSync()
            showSignInPrompt()
        case .restricted:
            // iCloud is restricted (parental controls, MDM)
            showRestrictionMessage()
        default:
            break
        }
    }
}
```

### Testing Tips

- **Use two iCloud accounts** to test sync and sharing (one on simulator, one on device)
- **Use the CloudKit Dashboard** to verify records are created correctly
- **Toggle airplane mode** to test offline behavior
- **Check the Dashboard logs** for errors that don't surface in your app

### Checkpoint 8

Before moving on, make sure you understand:
- [ ] Every CloudKit operation can fail — handle errors comprehensively
- [ ] `requestRateLimited` includes `retryAfterSeconds` — respect it
- [ ] Monitor `CKAccountChanged` for iCloud sign-in/sign-out
- [ ] Test with multiple accounts, offline mode, and the CloudKit Dashboard

---

## Practice Project

### Project Description

Add CloudKit sync to the note-taking app from the ios-data-persistence route. Notes should sync across the user's devices, and they should be able to share individual notes.

### Requirements

1. CloudKit capability enabled with a container configured
2. Notes sync between the user's devices
3. App works offline — local data persists, syncs when online
4. Basic error handling for network issues and account status
5. A share button that uses UICloudSharingController

### Choose Your Path

**Simple path (recommended for most apps):**
Use SwiftData's built-in CloudKit sync. Add `cloudKitDatabase: .private("iCloud.com.yourapp")` to your model container. Focus the project on sharing and error handling.

**Advanced path:**
Implement CKSyncEngine with manual record conversion. Gives you control over conflict resolution and sync behavior.

### Getting Started

**Step 1:** Enable CloudKit in Xcode and create a container

**Step 2:** Choose your sync approach (SwiftData built-in or CKSyncEngine)

**Step 3:** Verify sync works — create a note on one device, see it appear on another

**Step 4:** Add error handling — test airplane mode, iCloud sign-out

**Step 5:** Add sharing with UICloudSharingController

### Hints

<details>
<summary>If sync isn't working</summary>

- Verify you're signed into iCloud on both devices
- Check the CloudKit Dashboard for records and errors
- Make sure both devices use the same CloudKit container
- Check Xcode console for CloudKit error messages
- Allow a few seconds for sync — it's not instant
</details>

<details>
<summary>If sharing doesn't work</summary>

- Sharing requires a custom zone (SwiftData creates one automatically)
- Both users need iCloud accounts
- Check that UICloudSharingController's delegate is set
- The share and record must be saved in the same operation
</details>

---

## Summary

### Key Takeaways

- **CloudKit** provides cloud database, sync, and sharing without managing servers
- **Three databases**: public (everyone), private (per user), shared (invited users)
- **CKRecord** is loosely typed — dictionary-style field access
- **SwiftData's built-in CloudKit sync** is the simplest path for most apps
- **CKSyncEngine** gives explicit control over sync and conflict resolution
- **CKShare** + **UICloudSharingController** enables user-to-user sharing
- **Error handling is essential** — network operations always fail sometimes
- **Design for offline** — store locally, sync when online

### Skills You've Gained

You can now:
- Set up CloudKit in an Xcode project
- Perform CRUD operations with CKRecord
- Sync data between devices
- Share records between iCloud users
- Handle errors, conflicts, and offline scenarios

### Self-Assessment

- Can you choose between public, private, and shared databases for different data types?
- Could you add CloudKit sync to an existing SwiftData app?
- Do you know how to handle the user signing out of iCloud?

---

## Next Steps

### Continue Learning

**Explore related routes:**
- [App Store Publishing](/routes/app-store-publishing/map.md) — Ship your CloudKit-enabled app
- [iOS CI/CD with GitHub Actions](/routes/ios-ci-cd-with-github-actions/map.md) — Automate builds and deployment

### Additional Resources

**Documentation:**
- Apple's CloudKit documentation
- Apple's "Sharing CloudKit Data with Other iCloud Users" tutorial
- WWDC sessions on CKSyncEngine (2023+)
- CloudKit Dashboard: [icloud.developer.apple.com](https://icloud.developer.apple.com)
