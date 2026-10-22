# Local-First Smart Security Camera System
## Architectural Design & Implementation Blueprint

---

## 1. System Overview & Core Philosophy

This project defines a local-first, privacy-focused smart security camera and monitoring system tailored for execution on resource-constrained hardware such as a Raspberry Pi 4 (or standard Linux host). 

Rather than relying on continuous high-power generative Vision-Language Models (VLMs) or cloud-based analytics, the system leverages a **Direct Multimodal Embedding Pipeline**. By mapping short 5-second video segments into a shared vector space via SigLIP (or SigLIP 2 / CLIP), the system achieves natural language search capabilities and zero-shot event filtering using a single, lightweight model resident in memory.

### Key Principles
* **Privacy & Local Sovereignty:** All video capture, vectorization, rule evaluation, and natural language searching occur entirely on local hardware with zero cloud dependencies.
* **Asynchronous Decoupling:** Video frame capture operates in a isolated, hyper-lightweight daemon service, completely decoupled from the AI inference and processing pipeline.
* **Zero-Shot Rule Engine:** Natural language prompt matching eliminates the need for distinct object detection models (e.g., YOLO) or complex rule classifiers.
* **Resilient State Management:** SQLite acts as both the persistent metadata layer and the state queue, ensuring zero loss of unprocessed video events across system reboots or service restarts.

---

## 2. High-Level System Architecture

The software architecture is split into three core layers:

```
+-----------------------------------------------------------------------+
|                       1. CAPTURE DAEMON SERVICE                       |
|  - Continuous OpenCV Video Capture Loop                               |
|  - Motion Detection (Pixel Difference Thresholding)                   |
|  - 5-Second MP4 Clip Writer                                           |
|  - SQLite Metadata Ingestion (processed = 0)                          |
+-----------------------------------+-----------------------------------+
                                    |
                                    | ZeroMQ Inter-Process
                                    | Communication (PUSH / PULL)
                                    v
+-----------------------------------------------------------------------+
|                    2. FASTAPI AI & PROCESSING ENGINE                  |
|  - ZeroMQ Async Listener                                              |
|  - Keyframe Sampler (2-3 Frames per 5s Clip)                          |
|  - ONNX INT8 SigLIP Multimodal Encoder (Loaded Resident in RAM)       |
|  - Zero-Shot Natural Language Rule Evaluator (YAML Config)            |
|  - Decision Actions: DISCARD (purge file) / SAVE / ALERT              |
+-----------------------------------+-----------------------------------+
                                    |
                                    | Vector Indexing & Metadata
                                    v
+-----------------------------------------------------------------------+
|                       3. STORAGE & INDEXING LAYER                     |
|  - SQLite Database (`clips` metadata table)                           |
|  - `sqlite-vec` Virtual Table (Row-linked Float32 Vector Storage)     |
|  - Raw MP4 Storage (Local Disk / SSD)                                 |
+-----------------------------------------------------------------------+
```

---

## 3. Component Deep Dive

### Component 1: The Capture Daemon Service
The Capture Daemon runs as an independent Linux system daemon (`systemd`). Its sole focus is physics, image capture, and file I/O.

* **Stream Acquisition:** Continuously reads video frames from an RTSP camera stream or local USB camera module using OpenCV.
* **Motion Analysis:** Maintains a running frame difference buffer to monitor structural motion changes. When motion exceeds a configurable threshold, a recording sequence begins.
* **Clip Windowing:** Records a fixed 5-second window `.mp4` video file saved to local storage. If motion continues, sequential 5-second clips are created.
* **Database Registration:** Immediately after completing file write operations, the daemon writes a record into the primary SQLite database with the file path, timestamp, and an unfulfilled flag (`processed = 0`).
* **Non-Blocking Signaling:** Emits a small message payload (`"process_clip"`) over a local ZeroMQ `PUSH` socket pointing to a Unix Domain Socket (IPC) path. The daemon immediately loops back to camera monitoring without waiting for any acknowledgement or processing completion.

### Component 2: The Storage Layer (`sqlite-vec`)
Storage is powered by SQLite, augmented with the `sqlite-vec` extension for vector search capabilities.

* **Relational Clip Table:** Stores clip metadata including primary key integer IDs, local file paths, start timestamps, processing flags (`processed = 0` or `processed = 1`), and assigned action statuses (`DISCARDED`, `SAVED`, `ALERTED`).
* **Vector Virtual Table:** Built using `sqlite-vec` virtual table constructs (`vec0`). Instead of traditional auto-incrementing IDs or external string mappings, `sqlite-vec` matches vector entries to metadata records by sharing the exact `rowid` of the parent clip.
* **Vector Specifications:** Vector rows store normalized float array embeddings representing the multi-frame visual encoding produced by the SigLIP vision model.

### Component 3: The FastAPI Processing & Inference Engine
The FastAPI service hosts the AI runtime, handles background task queues, and exposes the external API endpoints for natural language searching. The processing engine and API server are co-located in the same process to share the resident SigLIP embedding model in memory, avoiding the cost and complexity of loading duplicate model instances across separate services.

* **Resident Model Initialization:** On startup, the process loads an INT8-quantized ONNX model variant of SigLIP into system memory (RAM/VRAM) once. The model remains permanently resident, preventing model loading overhead during video events or search queries.
* **ZeroMQ Listener Worker:** Runs an internal background loop connected to the ZeroMQ `PULL` socket. Upon receiving a socket ping, it queries SQLite for all pending records where `processed = 0`.
* **Keyframe Extraction:** Opens the corresponding 5-second `.mp4` file, extracts 2 to 3 representative frames (e.g., beginning, middle, and end of clip), resizes them to the required input dimensions (e.g., $224 \times 224$), and formats them into tensors.
* **Rule Engine Evaluation:** Runs the keyframes through SigLIP alongside user-defined text prompts from a YAML configuration file (categorized into `DISCARD`, `SAVE`, and `ALERT`). Cosine similarity matrix operations calculate the nearest semantic prompt match.
* **Action Routing:**
  * **DISCARD:** If the highest matching score corresponds to a discard prompt (e.g., "empty yard", "shadows on wall"), the physical `.mp4` file is deleted from disk, the SQLite entry is flagged, and no vectors are indexed.
  * **SAVE:** The extracted keyframe vectors are inserted into `sqlite-vec`, the `.mp4` file is kept, and the database record is updated to `processed = 1`.
  * **ALERT:** Performs all actions under `SAVE`, plus sends a Telegram notification message with clip details and a snapshot frame.
* **Search Endpoint:** Exposes a `POST /search` route. Receives a natural language string query from the user, converts the text string into the same embedding vector space using SigLIP's text encoder, executes a nearest-neighbor vector search in `sqlite-vec`, and returns matching video clip paths along with exact timestamps.

---

## 4. End-to-End Execution Walkthrough

```
[ Motion Occurs in Field of View ]
               |
               v
1. Capture Daemon detects pixel change
2. Capture Daemon buffers frame data for 5 seconds
3. Saves file: `/var/media/clips/clip_20261022_173000.mp4`
4. Inserts into SQLite: `INSERT INTO clips (file_path, processed) VALUES (..., 0)`
5. Emits ZeroMQ IPC signal: `ipc:///tmp/camera_pipeline.sock`
               |
               v (Daemon immediately resumes watching camera stream)
               |
6. FastAPI ZeroMQ PULL listener wakes up
7. Queries SQLite: `SELECT * FROM clips WHERE processed = 0`
8. Loads video file and extracts 2-3 keyframes (uniformly sampled across clip)
9. Converts keyframes into SigLIP image embeddings
               |
               v
10. Evaluates against `config.yaml` prompt matrix:
    ├── Matches "tree branches blowing"? ───> [Action: DELETE .mp4, SET processed = 1]
    ├── Matches "cat walking on grass"? ────> [Action: SAVE .mp4, INDEX vectors]
    └── Matches "person at front door"? ────> [Action: SAVE .mp4, INDEX vectors, TRIGGER notification]
               |
               v
11. Updates SQLite rowid and commits vector embeddings to `sqlite-vec`
```

---

## 5. Configuration & User Customization Engine

Users define system rules entirely via natural language strings inside a clean configuration file (`config.yaml`). The system processes these strings using zero-shot classification:

```yaml
# User-Defined Semantic Filtering Rules
rules:
  confidence_threshold: 0.25 # Minimum similarity score required to trigger action
  
  discard:
    - "an empty room or hallway"
    - "tree branches or shadows blowing in the wind"
    - "car headlights passing across the wall"
    
  save:
    - "a person walking past or standing"
    - "a dog or cat moving around"
    - "a vehicle pulling into the driveway"
    
  alert:
    - "a person standing near the window at night"
    - "smoke, fire, or unusual hazardous situation"
    - "a delivery driver dropping off a package"
```

---

## 6. Hardware Optimization Strategies for Raspberry Pi 4

To ensure stable 24/7 execution on a Raspberry Pi 4 without triggering CPU thermal throttling or Out-Of-Memory (OOM) faults, the following optimizations apply:

1. **ONNX Runtime with INT8 Quantization:** PyTorch framework binaries are omitted in favor of `onnxruntime`. The SigLIP vision model weights are quantized from FP32 down to INT8 format, reducing memory overhead to ~1.5 GB RAM and accelerating CPU matrix math on ARM Cortex-A72 cores.
2. **Keyframe Sub-sampling:** Rather than encoding every frame of a 5-second clip (150 frames at 30fps), keyframe sampling drops input volume to 2–3 frames per clip, dropping inference processing duration per event to roughly 1–1.5 seconds.
3. **Decoupled SystemD Services:** Separating the Capture Daemon from the FastAPI engine ensures that if memory limits trigger an engine failure under heavy usage, the core video capturing daemon remains untouched and recording uninterrupted.
4. **Automated Vector & File Pruning:** A background rolling cron schedule purges video files and vector table rows older than a user-configured retention limit (e.g., 7 days) to maintain bounded storage bounds on SD/SSD media.

---

## 7. Future Improvements & Roadmap

The following items are documented design considerations deferred from the initial implementation. They should be revisited as the system matures and real-world usage data becomes available.

### Phase 1: Post-Implementation Optimizations
* **Clip Boundary Gap/Overlap Handling:** After the initial implementation is functional, analyze whether frame loss or duplication occurs at 5-second clip boundaries during continuous motion events. Optimize the clip writer to ensure seamless temporal coverage if gaps are detected.
* **Keyframe Selection Strategy:** The current uniform sampling approach (2–3 frames across 5 seconds) is a starting point. Future iterations should explore motion-peak-based keyframe selection — extracting frames at moments of highest structural change within the clip rather than at fixed intervals.

### Phase 2: Resilience & Accuracy
* **Pre-Record Circular Buffer:** Implement a rolling in-memory frame buffer (~3 seconds) so that when motion is first detected, the saved clip includes the moments *before* the trigger event. This prevents missing the initial seconds of every recorded event.
* **ZeroMQ Signal Recovery on Startup:** Add an explicit startup routine to the FastAPI engine that queries SQLite for all `processed = 0` records and processes them immediately, ensuring no clips are lost if the engine was down when the daemon emitted ZeroMQ signals.
* **Confidence Tie-Breaking & Priority Hierarchy:** Introduce a category priority order (`ALERT > SAVE > DISCARD`) so that when multiple rule categories match with similar confidence scores, the higher-priority action always wins. Additionally, consider a minimum confidence *gap* threshold to avoid acting on ambiguous matches.

### Phase 3: User Interface & Extensibility
* **Web-Based Dashboard:** Build a frontend UI allowing users to:
  - Browse and playback saved clips with timeline navigation
  - Perform natural language searches visually
  - Adjust rule prompts and confidence thresholds in real time
  - Define custom action classes beyond DISCARD/SAVE/ALERT with user-specified webhook or script triggers
* **Pluggable Alert System:** Expand beyond Telegram to support configurable notification channels (MQTT, email, push notifications, custom webhooks) via a plugin interface.

### Phase 4: Data Lifecycle Management
* **Automated Data Expiration:** Implement a configurable TTL (time-to-live) policy for stored clips and their associated vector embeddings. A background scheduler should automatically purge expired records from both the filesystem and `sqlite-vec` to keep storage usage bounded on constrained devices.
* **Storage Usage Monitoring:** Add disk usage tracking with configurable watermark thresholds that trigger early pruning or user alerts before storage is exhausted.