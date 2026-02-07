# REQUIREMENTS.md

A tech-agnostic specification for fai. An engineer should be able to reimplement the system in any language using this document alone.

---

## 1. Vision

fai is a real-time, face-to-face conversation system with an AI character. The user sees an animated face on screen, speaks naturally (or types), and the character responds with a synthesized voice while its face moves in sync with the audio. The experience should feel as close as possible to a live video call with another person.

The system is modular: every stage of the pipeline — listening, understanding, speaking, animating, displaying — is independently replaceable.

---

## 2. Core Experience

### User Journey

1. The user provides a single portrait image of a face.
2. The system opens a display window showing that face.
3. The user speaks into a microphone (or types text as a fallback).
4. The system transcribes speech to text.
5. An LLM generates a conversational response.
6. The response text is synthesized into spoken audio.
7. The face animates in sync with the audio (lip-sync or breathing animation).
8. The user hears the response audio and sees the animated face.
9. The loop repeats until the user exits.

### Personality

The character behaves like a friendly, curious, slightly witty companion in a video call. Responses are concise (1-3 sentences by default, since they are spoken aloud) and conversational. The character asks follow-up questions when appropriate.

---

## 3. Functional Requirements

### 3.1 Perception (Input)

- **Voice input**: Capture audio from the system microphone for a configurable duration (default: 5 seconds per turn).
- **Speech-to-text**: Send captured audio to a speech recognition service and return the transcribed text.
- **Text input**: Accept typed text as an alternative input mode (useful for debugging or accessibility).
- **Speech recognition model selection**: Allow the user to specify which speech recognition model to use.

### 3.2 Dialogue (Thinking)

- **Response generation**: Given user text and conversation history, produce a natural-language response via a large language model.
- **Streaming response**: Support streaming the response token-by-token so the user sees output incrementally.
- **Conversation history**: Maintain a rolling history of user and assistant messages for context.
- **History trimming**: Automatically trim conversation history when it exceeds a token/message threshold, removing the oldest messages first to stay within limits.
- **Backend selection**: Support multiple LLM providers. The user selects one at launch.
- **Model selection**: Within each provider, allow the user to choose a specific model variant.

### 3.3 Voice (Speaking)

- **Text-to-speech synthesis**: Convert response text into audio (WAV format, mono, 16-bit PCM).
- **Streaming synthesis**: Support streaming audio generation where audio chunks arrive incrementally.
- **Audio playback**: Play synthesized audio through the system speakers. Playback should be non-blocking so animation can render concurrently.
- **Voice selection**: Support multiple voice options per TTS provider. The user selects one at launch.
- **Voice listing**: Provide a way to list all available voices for a given TTS provider.
- **Backend selection**: Support multiple TTS providers. The user selects one at launch.

### 3.4 Motion (Animating)

- **Lip-sync animation**: Given a reference face image and audio, generate video frames where the face's lips move in sync with the audio.
- **Breathing animation**: As a fallback (or in streaming mode), generate subtle breathing/idle animation on the reference face without lip-sync.
- **Streaming animation**: Accept audio chunks incrementally and produce animation frames as chunks arrive.
- **Backend selection**: Support multiple lip-sync backends. The system should auto-detect available backends or fall back to breathing animation.
- **Backend listing**: Provide a way to list all available lip-sync backends.

### 3.5 Render (Displaying)

- **Video display**: Open a window and render video frames at a consistent frame rate.
- **Frame timing**: Respect each frame's timestamp to maintain smooth playback pacing.
- **Graceful close**: The display window should close cleanly when the session ends.

### 3.6 Orchestrator (Coordination)

- **Turn-based loop**: Coordinate all components in sequence: input → dialogue → synthesis → animation → display. Repeat until the user exits.
- **Streaming loop**: In streaming mode, pipeline the stages so that dialogue, synthesis, and animation overlap for lower latency.
- **Keyboard interrupt**: The user exits the conversation at any time via interrupt signal (e.g., Ctrl+C). The system prints a farewell and shuts down cleanly.

---

## 4. Modes of Operation

### 4.1 Standard Mode (Turn-Based)

The default mode. Each conversation turn completes fully before the next begins:

1. Capture and transcribe user speech.
2. Generate the complete LLM response.
3. Synthesize the full audio.
4. Generate all animation frames.
5. Play audio and display animation concurrently.

Supports all features: lip-sync, session recording, all backends.

### 4.2 Streaming Mode (Low-Latency)

Optimized for faster time-to-first-response. Stages overlap where possible:

1. Capture and transcribe user speech.
2. Stream the LLM response (display text chunks as they arrive).
3. Stream audio synthesis (begin audio generation before full text is available, if the TTS provider supports it).
4. Animate with breathing motion (lip-sync not available in this mode).
5. Play audio and display animation as chunks arrive.

Constraints in streaming mode:
- No lip-sync animation (breathing animation only).
- No session recording.
- Latency depends on the TTS provider's streaming capability.

---

## 5. Extensibility Requirements

Each pipeline stage must be independently replaceable through a backend abstraction:

| Stage | What is swappable | Selection mechanism |
|---|---|---|
| Perception | Speech recognition provider/model | CLI flag at launch |
| Dialogue | LLM provider and model | CLI flag at launch |
| Voice | TTS provider and voice | CLI flag at launch |
| Motion | Lip-sync / animation engine | CLI flag at launch, or auto-detect |

### Backend Protocol

Each backend type defines a minimal interface (function signatures or protocol). New backends must satisfy this interface and nothing more. The orchestrator is unaware of backend internals.

### Auto-Detection

For motion backends, the system should attempt to detect which backends are available on the host (e.g., via environment variables or directory checks) and select the best one automatically. If none are available, fall back to breathing animation.

---

## 6. Session Requirements

### 6.1 Recording

- When recording is enabled, each conversation turn saves:
  - User speech audio (WAV).
  - AI response audio (WAV).
  - Animated video frames (video file).
  - Transcript text (user input and AI response).
- All files for a turn are stored in a numbered subdirectory within the session directory.
- A session metadata file (JSON) is written at the end, containing:
  - Session ID (timestamp-based).
  - Creation timestamp.
  - Total turn count.
  - Per-turn data: turn number, timestamp, user text, response text, file references.
  - Configuration metadata: input mode, backends used, voice, model, timeout.
- The user specifies the output directory. Default: `./recordings`.
- Recording is not available in streaming mode.

### 6.2 Playback

- Given a session directory, replay the recorded session:
  - Print each turn's transcript to the terminal.
  - Play the response audio through speakers.
  - Display the response video in the render window.
- Audio and video play concurrently (audio non-blocking, video in the display window).
- Gracefully handle missing audio or video files with warnings.

---

## 7. Quality Requirements

### 7.1 Error Handling

- **API retry**: Transient API failures (connection errors, timeouts, rate limits, server errors) are retried with exponential backoff and jitter.
  - Default: 3 retries, 1-second base delay, 60-second max delay, 2x exponential base.
- **API key validation**: At startup, validate that all required API keys are present for the selected backends. Exit with a clear error message if any are missing.
- **Input validation**: Validate all user-provided inputs at startup:
  - Face image path exists.
  - Timeout is positive (if specified).
  - Selected model is compatible with the selected dialogue backend.
- **Graceful degradation**: If a lip-sync backend is unavailable, fall back to breathing animation rather than failing.

### 7.2 Logging

- Structured logging with configurable verbosity levels (debug, info, warning, error, critical).
- Default level: warning (quiet by default).
- Log key events: conversation start, user input received, LLM response generated, audio synthesized, retries, session save, conversation end.
- The user sets the log level at launch.

### 7.3 Configuration

- All configuration is provided via CLI flags at launch.
- API keys and environment-specific paths are loaded from environment variables (typically via a `.env` file).
- Sensible defaults for all optional settings so the minimal invocation requires only a face image path.

### 7.4 Performance

- In streaming mode, time-to-first-audio should be minimized by pipelining dialogue and synthesis.
- Animation frame rate should be consistent and smooth (target: 25 fps).
- Audio playback and video display should run concurrently, not sequentially.

---

## 8. CLI Interface Specification

The system is invoked from the command line. The minimal invocation requires only a face image path. All other settings have sensible defaults.

### Required Arguments

| Argument | Description |
|---|---|
| `face_image` | Path to the reference face image (portrait photo) |

### Optional Flags

| Flag | Description | Default |
|---|---|---|
| `--text` | Use text input instead of microphone | Off (voice input) |
| `--backend` | Lip-sync backend (auto, specific names, or none) | auto |
| `--dialogue` | Dialogue LLM provider | First available provider |
| `--model` | Specific LLM model within the selected provider | Provider's default |
| `--tts` | TTS provider | First available provider |
| `--voice` | TTS voice name | Provider's default voice |
| `--record` | Enable session recording | Off |
| `--output-dir` | Directory for session recordings | `./recordings` |
| `--stream` | Enable streaming (low-latency) mode | Off (turn-based) |
| `--playback` | Replay a recorded session from a directory path | — |
| `--whisper-model` | Speech recognition model | Provider's default |
| `--timeout` | Timeout in seconds for API calls | Provider SDK default |
| `--log-level` | Logging verbosity | warning |
| `--list-backends` | List available lip-sync backends and exit | — |
| `--list-voices` | List available voices for the selected TTS provider and exit | — |

### Validation Rules

- `face_image` is required unless `--playback`, `--list-backends`, or `--list-voices` is used.
- `--timeout` must be a positive number if specified.
- `--model` must be compatible with the selected `--dialogue` provider.
- `--record` is ignored in `--stream` mode (with a warning).
- `--backend` other than auto/none is ignored in `--stream` mode (with a warning).
