# CLAUDE.md

## Project Overview

`fai` — a single-repo Python project for real-time AI character video interaction ("AI Facetime"). A user speaks (or types), an LLM responds, and an animated face speaks the response back via an OpenCV window.

## Tech Stack

- **Language:** Python 3.12+
- **Package manager:** `uv` (use `uv add` to add dependencies, never `pip install`)
- **Testing:** pytest
- **Linting/Formatting:** pre-commit (ruff, mypy)

## Project Structure

```
fai/
├── __init__.py
├── __main__.py          # entry point: python -m fai
├── cli.py               # argument parsing (argparse)
├── perception/          # audio capture + ASR (speech-to-text)
├── dialogue/            # LLM response generation
├── voice/               # TTS (text-to-speech)
├── motion/              # audio → facial animation parameters
├── render/              # OpenCV display of animated face
└── orchestrator/        # main conversation loop, component coordination
```

## CLI Interface

```bash
uv run fai face.jpg           # voice mode (default)
uv run fai face.jpg --text    # text input mode (for debugging)
```

- First positional argument: path to reference face image (required)
- `--text`: use keyboard input instead of microphone

## Architecture

Turn-based conversation loop (no streaming for MVP):

```
1. perception: record user audio → transcribe to text (OpenAI Whisper API)
2. dialogue:   user text → LLM response text (OpenAI / Claude API)
3. voice:      response text → audio .wav (OpenAI TTS / ElevenLabs)
4. motion:     reference image + audio → animated video frames (SadTalker / Wav2Lip)
5. render:     display frames in OpenCV window
6. orchestrator: runs the loop, passes data between components
```

Each component exposes a simple function interface. The orchestrator calls them sequentially.

## Configuration

- API keys are loaded from `.env` file via `python-dotenv`
- See `.env.example` for required variables

## Coding Conventions

- Keep each component as a single file until complexity justifies splitting
- Type hints on all function signatures
- Docstrings on public functions (one-liner is fine)
- Prefer simple functions over classes
- Each component's `__init__.py` re-exports its public API
- Use `pathlib.Path` for file paths, not strings

## Testing

- Framework: pytest
- Run: `uv run pytest`
- **Use functions, not test classes**
- **Use fixtures** (`@pytest.fixture`) for shared setup (e.g., sample audio, reference image)
- Place tests in `tests/` mirroring the component structure:
  ```
  tests/
  ├── conftest.py          # shared fixtures
  ├── test_perception.py
  ├── test_dialogue.py
  ├── test_voice.py
  ├── test_motion.py
  ├── test_render.py
  └── test_orchestrator.py
  ```
- Mock external API calls (OpenAI, ElevenLabs) — never hit real APIs in tests

## Validation

Always run before committing:

```bash
uv run pytest
uv run pre-commit run --all-files
```

## Dependencies (MVP)

- `openai` — Whisper API, LLM, TTS
- `python-dotenv` — .env loading
- `sounddevice` — microphone capture
- `opencv-python` — video display
- `numpy` — array ops
- SadTalker or Wav2Lip — motion/lip-sync (exact dependency TBD)

## Design Principles

- Get it working first, optimize later
- Each component should be replaceable independently
- Minimal abstractions — a function that takes input and returns output
- No streaming for MVP — simple sequential turn-based loop
