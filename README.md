# fai

Real-time AI character video interaction. Talk to an AI-generated face.

## Quick Start

```bash
uv sync
cp .env.example .env  # add your API keys
uv run fai face.jpg
```

## Usage

```bash
# Voice conversation (default)
uv run fai face.jpg

# Text mode (type instead of talk)
uv run fai face.jpg --text
```

## Architecture

```
You speak → perception (ASR) → dialogue (LLM) → voice (TTS) → motion (lip-sync) → render (display)
```

| Component | Function | MVP Implementation |
|-----------|----------|--------------------|
| `perception/` | Audio capture + transcription | OpenAI Whisper API |
| `dialogue/` | Response generation | OpenAI / Claude API |
| `voice/` | Text-to-speech | OpenAI TTS / ElevenLabs |
| `motion/` | Audio → face animation | SadTalker / Wav2Lip |
| `render/` | Display animated face | OpenCV window |
| `orchestrator/` | Conversation loop + coordination | Python glue |

## Development

```bash
uv sync --dev
uv run pytest
uv run pre-commit run --all-files
```

## Configuration

API keys go in `.env`:

```
OPENAI_API_KEY=sk-...
```
