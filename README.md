# Voice

Voice is an offline, OS-independent speech-to-text app built with Faster-Whisper, FastAPI, and a browser-based interface.

No cloud, no API keys, no subscriptions. Everything runs locally on your machine.

## What it does

- Record audio directly from your browser
- Transcribe speech locally with Faster-Whisper
- Support 99 languages with automatic language detection
- Keep a searchable history of past recordings
- Copy and edit transcripts

## Quick start

```bash
# 1. Clone or download the project
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python app.py
```

The app opens automatically at http://127.0.0.1:47891. A platform-specific launcher is created on first run: `run.bat` on Windows, `run.sh` on Linux, and `run.command` on macOS. Use that launcher next time.

For full setup instructions including virtual environments, PortAudio, and model pre-downloading, see [SETUP.md](SETUP.md).

## Project structure

```
Voice/
  app.py            # FastAPI backend and WebSocket transcription service
  index.html        # Frontend UI served by FastAPI
  launch.sh         # Helper launcher for Linux
  requirements.txt  # Python dependencies
  README.md         # Project overview
  SETUP.md          # Detailed setup guide
  LICENSE           # MIT license
  assets/           # App icons and static assets
  run.sh            # Linux launcher created on first run
  run.bat           # Windows launcher created on first run
  run.command       # macOS launcher created on first run
```

## Tech stack

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) for local Whisper inference
- [FastAPI](https://fastapi.tiangolo.com/) for the backend and WebSocket server
- [Uvicorn](https://www.uvicorn.org/) as the ASGI server
- Vanilla JavaScript for the frontend, with no framework
