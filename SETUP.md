# Setup Guide

This document covers everything you need to get Voice running from scratch on any OS.

## Requirements

- Python 3.9 or higher
- A working microphone
- An internet connection on first run only, because the models are downloaded once and cached

## Step 1 - Create a virtual environment

Using a venv keeps dependencies isolated from your system Python. This is recommended but not required.

**Windows**
```bat
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt once activated.

## Step 2 - Install dependencies

```bash
pip install -r requirements.txt
```

### PortAudio (Linux only)

`sounddevice` requires PortAudio on Linux. Install it before running `pip install`:

```bash
# Debian / Ubuntu
sudo apt install portaudio19-dev

# Fedora
sudo dnf install portaudio-devel

# Arch
sudo pacman -S portaudio
```

On Windows and macOS, PortAudio is bundled with the `sounddevice` wheel, so no extra step is needed.

## Step 3 - Run

```bash
python app.py
```

On first run, the app will:

1. Download any missing Whisper models, from base through medium, which is about 2.2 GB in total
2. Create a platform-specific launcher file for future use
3. Start the server and open http://127.0.0.1:47891 in your browser automatically

## Using the launcher next time

After the first run, a launcher file is created in the project folder. Use it instead of opening a terminal every time.

| OS | File | How to use |
|----|------|------------|
| Windows | `run.bat` | Double-click, or drag to Desktop or the taskbar |
| macOS | `run.command` | Double-click from Finder, or drag to the Dock |
| Linux | `run.sh` | Double-click from the file manager, or run `./run.sh` in a terminal |

The launcher activates the venv automatically if one is present.

## Pre-downloading models manually

Models are downloaded automatically on first run. If you want to pre-download them while you have a good connection, run this once:

```python
from faster_whisper import WhisperModel
for size in ["base", "small", "medium"]:
    WhisperModel(size, device="cpu", compute_type="int8")
    print(f"{size} cached")
```

Models are stored in `~/.cache/huggingface/hub/` and are not downloaded again.

## Model reference

| Model | Download size | RAM usage | Speed on an i3 or i5 CPU | Best for |
|-------|-------------|-----------|---------------------------|----------|
| base | about 145 MB | about 300 MB | about 4x real time | Everyday use, default option |
| small | about 460 MB | about 700 MB | about 1.5x real time | Better accuracy for mixed languages |
| medium | about 1.5 GB | about 2.5 GB | about 0.6x real time | Short clips where accuracy matters |

**Recommendation:** start with `base`. Switch to `small` if you find it missing words, especially for Hindi, Gujarati, or English mixed speech.

## Language detection

The app uses automatic language detection by default. Whisper identifies the language in the first few seconds of audio with about 95 percent accuracy and adds less than 0.1 seconds of overhead. This is the recommended setting for most use cases, especially if you mix languages mid-sentence, such as Hinglish or Gujarati-English.

Manual language selection is useful when:
- Recording very short clips under 3 seconds
- The detector is consistently picking the wrong language for a specific accent

## Troubleshooting

**Microphone not detected**
Make sure your browser has microphone permission for `localhost`. Check your OS audio input settings to confirm the correct mic is set as the default.

**`sounddevice` install fails on Linux**
Install PortAudio first, as described in Step 2, then run `pip install -r requirements.txt` again.

**The server starts but the browser does not open**
Navigate manually to http://127.0.0.1:47891. This can happen if your default browser takes too long to launch before the auto-open timeout.

**Model download is slow or fails**
The Hugging Face CDN can be slow in some regions. Try again or use a VPN. Downloaded portions are cached, so a failed download will resume from where it left off on the next run.

**Port 47891 is already in use**
Another process is using the port. Either stop it or edit the last line of `app.py` and change `port=47891` to any free port such as `47892`.

## Uninstalling

To remove everything cleanly:

1. Delete the project folder
2. Delete the model cache at `~/.cache/huggingface/hub/models--Systran--faster-whisper-*`
3. Delete the venv folder if you created one inside the project
