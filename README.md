# Voice

Small, always-on-top dictation window for Fedora/GNOME. Fully offline: audio is captured
directly from the mic and transcribed locally with faster-whisper.

- Native GTK4/libadwaita window with its own dock icon (`dev.local.Voice`)
- Record / stop: click the mic, `Ctrl+R`, or run `./launch.sh --toggle` (bind it to a global shortcut)
- `Esc` discards a recording, `Ctrl+H` toggles history
- Transcript is auto-copied to the clipboard when done
- History lives in `~/.local/share/voice/history.json`; entries older than 30 days are removed at startup
- Models (menu → Model): `small.en` (default, fast), `large-v3-turbo` (most accurate, ~1× realtime on this CPU), `small` (multilingual)
- Speech is transcribed at each pause while you talk, so only the last phrase is left when you stop

## Setup

System packages: `python3-gobject gtk4 libadwaita portaudio`.

```bash
python3 -m venv --system-site-packages .venv   # system site packages provide GTK bindings
.venv/bin/pip install -r requirements.txt
cp dev.local.Voice.desktop ~/.local/share/applications/
install -Dm644 assets/dev.local.Voice.svg ~/.local/share/icons/hicolor/scalable/apps/dev.local.Voice.svg
```

## Keep on top

GNOME on Wayland doesn't let apps pin themselves. Either use `Alt+Space → Always on Top`, or
install the bundled extension once (takes effect after logging out and back in):

```bash
ln -s "$PWD/gnome-extension/voice-on-top@local" ~/.local/share/gnome-shell/extensions/
gnome-extensions enable voice-on-top@local
```
