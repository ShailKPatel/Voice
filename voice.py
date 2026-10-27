#!/usr/bin/env python3
"""Voice: small native GTK4 speech-to-text window backed by faster-whisper.

Runs entirely offline. Single instance: launching it again focuses the window,
`voice.py --toggle` starts/stops recording (bind it to a keyboard shortcut).
"""
import json
import os
import queue
import sys
import threading
import time

# Hugging Face's Xet transfer stalls on this machine; plain HTTP downloads fine.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk, Pango  # noqa: E402

import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402

APP_ID = "dev.local.Voice"
SAMPLE_RATE = 16000
RETENTION_DAYS = 30

DATA_DIR = os.path.join(GLib.get_user_data_dir(), "voice")
HISTORY_PATH = os.path.join(DATA_DIR, "history.json")
SETTINGS_PATH = os.path.join(GLib.get_user_config_dir(), "voice", "settings.json")

MODELS = [
    ("small.en", "Fast — English (recommended)"),
    ("large-v3-turbo", "Turbo — most accurate, slower"),
    ("small", "Fast — multilingual"),
]
LANGUAGES = [("en", "English"), ("hi", "Hindi"), ("gu", "Gujarati"), ("de", "German")]
DEFAULT_SETTINGS = {"model": "small.en", "language": "en", "auto_copy": True}

CSS = """
.record {
  min-width: 56px; min-height: 56px; border-radius: 999px; padding: 0;
  background: alpha(currentColor, 0.08);
}
.record.recording { background: #e01b24; color: white; animation: pulse 1.4s ease-in-out infinite; }
.record.busy { background: alpha(@accent_bg_color, 0.25); }
@keyframes pulse {
  0%   { box-shadow: 0 0 0 0 alpha(#e01b24, 0.55); }
  70%  { box-shadow: 0 0 0 12px alpha(#e01b24, 0); }
  100% { box-shadow: 0 0 0 0 alpha(#e01b24, 0); }
}
.timer { font-size: 22px; font-weight: 700; font-feature-settings: "tnum"; }
.timer.recording { color: #e01b24; }
.transcript { padding: 10px; border-radius: 10px; background: alpha(currentColor, 0.05); }
.transcript text { background: transparent; }
levelbar trough { min-height: 4px; }
.hist-text { font-size: 13px; }
"""


# ── persistence ────────────────────────────────────────────────────────────

def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def load_history():
    """Load history and drop anything older than RETENTION_DAYS (runs at startup)."""
    items = load_json(HISTORY_PATH, [])
    cutoff = time.time() - RETENTION_DAYS * 86400
    kept = [h for h in items if h.get("created", 0) >= cutoff]
    if len(kept) != len(items):
        save_json(HISTORY_PATH, kept)
    return kept


# ── audio + model ──────────────────────────────────────────────────────────

class Recorder:
    """Captures mic audio and reports natural pauses so speech can be transcribed while recording."""

    BLOCK = 1600  # 100 ms

    def __init__(self):
        self.stream = None
        self.chunks = []
        self.rms = []
        self.cut = 0  # index into chunks already handed off for transcription
        self.level = 0.0

    def _callback(self, indata, frames, time_info, status):
        mono = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(mono ** 2)))
        self.chunks.append(mono)
        self.rms.append(rms)
        self.level = min(1.0, rms * 12)

    def start(self):
        self.chunks, self.rms, self.cut = [], [], 0
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                     blocksize=self.BLOCK, callback=self._callback)
        self.stream.start()

    def take_phrase(self, min_secs=6.0, pause_secs=0.6):
        """Return audio up to the latest pause if enough new speech has accumulated, else None."""
        n = len(self.rms)
        pause_blocks = int(pause_secs * SAMPLE_RATE / self.BLOCK)
        if (n - self.cut) * self.BLOCK < min_secs * SAMPLE_RATE or n < pause_blocks:
            return None
        floor = float(np.percentile(self.rms[:n], 15))
        if max(self.rms[n - pause_blocks:n]) > floor * 2.5 + 0.004:
            return None
        audio = np.concatenate(self.chunks[self.cut:n])
        self.cut = n
        return audio

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.level = 0.0
        rest = self.chunks[self.cut:]
        return np.concatenate(rest) if rest else np.zeros(0, np.float32)


class Transcriber:
    """Owns the Whisper model; all work happens on one background thread."""

    PROMPT = "Hello, this is a dictated message, with proper punctuation."

    def __init__(self, on_event):
        self.on_event = on_event  # called on the GTK main loop
        self.jobs = queue.Queue()
        self.model = None
        self.model_name = None
        self.wanted_model = DEFAULT_SETTINGS["model"]
        self.session = 0  # bump to drop queued work from a cancelled recording
        self.prev_text = ""
        threading.Thread(target=self._run, daemon=True).start()

    def _emit(self, kind, payload):
        def deliver():
            self.on_event(kind, payload)
            return GLib.SOURCE_REMOVE
        GLib.idle_add(deliver)

    def load(self, name):
        self.jobs.put(("load", name, None))

    def transcribe(self, audio, language, tag):
        """tag = (session, part index); results come back as a "part" event."""
        self.jobs.put(("transcribe", audio, (language, tag)))

    def _ensure(self, name):
        if self.model_name == name:
            return
        self._emit("loading", name)
        from faster_whisper import WhisperModel
        self.model = None
        options = dict(device="cpu", compute_type="int8",
                       # Physical cores only: hyperthreads make CTranslate2 ~2x slower here.
                       cpu_threads=max(1, (os.cpu_count() or 8) // 2))
        try:
            # Cached models load straight from disk, skipping the Hugging Face round trip.
            self.model = WhisperModel(name, local_files_only=True, **options)
        except Exception:
            self._emit("downloading", name)
            self.model = WhisperModel(name, **options)
        self.model_name = name
        self._emit("ready", name)

    def _run(self):
        while True:
            kind, a, b = self.jobs.get()
            try:
                if kind == "load":
                    self._ensure(a)
                    continue
                language, (session, index) = b
                if session != self.session:
                    continue
                if index == 0:
                    self.prev_text = ""
                model = self.wanted_model
                if language != "en" and model.endswith(".en"):
                    model = model[:-3]  # English-only models can't do Hindi/Gujarati/German
                self._ensure(model)
                started = time.monotonic()
                segments, info = self.model.transcribe(
                    a,
                    language=language,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 700},
                    condition_on_previous_text=False,
                    # Previous phrase keeps names/casing consistent across chunks.
                    initial_prompt=(self.PROMPT + " " + self.prev_text[-200:]).strip(),
                )
                text = " ".join(s.text.strip() for s in segments).strip()
                if text:
                    self.prev_text = text
                self._emit("part", {
                    "session": session,
                    "index": index,
                    "text": text,
                    "lang": info.language,
                    "took": time.monotonic() - started,
                })
            except Exception as e:  # surface every failure in the UI instead of dying
                self._emit("error", str(e))


# ── UI ─────────────────────────────────────────────────────────────────────

class VoiceWindow(Adw.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app, title="Voice", default_width=380, default_height=300)
        self.app = app
        self.settings = {**DEFAULT_SETTINGS, **load_json(SETTINGS_PATH, {})}
        if self.settings["language"] not in dict(LANGUAGES):  # dropped option from an older version
            self.settings["language"] = DEFAULT_SETTINGS["language"]
        self.history = load_history()
        self.recorder = Recorder()
        self.rec_started = None
        self.busy = False
        self.model_ready = False
        self.parts, self.part_count, self.expected = {}, 0, None
        self.base_text = ""

        self.transcriber = Transcriber(self.on_transcriber_event)
        self.transcriber.wanted_model = self.settings["model"]
        self.transcriber.load(self.settings["model"])

        self._build()
        self._install_actions()
        self.refresh_history()
        GLib.timeout_add(80, self._tick)

    # layout
    def _build(self):
        # No title, no minimize/maximize: just the history toggle and close.
        header = Adw.HeaderBar(show_title=False, decoration_layout=":close")
        self.hist_btn = Gtk.ToggleButton(icon_name="document-open-recent-symbolic",
                                         tooltip_text="History & settings (Ctrl+H)")
        self.hist_btn.connect("toggled", lambda b: self.stack.set_visible_child_name(
            "history" if b.get_active() else "main"))
        header.pack_start(self.hist_btn)

        self.stack = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE)
        self.stack.add_named(self._build_main(), "main")
        self.stack.add_named(self._build_history(), "history")

        view = Adw.ToolbarView()
        view.add_top_bar(header)
        self.toasts = Adw.ToastOverlay(child=self.stack)
        view.set_content(self.toasts)
        self.set_content(view)

    def _build_main(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10,
                      margin_start=12, margin_end=12, margin_bottom=12, margin_top=4)

        top = Gtk.Box(spacing=12)
        self.rec_btn = Gtk.Button(tooltip_text="Record / stop (Ctrl+R)", valign=Gtk.Align.CENTER)
        self.rec_btn.add_css_class("record")
        self.rec_icon = Gtk.Image(icon_name="audio-input-microphone-symbolic", pixel_size=24)
        self.rec_btn.set_child(self.rec_icon)
        self.rec_btn.connect("clicked", lambda *_: self.toggle_recording())
        top.append(self.rec_btn)

        info = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4, hexpand=True,
                       valign=Gtk.Align.CENTER)
        row = Gtk.Box(spacing=8)
        self.timer = Gtk.Label(label="0:00", xalign=0)
        self.timer.add_css_class("timer")
        row.append(self.timer)
        self.spinner = Adw.Spinner(visible=False)
        row.append(self.spinner)
        info.append(row)
        self.status = Gtk.Label(label="Loading model…", xalign=0, ellipsize=Pango.EllipsizeMode.END)
        self.status.add_css_class("dim-label")
        self.status.add_css_class("caption")
        info.append(self.status)
        self.level = Gtk.LevelBar(min_value=0, max_value=1, visible=False)
        info.append(self.level)
        top.append(info)

        self.cancel_btn = Gtk.Button(icon_name="edit-delete-symbolic", visible=False,
                                     tooltip_text="Discard recording (Esc)", valign=Gtk.Align.CENTER)
        self.cancel_btn.add_css_class("flat")
        self.cancel_btn.connect("clicked", lambda *_: self.cancel_recording())
        top.append(self.cancel_btn)
        box.append(top)

        self.textview = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.textview.add_css_class("transcript")
        scroller = Gtk.ScrolledWindow(child=self.textview, vexpand=True, min_content_height=80)
        box.append(scroller)

        actions = Gtk.Box(spacing=6, halign=Gtk.Align.END)
        clear = Gtk.Button(label="Clear")
        clear.add_css_class("flat")
        clear.connect("clicked", lambda *_: self.clear())
        self.copy_btn = Gtk.Button(label="Copy", sensitive=False)
        self.copy_btn.add_css_class("suggested-action")
        self.copy_btn.connect("clicked", lambda *_: self.copy(self._buffer_text()))
        self.textview.get_buffer().connect(
            "changed", lambda *_: self.copy_btn.set_sensitive(bool(self._buffer_text().strip())))
        actions.append(clear)
        actions.append(self.copy_btn)
        box.append(actions)
        return box

    def _build_history(self):
        # Settings, search and list scroll together so the list isn't squeezed
        # into whatever height is left over; only the footer stays pinned.
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8,
                      margin_start=12, margin_end=12, margin_bottom=8, margin_top=4)
        page.append(Gtk.ScrolledWindow(child=box, vexpand=True,
                                       hscrollbar_policy=Gtk.PolicyType.NEVER))
        box.append(self._build_settings())
        self.search = Gtk.SearchEntry(placeholder_text="Search history")
        self.search.connect("search-changed", lambda *_: self.refresh_history())
        box.append(self.search)

        self.hist_list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        self.hist_list.add_css_class("boxed-list")
        self.hist_list.connect("row-activated", self._on_history_row)
        self.hist_empty = Adw.StatusPage(icon_name="document-open-recent-symbolic",
                                         title="No recordings yet")
        self.hist_empty.add_css_class("compact")
        self.hist_stack = Gtk.Stack()
        self.hist_stack.add_named(self.hist_list, "list")
        self.hist_stack.add_named(self.hist_empty, "empty")
        box.append(self.hist_stack)

        footer = Gtk.Box(spacing=6, margin_start=12, margin_end=12, margin_bottom=12)
        self.hist_count = Gtk.Label(xalign=0, hexpand=True)
        self.hist_count.add_css_class("dim-label")
        self.hist_count.add_css_class("caption")
        clear_all = Gtk.Button(label="Clear all")
        clear_all.add_css_class("flat")
        clear_all.connect("clicked", lambda *_: self._clear_history())
        footer.append(self.hist_count)
        footer.append(clear_all)
        page.append(footer)
        return page

    def _build_settings(self):
        group = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        group.add_css_class("boxed-list")
        group.append(self._combo_row("Model", MODELS, "model"))
        group.append(self._combo_row("Language", LANGUAGES, "language"))
        auto = Adw.SwitchRow(title="Auto-copy when done", active=self.settings["auto_copy"])
        auto.connect("notify::active", lambda r, _: self._set_setting("auto_copy", r.get_active()))
        group.append(auto)
        return group

    def _combo_row(self, title, options, key):
        keys = [k for k, _ in options]
        row = Adw.ComboRow(title=title, model=Gtk.StringList.new([label for _, label in options]))
        if self.settings[key] in keys:
            row.set_selected(keys.index(self.settings[key]))
        row.connect("notify::selected", lambda r, _: self._set_setting(key, keys[r.get_selected()]))
        return row

    def _install_actions(self):
        keys = Gtk.EventControllerKey()
        keys.connect("key-pressed", self._on_key)
        self.add_controller(keys)

    # settings
    def _set_setting(self, key, value):
        if self.settings[key] == value:
            return
        self.settings[key] = value
        save_json(SETTINGS_PATH, self.settings)
        if key == "model":
            self.transcriber.wanted_model = value
            self.transcriber.load(value)

    def _on_key(self, _ctrl, keyval, _code, state):
        ctrl = state & Gdk.ModifierType.CONTROL_MASK
        if ctrl and keyval in (Gdk.KEY_r, Gdk.KEY_space):
            self.toggle_recording()
            return True
        if ctrl and keyval == Gdk.KEY_h:
            self.hist_btn.set_active(not self.hist_btn.get_active())
            return True
        if keyval == Gdk.KEY_Escape and self.rec_started:
            self.cancel_recording()
            return True
        return False

    # recording
    def toggle_recording(self):
        if self.rec_started:
            self.stop_recording()
        elif not self.busy:
            self.start_recording()

    def start_recording(self):
        try:
            self.recorder.start()
        except Exception as e:
            self.set_status(f"Mic error: {e}")
            return
        self.transcriber.session += 1
        self.parts, self.part_count, self.expected = {}, 0, None
        self.base_text = self._buffer_text().strip()
        self.rec_started = time.monotonic()
        self.hist_btn.set_active(False)
        self.rec_btn.add_css_class("recording")
        self.timer.add_css_class("recording")
        self.rec_icon.set_from_icon_name("media-playback-stop-symbolic")
        self.level.set_visible(True)
        self.cancel_btn.set_visible(True)
        self.set_status("Listening… click again to stop")

    def _reset_record_ui(self):
        self.rec_started = None
        self.rec_btn.remove_css_class("recording")
        self.timer.remove_css_class("recording")
        self.rec_icon.set_from_icon_name("audio-input-microphone-symbolic")
        self.level.set_visible(False)
        self.cancel_btn.set_visible(False)

    def cancel_recording(self):
        if not self.rec_started:
            return
        self.recorder.stop()
        self.transcriber.session += 1
        self._reset_record_ui()
        self.textview.get_buffer().set_text(self.base_text)
        self.timer.set_label("0:00")
        self.set_status("Discarded")

    def _submit(self, audio):
        tag = (self.transcriber.session, self.part_count)
        self.transcriber.transcribe(audio, self.settings["language"], tag)
        self.part_count += 1

    def stop_recording(self):
        if not self.rec_started or self.busy:
            return
        self.rec_dur = round(time.monotonic() - self.rec_started, 1)
        self._reset_record_ui()
        self.set_busy(True)
        self.set_status("Finishing…")
        # Keep the mic open briefly so the last word isn't clipped.
        GLib.timeout_add(350, self._finish_recording)

    def _finish_recording(self):
        tail = self.recorder.stop()
        if len(tail) >= SAMPLE_RATE * 0.4:
            self._submit(tail)
        if self.part_count == 0:
            self.set_busy(False)
            self.set_status("Too short — nothing recorded")
            return GLib.SOURCE_REMOVE
        self.expected = self.part_count
        self.stopped_at = time.monotonic()
        if not self.model_ready:
            self.set_status("Waiting for model, then transcribing…")
        self._maybe_finish()
        return GLib.SOURCE_REMOVE

    def set_busy(self, busy):
        self.busy = busy
        self.spinner.set_visible(busy)
        self.rec_btn.set_sensitive(not busy)
        (self.rec_btn.add_css_class if busy else self.rec_btn.remove_css_class)("busy")

    def set_status(self, text):
        self.status.set_label(text)

    def _tick(self):
        if self.rec_started:
            secs = int(time.monotonic() - self.rec_started)
            self.timer.set_label(f"{secs // 60}:{secs % 60:02d}")
            self.level.set_value(self.recorder.level)
            # Whisper encodes 30s windows whatever the input length, so slow models get longer chunks.
            min_secs = 20 if "large" in self.settings["model"] else 6
            phrase = self.recorder.take_phrase(min_secs)
            if phrase is not None:
                self._submit(phrase)
        return True

    def _joined_parts(self):
        return " ".join(self.parts[i] for i in sorted(self.parts) if self.parts[i])

    # transcriber events (main loop)
    def on_transcriber_event(self, kind, payload):
        if kind == "loading":
            self.model_ready = False
            self.set_status(f"Loading {payload}…")
        elif kind == "downloading":
            self.set_status(f"Downloading {payload} (first use only)…")
        elif kind == "ready":
            self.model_ready = True
            if not self.busy and not self.rec_started:
                self.set_status(f"Ready · {payload} · Ctrl+R to record")
        elif kind == "error":
            self.expected = None
            self.set_busy(False)
            self.set_status(f"Error: {payload}")
        elif kind == "part":
            if payload["session"] != self.transcriber.session:
                return
            self.parts[payload["index"]] = payload["text"]
            self.lang = payload["lang"]
            live = self._joined_parts()
            if live:
                self.textview.get_buffer().set_text(
                    f"{self.base_text}\n\n{live}" if self.base_text else live)
            self._maybe_finish()

    def _maybe_finish(self):
        if self.expected is None or len(self.parts) < self.expected:
            return
        self.expected = None
        self.set_busy(False)
        text = self._joined_parts()
        if not text:
            self.set_status("No speech detected")
            return
        self.set_status(f"Done in {time.monotonic() - self.stopped_at:.1f}s after stop")
        entry = {"created": time.time(), "model": self.settings["model"], "text": text,
                 "lang": self.lang, "dur": self.rec_dur}
        self.history.insert(0, entry)
        save_json(HISTORY_PATH, self.history)
        self.refresh_history()
        if self.settings["auto_copy"]:
            self.copy(text, "Copied to clipboard")
        if not self.is_active():
            n = Gio.Notification.new("Transcription ready")
            n.set_body(text[:120])
            self.app.send_notification("done", n)

    # helpers
    def _buffer_text(self):
        buf = self.textview.get_buffer()
        return buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)

    def clear(self):
        """Reset the main view: transcript, timer and status line."""
        self.textview.get_buffer().set_text("")
        if self.rec_started or self.busy:
            return
        self.timer.set_label("0:00")
        self.set_status(f"Ready · {self.settings['model']} · Ctrl+R to record"
                        if self.model_ready else "Loading model…")

    def copy(self, text, toast="Copied"):
        if not text.strip():
            return
        self.get_clipboard().set(text)
        self.toasts.add_toast(Adw.Toast(title=toast, timeout=2))

    def refresh_history(self):
        q = self.search.get_text().lower().strip()
        self.hist_list.remove_all()
        shown = 0
        for h in self.history:
            if q and q not in h["text"].lower():
                continue
            shown += 1
            row = Gtk.ListBoxRow(activatable=True)
            row.entry = h
            vb = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2,
                         margin_start=10, margin_end=10, margin_top=8, margin_bottom=8)
            t = Gtk.Label(label=h["text"], xalign=0, wrap=True, lines=2,
                          ellipsize=Pango.EllipsizeMode.END)
            t.add_css_class("hist-text")
            when = GLib.DateTime.new_from_unix_local(int(h["created"])).format("%d %b %H:%M")
            meta = Gtk.Label(label=f"{when} · took {h.get('took', 0)}s", xalign=0)
            meta.add_css_class("dim-label")
            meta.add_css_class("caption")
            vb.append(t)
            vb.append(meta)
            row.set_child(vb)
            row.set_tooltip_text("Click to copy")
            self.hist_list.append(row)
        self.hist_stack.set_visible_child_name("list" if shown else "empty")
        self.hist_count.set_label(f"{len(self.history)} recordings · auto-deleted after {RETENTION_DAYS} days")

    def _on_history_row(self, _list, row):
        self.textview.get_buffer().set_text(row.entry["text"])
        self.copy(row.entry["text"])

    def _clear_history(self):
        dialog = Adw.AlertDialog(heading="Delete all history?", body="This cannot be undone.")
        dialog.add_response("cancel", "Cancel")
        dialog.add_response("delete", "Delete")
        dialog.set_response_appearance("delete", Adw.ResponseAppearance.DESTRUCTIVE)

        def on_response(_d, response):
            if response == "delete":
                self.history = []
                save_json(HISTORY_PATH, self.history)
                self.refresh_history()
        dialog.connect("response", on_response)
        dialog.present(self)


class VoiceApp(Adw.Application):
    def __init__(self):
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.HANDLES_COMMAND_LINE)
        self.add_main_option("toggle", 0, GLib.OptionFlags.NONE, GLib.OptionArg.NONE,
                             "Start or stop recording", None)
        self.window = None

    def do_startup(self):
        Adw.Application.do_startup(self)
        provider = Gtk.CssProvider()
        provider.load_from_string(CSS)
        Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), provider,
                                                  Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        icon_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        Gtk.IconTheme.get_for_display(Gdk.Display.get_default()).add_search_path(icon_dir)
        Gtk.Window.set_default_icon_name(APP_ID)

    def do_command_line(self, cmd):
        if self.window is None:
            self.window = VoiceWindow(self)
        self.window.present()
        if cmd.get_options_dict().contains("toggle"):
            self.window.toggle_recording()
        return 0


if __name__ == "__main__":
    sys.exit(VoiceApp().run(sys.argv))
