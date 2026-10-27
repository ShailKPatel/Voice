#!/bin/bash
# Starts Voice (or focuses it if already running). Pass --toggle to start/stop recording.
cd "$(dirname "$(readlink -f "$0")")"
exec .venv/bin/python voice.py "$@"
