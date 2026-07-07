#!/bin/bash
PROJECT_DIR="/home/crusie/3. Code/Projects/local-speech-to-text-main"
PORT=47891
URL="http://localhost:$PORT"
LOG="/tmp/whisper-stt.log"

# If server is already running, just open browser
if curl -sf --max-time 1 "$URL" > /dev/null 2>&1; then
    xdg-open "$URL"
    exit 0
fi

# Start server detached so it keeps running after this script exits
cd "$PROJECT_DIR"
nohup ".venv/bin/python3" app.py >> "$LOG" 2>&1 &
disown

# Wait until server responds (up to 60s to allow model loading on first run)
for i in $(seq 1 60); do
    sleep 1
    if curl -sf --max-time 1 "$URL" > /dev/null 2>&1; then
        xdg-open "$URL"
        exit 0
    fi
done

exit 1
