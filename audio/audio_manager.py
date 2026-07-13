# Audio I/O Manager — handles raw mic capture and speaker playback.
# Decouples hardware I/O from STT/TTS processing logic.
# Also handles interrupt: stops playback when VAD detects user speaking (barge-in, stretch goal).
# Implementation: Part 1 (capture) + Part 4 (playback)
