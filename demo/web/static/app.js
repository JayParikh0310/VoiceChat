// Browser mic capture → WebSocket → audio playback
// Implementation: Part 8 (stretch goal)

const STATUS_TEXT = {
  idle: "Click to start",
  listening: "Listening...",
  processing: "Thinking...",
  speaking: "Speaking...",
  error: "Something went wrong",
};

// If no new status arrives this long after "listening", assume the pipeline
// is now processing (STT/guardrails/LLM) — the server has no clean single
// point to announce this without extra hooks into tested Part 7 code, so the
// client infers it instead. See docs/tradeoffs.md.
const PROCESSING_INFER_DELAY_MS = 500;

const button = document.getElementById("session-btn");
const statusEl = document.getElementById("status");
const latencyEl = document.getElementById("latency");

let audioContext = null;
let micStream = null;
let workletNode = null;
let socket = null;
let nextPlaybackTime = 0;
let processingTimer = null;

function setState(state) {
  button.dataset.state = state;
  statusEl.textContent = STATUS_TEXT[state] ?? state;
  button.setAttribute(
    "aria-label",
    state === "idle" ? "Start conversation" : "End conversation"
  );
}

function armProcessingInference() {
  clearTimeout(processingTimer);
  processingTimer = setTimeout(() => setState("processing"), PROCESSING_INFER_DELAY_MS);
}

function playChunk(float32Audio, sampleRate = 24000) {
  const buffer = audioContext.createBuffer(1, float32Audio.length, sampleRate);
  buffer.copyToChannel(float32Audio, 0);
  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(audioContext.destination);
  const startAt = Math.max(audioContext.currentTime, nextPlaybackTime);
  source.start(startAt);
  nextPlaybackTime = startAt + buffer.duration;
}

function handleServerMessage(event) {
  if (typeof event.data === "string") {
    const msg = JSON.parse(event.data);
    if (msg.type === "status") {
      clearTimeout(processingTimer);
      setState(msg.state);
      if (msg.state === "listening") {
        armProcessingInference();
      }
    } else if (msg.type === "latency") {
      const stageText = Object.entries(msg.stages)
        .map(([stage, ms]) => `${stage}: ${ms.toFixed(0)}ms`)
        .join(" | ");
      latencyEl.textContent = `${stageText} | total: ${msg.total_ms.toFixed(0)}ms`;
      latencyEl.hidden = false;
    } else if (msg.type === "error") {
      console.error("Server error:", msg.message);
      setState("error");
      setTimeout(() => {
        if (socket && socket.readyState === WebSocket.OPEN) {
          setState("listening");
          armProcessingInference();
        }
      }, 1500);
    }
    return;
  }

  clearTimeout(processingTimer);
  playChunk(new Float32Array(event.data));
}

async function startSession() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    console.error("Microphone permission denied:", err);
    setState("error");
    return;
  }

  audioContext = new AudioContext();
  nextPlaybackTime = 0;
  await audioContext.audioWorklet.addModule("/static/mic-worklet.js");

  const micSource = audioContext.createMediaStreamSource(micStream);
  workletNode = new AudioWorkletNode(audioContext, "mic-capture-processor", {
    processorOptions: { frameMs: 32 },
  });
  micSource.connect(workletNode);

  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(`${protocol}//${location.host}/ws`);
  socket.binaryType = "arraybuffer";

  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({ type: "session_start", input_sample_rate: audioContext.sampleRate }));
    workletNode.port.onmessage = (event) => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(event.data.buffer);
      }
    };
    setState("listening");
    armProcessingInference();
  });

  socket.addEventListener("message", handleServerMessage);

  socket.addEventListener("close", () => {
    if (button.dataset.state !== "idle") {
      endSession();
    }
  });

  socket.addEventListener("error", (err) => {
    console.error("WebSocket error:", err);
  });
}

function endSession() {
  clearTimeout(processingTimer);
  if (socket) {
    socket.close();
    socket = null;
  }
  if (micStream) {
    micStream.getTracks().forEach((track) => track.stop());
    micStream = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  workletNode = null;
  latencyEl.hidden = true;
  setState("idle");
}

button.addEventListener("click", () => {
  if (button.dataset.state === "idle") {
    setState("listening"); // immediate feedback while mic permission/connection are pending
    startSession();
  } else {
    endSession();
  }
});
