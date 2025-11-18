// Initialize Socket.IO connection
// Explicitly configure to use HTTP (not HTTPS) and disable auto-upgrade
const socket = io({
  transports: ["websocket", "polling"],
  upgrade: true,
  rememberUpgrade: false,
  autoConnect: true,
  reconnection: true,
  reconnectionDelay: 1000,
  reconnectionAttempts: 5,
});

// DOM elements
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const transcriptionArea = document.getElementById("transcriptionArea");
const speakerCountEl = document.getElementById("speakerCount");
const lineCountEl = document.getElementById("lineCount");
const audioModeSelect = document.getElementById("audioMode");

// State
let isRecording = false;
let speakers = new Set();
let lineCount = 0;
let currentPartialLine = null;
let currentActiveBlock = null; // Track the currently active block (being updated)
let activeBlockSpeaker = null; // Track the speaker of the active block

// Socket event handlers
socket.on("connect", () => {
  console.log("Connected to server");
  updateStatus("ready", "Connected");
});

socket.on("connected", (data) => {
  console.log("Server message:", data.status);
  if (data.audio_devices) {
    console.log("Available audio devices:", data.audio_devices);
  }
});

socket.on("audio_devices", (data) => {
  console.log("Available audio devices:", data.devices);
});

socket.on("recording_status", (data) => {
  isRecording = data.is_recording;
  updateUI();
});

socket.on("transcription", (data) => {
  handleTranscription(data);
});

socket.on("error", (data) => {
  console.error("Error:", data.message);
  // Show a more user-friendly error message
  const errorMsg = data.message || "An unknown error occurred";
  
  // Format multi-line error messages better
  const formattedMsg = errorMsg.replace(/\n/g, "\n\n");
  
  alert("Error: " + formattedMsg);
  updateStatus("ready", "Error occurred");
  updateUI();
});

// Handle speaker conversion alerts - show in browser
socket.on("speaker_alert", (data) => {
  const { message, from_speaker, to_speaker, type } = data;
  console.log("🔊 Speaker Alert:", message);
  
  // Create a visible alert banner at the top of the page
  const alertDiv = document.createElement("div");
  alertDiv.style.cssText = `
    position: fixed;
    top: 20px;
    left: 50%;
    transform: translateX(-50%);
    background: ${type === "conversion_confirmed" ? "#4CAF50" : "#FF9800"};
    color: white;
    padding: 15px 25px;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    z-index: 10000;
    font-size: 16px;
    font-weight: bold;
    animation: slideDown 0.3s ease-out;
  `;
  alertDiv.textContent = message;
  
  // Add animation style if not exists
  if (!document.getElementById("speaker-alert-style")) {
    const style = document.createElement("style");
    style.id = "speaker-alert-style";
    style.textContent = `
      @keyframes slideDown {
        from {
          opacity: 0;
          transform: translateX(-50%) translateY(-20px);
        }
        to {
          opacity: 1;
          transform: translateX(-50%) translateY(0);
        }
      }
    `;
    document.head.appendChild(style);
  }
  
  document.body.appendChild(alertDiv);
  
  // Remove alert after 3 seconds
  setTimeout(() => {
    alertDiv.style.animation = "slideDown 0.3s ease-out reverse";
    setTimeout(() => {
      if (alertDiv.parentNode) {
        alertDiv.parentNode.removeChild(alertDiv);
      }
    }, 300);
  }, 3000);
});

// Button event handlers
startBtn.addEventListener("click", () => {
    const audioMode = audioModeSelect.value; // Get selected audio mode
    socket.emit("start_recording", { audio_mode: audioMode });
    updateStatus("recording", "Recording...");
    transcriptionArea.innerHTML = ""; // Clear previous transcriptions
    speakers.clear();
    lineCount = 0;
    currentActiveBlock = null; // Reset active block tracking
    activeBlockSpeaker = null; // Reset active speaker
    currentPartialLine = null; // Reset partial line
    updateStats();
});

stopBtn.addEventListener("click", () => {
  socket.emit("stop_recording");
  updateStatus("ready", "Stopped");
  
  // Remove generic partial line
  if (currentPartialLine) {
    currentPartialLine.remove();
    currentPartialLine = null;
  }
  
  // Finalize any active block (convert grey to green)
  if (currentActiveBlock) {
    currentActiveBlock.classList.remove("partial");
    currentActiveBlock.classList.add("final");
    let timestampEl = currentActiveBlock.querySelector(".timestamp");
    if (!timestampEl) {
      timestampEl = document.createElement("div");
      timestampEl.className = "timestamp";
      currentActiveBlock.appendChild(timestampEl);
    }
    timestampEl.textContent = new Date().toLocaleTimeString();
    currentActiveBlock = null;
    activeBlockSpeaker = null;
  }
});

function updateStatus(state, text) {
  statusText.textContent = text;
  statusDot.className = "dot";
  if (state === "recording") {
    statusDot.classList.add("recording");
  } else if (state === "ready") {
    statusDot.classList.add("ready");
  }
}

function updateUI() {
    startBtn.disabled = isRecording;
    stopBtn.disabled = !isRecording;
    audioModeSelect.disabled = isRecording; // Disable mode selection during recording
}

function handleTranscription(data) {
  const { speaker, text, is_partial, timestamp } = data;
  console.log("Received transcription:", {
    speaker,
    text: text ? text.substring(0, 1000) : "",
    is_partial,
  });

  if (is_partial) {
    // Handle partial transcriptions - these are live updates that should update the active block
    if (speaker === "...") {
      // Generic partial update - handle with "..." placeholder
      if (currentPartialLine) {
        // Update existing partial line
        const textEl = currentPartialLine.querySelector(".transcription-text");
        if (textEl) {
          textEl.textContent = text;
        }
      } else {
        // Create new partial line
        currentPartialLine = createTranscriptionLine(speaker, text, true);
        transcriptionArea.appendChild(currentPartialLine);
      }
    } else if (speaker && text) {
      // Partial update from actual speaker - update the active block in real-time
      if (
        currentActiveBlock &&
        activeBlockSpeaker === speaker &&
        currentActiveBlock.querySelector(".speaker-label").textContent === speaker
      ) {
        // Update existing active block
        const textEl = currentActiveBlock.querySelector(".transcription-text");
        if (textEl) {
          textEl.textContent = text;
        }
      } else {
        // New speaker or new block - create new block
        if (currentPartialLine) {
          currentPartialLine.remove();
          currentPartialLine = null;
        }
        
        // Create new active block for this speaker
        const line = createTranscriptionLine(speaker, text, true);
        transcriptionArea.appendChild(line);
        currentActiveBlock = line;
        activeBlockSpeaker = speaker;
        
        // Track speaker
        if (!speakers.has(speaker)) {
          speakers.add(speaker);
          updateStats();
        }
        
        // Auto-scroll
        transcriptionArea.scrollTop = transcriptionArea.scrollHeight;
      }
    }
  } else {
    // Final transcription - this is a complete finalized block from a speaker
    // Remove partial lines and finalize active block
    if (currentPartialLine) {
      const partialSpeaker = currentPartialLine.querySelector(".speaker-label");
      if (partialSpeaker && partialSpeaker.textContent === "...") {
        currentPartialLine.remove();
        currentPartialLine = null;
      }
    }

    // Final block - convert active block to finalized or create new finalized block
    if (speaker && text) {
      // Check if this finalizes the current active block (same speaker)
      if (
        currentActiveBlock &&
        activeBlockSpeaker === speaker &&
        currentActiveBlock.querySelector(".speaker-label").textContent === speaker
      ) {
        // Finalize the current active block - convert it to a finalized block
        const textEl = currentActiveBlock.querySelector(".transcription-text");
        if (textEl) {
          textEl.textContent = text;
        }
        
        // Add timestamp if not present
        let timestampEl = currentActiveBlock.querySelector(".timestamp");
        if (!timestampEl) {
          timestampEl = document.createElement("div");
          timestampEl.className = "timestamp";
          currentActiveBlock.appendChild(timestampEl);
        }
        timestampEl.textContent = new Date().toLocaleTimeString();
        
        // Remove partial class, add final class
        currentActiveBlock.classList.remove("partial");
        currentActiveBlock.classList.add("final");
        
        // Finalize this block - it's no longer active (will be replaced if same speaker continues)
        currentActiveBlock = null;
        activeBlockSpeaker = null;
      } else {
        // Different speaker or no active block - create a new finalized block
        // Previous active block is already finalized (if any)
        if (currentActiveBlock) {
          // Convert previous active block to finalized
          currentActiveBlock.classList.remove("partial");
          currentActiveBlock.classList.add("final");
          let timestampEl = currentActiveBlock.querySelector(".timestamp");
          if (!timestampEl) {
            timestampEl = document.createElement("div");
            timestampEl.className = "timestamp";
            currentActiveBlock.appendChild(timestampEl);
          }
          timestampEl.textContent = new Date().toLocaleTimeString();
          currentActiveBlock = null;
          activeBlockSpeaker = null;
        }

        // Create new finalized block for this speaker turn
        const line = createTranscriptionLine(speaker, text, false);
        transcriptionArea.appendChild(line);

        // Track speaker
        if (speaker !== "...") {
          speakers.add(speaker);
        }
        lineCount++;
        updateStats();
      }

      // Auto-scroll to bottom
      transcriptionArea.scrollTop = transcriptionArea.scrollHeight;
    }
  }
}

function createTranscriptionLine(speaker, text, isPartial) {
  const line = document.createElement("div");
  line.className = `transcription-line ${isPartial ? "partial" : "final"}`;

  const speakerLabel = document.createElement("span");
  speakerLabel.className = "speaker-label";
  speakerLabel.textContent = speaker || "Unknown";

  const textEl = document.createElement("span");
  textEl.className = "transcription-text";
  textEl.textContent = text;

  line.appendChild(speakerLabel);
  line.appendChild(textEl);

  if (!isPartial) {
    const timestamp = document.createElement("div");
    timestamp.className = "timestamp";
    timestamp.textContent = new Date().toLocaleTimeString();
    line.appendChild(timestamp);
  }

  return line;
}

function updateStats() {
  speakerCountEl.textContent = speakers.size;
  lineCountEl.textContent = lineCount;
}

// Initialize UI
updateUI();
updateStatus("ready", "Ready");
