// Initialize Socket.IO connection
// Explicitly configure to use HTTP (not HTTPS) and disable auto-upgrade
const socket = io({
    transports: ['websocket', 'polling'],
    upgrade: true,
    rememberUpgrade: false,
    autoConnect: true,
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionAttempts: 5
});

// DOM elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const transcriptionArea = document.getElementById('transcriptionArea');
const speakerCountEl = document.getElementById('speakerCount');
const lineCountEl = document.getElementById('lineCount');

// State
let isRecording = false;
let speakers = new Set();
let lineCount = 0;
let currentPartialLine = null;

// Socket event handlers
socket.on('connect', () => {
    console.log('Connected to server');
    updateStatus('ready', 'Connected');
});

socket.on('connected', (data) => {
    console.log('Server message:', data.status);
});

socket.on('recording_status', (data) => {
    isRecording = data.is_recording;
    updateUI();
});

socket.on('transcription', (data) => {
    handleTranscription(data);
});

socket.on('error', (data) => {
    console.error('Error:', data.message);
    alert('Error: ' + data.message);
    updateStatus('ready', 'Error occurred');
    updateUI();
});

// Button event handlers
startBtn.addEventListener('click', () => {
    socket.emit('start_recording');
    updateStatus('recording', 'Recording...');
    transcriptionArea.innerHTML = ''; // Clear previous transcriptions
    speakers.clear();
    lineCount = 0;
    updateStats();
});

stopBtn.addEventListener('click', () => {
    socket.emit('stop_recording');
    updateStatus('ready', 'Stopped');
    if (currentPartialLine) {
        currentPartialLine.remove();
        currentPartialLine = null;
    }
});

function updateStatus(state, text) {
    statusText.textContent = text;
    statusDot.className = 'dot';
    if (state === 'recording') {
        statusDot.classList.add('recording');
    } else if (state === 'ready') {
        statusDot.classList.add('ready');
    }
}

function updateUI() {
    startBtn.disabled = isRecording;
    stopBtn.disabled = !isRecording;
}

function handleTranscription(data) {
    const { speaker, text, is_partial, timestamp } = data;

    if (is_partial) {
        // Update or create partial line
        if (currentPartialLine) {
            const textEl = currentPartialLine.querySelector('.transcription-text');
            textEl.textContent = text;
        } else {
            currentPartialLine = createTranscriptionLine(speaker, text, true);
            transcriptionArea.appendChild(currentPartialLine);
        }
    } else {
        // Remove partial line if exists
        if (currentPartialLine) {
            currentPartialLine.remove();
            currentPartialLine = null;
        }

        // Add final transcription line
        const line = createTranscriptionLine(speaker, text, false);
        transcriptionArea.appendChild(line);
        
        // Track speaker
        if (speaker && speaker !== '...') {
            speakers.add(speaker);
        }
        lineCount++;
        updateStats();

        // Auto-scroll to bottom
        transcriptionArea.scrollTop = transcriptionArea.scrollHeight;
    }
}

function createTranscriptionLine(speaker, text, isPartial) {
    const line = document.createElement('div');
    line.className = `transcription-line ${isPartial ? 'partial' : 'final'}`;

    const speakerLabel = document.createElement('span');
    speakerLabel.className = 'speaker-label';
    speakerLabel.textContent = speaker || 'Unknown';

    const textEl = document.createElement('span');
    textEl.className = 'transcription-text';
    textEl.textContent = text;

    line.appendChild(speakerLabel);
    line.appendChild(textEl);

    if (!isPartial) {
        const timestamp = document.createElement('div');
        timestamp.className = 'timestamp';
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
updateStatus('ready', 'Ready');

