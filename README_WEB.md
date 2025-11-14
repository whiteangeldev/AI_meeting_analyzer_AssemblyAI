# Live Caption Web Interface

A modern web-based live caption application for real-time meeting transcription with speaker diarization.

## Features

- 🎤 Real-time audio transcription
- 👥 Automatic speaker identification
- 📝 Live caption display with partial and final transcripts
- 🎨 Modern, responsive UI design
- 📊 Real-time statistics (speakers, lines)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have your `ASSEMBLYAI_API_KEY` in a `.env` file:
```
ASSEMBLYAI_API_KEY=your_api_key_here
```

## Running the Web App

Start the Flask server:
```bash
python web_app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Click "Start Recording" to begin transcription
2. Speak into your microphone
3. Watch real-time transcriptions appear with speaker labels
4. Click "Stop Recording" when done

## Architecture

- **Backend**: Flask with Flask-SocketIO for WebSocket communication
- **Frontend**: HTML/CSS/JavaScript with Socket.IO client
- **Audio Processing**: Uses the same audio capture and transcription pipeline as the console app
- **Real-time Updates**: WebSocket events for live transcription streaming

## File Structure

```
├── web_app.py              # Flask application
├── templates/
│   └── index.html          # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css      # Styling
│   └── js/
│       └── app.js         # Frontend JavaScript
└── [existing files]       # Audio processing modules
```

## Notes

- The web app uses the same transcription engine as the console version
- Partial transcripts appear in italic/gray, final transcripts in bold
- Each speaker gets a unique color-coded label
- The interface auto-scrolls to show the latest transcriptions

