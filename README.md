# Re:MEMBER AI Voice Agent

A powerful AI voice agent using Deepgram's Voice Agent API with support for BYO (Bring Your Own) LLM integration. Features ultra-fast voice conversations with real-time STT, LLM, and TTS processing for natural interactions.

## Key Features

- **BYO LLM Support**: Use Groq for ultra-fast inference ($0.07/min) or Deepgram's standard LLM ($0.08/min)
- **Real-time Voice Processing**: <300ms end-to-end latency with Deepgram Nova-3 STT
- **Natural Conversations**: Advanced barge-in handling and conversation flow
- **Multiple Use Cases**:
  - AI sales agent
  - AI Support Agent  
  - AI Cold Calling
  - AI Compliance monitoring

## LLM Configuration Options

### BYO LLM (Groq) - $0.07/minute
- **Model**: meta-llama/llama-4-maverick-17b-128e-instruct
- **Latency**: <200ms token generation
- **Provider**: Groq LPU inference
- **Setup**: Requires Groq API key

### Standard LLM (Deepgram) - $0.08/minute  
- **Model**: gpt-4o-mini
- **Latency**: ~300ms token generation
- **Provider**: Deepgram native
- **Setup**: Only Deepgram API key required

Switch between configurations with `USE_BYO_LLM=true/false` in your `.env` file or `config_defaults.py`.

## Features

- Real-time voice streaming between Twilio and OpenAI
- Automatic speech detection and response cancellation
- Configurable voice settings and system prompts
- Environment-based configuration
- WebSocket-based communication
- Support for G711 ULAW audio format
- Interrupt handling for natural conversation flow
- Silence detection with auto prompt and hangup
- Automatic call hangup after the `end_call` tool or three derailments using Twilio's [`<Hangup>` verb](https://www.twilio.com/docs/voice/twiml/hangup)
- Session management and real-time updates
- Structured JSON responses from GPT-4o using
  `response_format=json` in the WebSocket connection
- Call summaries persisted to SQLite or Postgres
- Call transcripts saved to the `transcripts/` directory, including `call_<id>.txt` plain text logs for QA review

## Prerequisites

Before you begin, ensure you have:

- Python 3.11 (recommended)
- [Poetry](https://python-poetry.org/)
- An OpenAI API key with Realtime API access
- A Twilio account with:
  - Account SID
  - Auth Token
  - Phone Number
- ngrok or similar tool for exposing local server to the internet

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rehan-dev/ai-call-agent.git
cd ai-call-agent
```

2. Install required dependencies using Poetry:
```bash
poetry install
```
# or use pip with the provided requirements file
```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your credentials:
```env
OPENAI_API_KEY=your_openai_api_key
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=your_twilio_phone_number
NGROK_URL=your_ngrok_url
PORT=5050
GOOGLE_CRED_JSON=path_or_json
CALENDAR_ID=your_calendar_id
```
The application automatically loads variables from `.env` using `python-dotenv`.

## Usage

1. Start the server:
```bash
uvicorn main:app --port 5050
```

2. Expose your local server using ngrok:
```bash
ngrok http 5050
```

3. Update your Twilio Voice webhook URL to point to your ngrok URL + `/outgoing-call`

4. Make a call using the API endpoint:
```bash
curl -X POST "http://localhost:5050/make-call" -H "Content-Type: application/json" -d '{"to_phone_number": "+1234567890"}'
```

## Project Structure

```
.
├── main.py              # Main application file
├── prompts/            
│   └── system_prompt.txt # System instructions for AI
├── pyproject.toml       # Project configuration
├── poetry.lock          # Locked dependencies
├── requirements.txt     # Python dependencies
├── .env.example        # Sample environment file
├── .gitignore          # Git ignore file
├── LICENSE             # License file
└── README.md           # Project documentation
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `TWILIO_ACCOUNT_SID`: Your Twilio Account SID
- `TWILIO_AUTH_TOKEN`: Your Twilio Auth Token
- `TWILIO_PHONE_NUMBER`: Your Twilio phone number
- `NGROK_URL`: Your ngrok URL
- `PORT`: Server port (default: 5050)
- `GOOGLE_CRED_JSON`: Path or JSON with Google credentials
- `CALENDAR_ID`: Google Calendar ID used for scheduling
- `DISALLOWED_TOPICS_REGEX`: Regex pattern for topics the agent should avoid
- `DATABASE_URL`: SQLAlchemy database URL (e.g. `sqlite:///./calls.db` or `postgresql://user:pass@host/db`)

### System Prompt

The AI's behavior can be customized by modifying the system prompt in `prompts/system_prompt.txt`.

### Agents Manifest

The `openai.yaml` file defines approved function tools for the OpenAI Agents SDK. Only the
`offer_time_slots`, `schedule_meeting`, and `end_call` actions may be invoked by the model.

## API Endpoints

- `GET /`: Health check endpoint
- `POST /make-call`: Initiate a new call
- `POST /outgoing-call`: Webhook for Twilio voice calls
- `WebSocket /media-stream`: WebSocket endpoint for media streaming
- `POST /offer-time-slots`: Return available slots for today (used by the agent)
- `POST /end-call`: Hang up the current call (used by the agent)

## Demo Mode (No Twilio or Calendar)

You can test the voice experience locally without any phone or calendar setup:

```bash
pip install -r requirements.txt
python demo_mode.py
```

Talk into your microphone and the agent will reply using the OpenAI Realtime API.
Available meeting slots are defined inside `demo_mode.py` and do not require
Google Calendar.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Rehan Khan**
[LinkedIn Profile](https://www.linkedin.com/in/rehankhantht/)

## Acknowledgments

- OpenAI for providing the Realtime API
- Twilio for their excellent voice services
- The open-source community for inspiration and support

## Disclaimer

This project is not officially affiliated with OpenAI or Twilio. Use at your own risk.
# ai-call
