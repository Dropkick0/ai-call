import time
import os
import json
import base64
import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone

from guardrails.validator_base import register_validator, Validator
from guardrails.classes.validation.validation_result import PassResult, FailResult
from guardrails.guard import Guard
from pydantic import BaseModel, Field

import structlog
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect
from dotenv import load_dotenv
import gcal
from db import init_db, save_call_summary
from metrics import compute_call_metrics, write_report

load_dotenv()

logging.basicConfig(format="%(message)s", level=logging.INFO)
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()

# Global variable to track the active call SID
CURRENT_CALL_ID = None
# Track metrics for the current call
CURRENT_METRICS = None


@register_validator("intent_whitelist", data_type="string")
class IntentWhitelist(Validator):
    """Validate that intent is one of the allowed choices."""

    def __init__(self, intents, **kwargs):
        super().__init__(**kwargs)
        self.intents = set(intents)

    def _validate(self, value, metadata):
        if isinstance(value, str) and value in self.intents:
            return PassResult()
        return FailResult(error_message=f"Intent '{value}' not allowed")


class LLMOutput(BaseModel):
    intent: str = Field(
        json_schema_extra={
            "validators": [IntentWhitelist(["greeting", "ask_date"])]
        }
    )
    text: str


intent_guard = Guard.for_pydantic(LLMOutput)


def contains_disallowed_topic(text: str) -> bool:
    """Return True if text matches the disallowed topics regex."""
    if DISALLOWED_TOPICS_PATTERN and text:
        return bool(DISALLOWED_TOPICS_PATTERN.search(text))
    return False


def load_prompt(file_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    prompt_path = os.path.join(dir_path, "prompts", f"{file_name}.txt")

    try:
        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        logger.error("prompt.load_failed", path=prompt_path)
        raise


# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # requires OpenAI Realtime API Access
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
NGROK_URL = os.getenv("NGROK_URL")
PORT = int(os.getenv("PORT", 5050))
GOOGLE_CRED_JSON = os.getenv("GOOGLE_CRED_JSON")
CALENDAR_ID = os.getenv("CALENDAR_ID")
DISALLOWED_TOPICS_REGEX = os.getenv("DISALLOWED_TOPICS_REGEX")

if DISALLOWED_TOPICS_REGEX:
    DISALLOWED_TOPICS_PATTERN = re.compile(DISALLOWED_TOPICS_REGEX, re.IGNORECASE)
else:
    DISALLOWED_TOPICS_PATTERN = None

SYSTEM_MESSAGE = load_prompt("system_prompt")


def get_todays_free_slots():
    """Return formatted free time slots for today."""
    if not CALENDAR_ID:
        raise ValueError("CALENDAR_ID environment variable not set")
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    global CURRENT_METRICS
    try:
        slots = gcal.list_free_slots(CALENDAR_ID, start, end)
    except Exception as exc:
        logger.error("slots.fetch_failed", error=str(exc))
        if CURRENT_METRICS is not None:
            CURRENT_METRICS["calendar_errors"] += 1
        return []
    return [f"{s[0].strftime('%I:%M %p')} - {s[1].strftime('%I:%M %p')}" for s in slots]
VOICE = "shimmer"
LOG_EVENT_TYPES = [
    "response.content.done",
    "rate_limits.updated",
    "response.done",
    "input_audio_buffer.committed",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.speech_started",
    "session.created",
]

app = FastAPI()
init_db()

if not OPENAI_API_KEY:
    raise ValueError("Missing the OpenAI API key. Please set it in the .env file.")

if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_PHONE_NUMBER:
    raise ValueError("Missing Twilio configuration. Please set it in the .env file.")


@app.get("/", response_class=HTMLResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}


@app.post("/make-call")
async def make_call(to_phone_number: str):
    """Make an outgoing call to the specified phone number."""
    if not to_phone_number:
        return {"error": "Phone number is required"}
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        call = client.calls.create(
            url=f"{NGROK_URL}/outgoing-call",
            to=to_phone_number,
            from_=TWILIO_PHONE_NUMBER,
        )
        start_ts = datetime.utcnow().isoformat()
        logger.info(
            "call.initiated",
            call_id=call.sid,
            start_time=start_ts,
            to=to_phone_number,
        )
    except Exception as e:
        logger.error("call.initiation_failed", error=str(e))

    return {"call_sid": call.sid}


@app.post("/offer-time-slots")
async def offer_time_slots(prospect_name: str):
    """Return free slots for today."""
    slots = get_todays_free_slots()
    return {"prospect_name": prospect_name, "time_slots": slots}


@app.post("/schedule-meeting")
async def schedule_meeting(prospect_name: str, time_slot: str, email: str):
    """Create a calendar event using the chosen time slot."""
    if not CALENDAR_ID:
        raise ValueError("CALENDAR_ID environment variable not set")

    global CURRENT_METRICS
    try:
        start_str, end_str = [s.strip() for s in time_slot.split("-")]
        today = datetime.now(timezone.utc)
        start_dt = datetime.strptime(start_str, "%I:%M %p")
        end_dt = datetime.strptime(end_str, "%I:%M %p")
        start = today.replace(hour=start_dt.hour, minute=start_dt.minute, second=0, microsecond=0)
        end = today.replace(hour=end_dt.hour, minute=end_dt.minute, second=0, microsecond=0)
    except Exception as exc:
        logger.error("schedule.parse_failed", time_slot=time_slot, error=str(exc))
        if CURRENT_METRICS is not None:
            CURRENT_METRICS["calendar_errors"] += 1
        return {"error": "Invalid time slot"}

    try:
        event = gcal.create_event(
            CALENDAR_ID,
            start,
            end,
            f"Call with {prospect_name}"
        )
        logger.info("schedule.created", event_id=event.get("id"))
    except Exception as exc:
        logger.error("schedule.failed", error=str(exc))
        if CURRENT_METRICS is not None:
            CURRENT_METRICS["calendar_errors"] += 1
        return {"error": "Failed to schedule meeting"}

    return {"status": "scheduled", "event_id": event.get("id"), "email": email}


@app.post("/end-call")
async def end_call():
    """Hang up the active call using Twilio's <Hangup> verb."""
    global CURRENT_CALL_ID
    if not CURRENT_CALL_ID:
        return {"error": "No active call"}
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        vr = VoiceResponse()
        vr.hangup()
        client.calls(CURRENT_CALL_ID).update(twiml=str(vr))
        logger.info("call.hangup", call_id=CURRENT_CALL_ID)
        return {"status": "hangup", "call_id": CURRENT_CALL_ID}
    except Exception as exc:
        logger.error("hangup.failed", call_id=CURRENT_CALL_ID, error=str(exc))
        return {"error": "Failed to hang up"}


@app.api_route("/outgoing-call", methods=["GET", "POST"])
async def handle_outgoing_call(request: Request):
    """Handle outgoing call and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    response.say("This calls may be recorded for compliance purposes")
    response.pause(length=1)
    response.say("Connecting with Compliance Agent")
    connect = Connect()
    connect.stream(
        url=f"wss://{request.url.hostname}/media-stream",
        media_stream_timeout="5",
    )
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI."""
    logger.info("client.connected")
    await websocket.accept()

    async with websockets.connect(
        "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01&response_format=json",
        extra_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        },
    ) as openai_ws:
        session = {"state": "awaiting_greeting"}
        await send_session_update(openai_ws, session)
        stream_sid = None
        session_id = None
        call_id = None
        start_ts = None
        transcripts = []
        silence_count = 0
        derailment_count = 0
        guardrail_rejects = 0
        latencies = []
        speech_start_time = None
        global CURRENT_METRICS
        CURRENT_METRICS = {
            "guardrail_rejects": 0,
            "calendar_errors": 0,
            "latencies": [],
        }

        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
            nonlocal stream_sid, call_id, start_ts
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data["event"] == "media" and openai_ws.open:
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data["media"]["payload"],
                        }
                        await openai_ws.send(json.dumps(audio_append))
                    elif data["event"] == "start":
                        stream_sid = data["start"]["streamSid"]
                        call_id = data["start"].get("callSid")
                        start_ts = datetime.utcnow().isoformat()
                        global CURRENT_CALL_ID
                        CURRENT_CALL_ID = call_id
                        logger.info(
                            "stream.started",
                            call_id=call_id,
                            stream_sid=stream_sid,
                            start_time=start_ts,
                        )
                    elif data["event"] == "dtmf":
                        digits = (
                            data.get("dtmf", {}).get("digits")
                            or data.get("digits")
                            or data.get("dtmf")
                        )
                        if digits:
                            session.setdefault("digits", "")
                            session["digits"] += digits
                            logger.info(
                                "digits.received", call_id=call_id, digits=digits
                            )
                    elif data["event"] == "media_stream_timeout":
                        logger.info("silence.detected", call_id=call_id)
                        silence_count += 1
                        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                        if silence_count == 1:
                            vr = VoiceResponse()
                            vr.say("I didn't catch that. Are you still there?")
                            client.calls(call_id).update(twiml=str(vr))
                        else:
                            logger.info("silence.hangup", call_id=call_id)
                            client.calls(call_id).update(status="completed")
                            if openai_ws.open:
                                await openai_ws.close()
                            await websocket.close()
                            break
            except WebSocketDisconnect:
                logger.info("client.disconnected", call_id=call_id)
                if openai_ws.open:
                    await openai_ws.close()

        async def send_to_twilio():
            """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
            nonlocal stream_sid, session_id, transcripts, session, derailment_count
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)
                    if response["type"] in LOG_EVENT_TYPES:
                        logger.info(
                            "openai.event",
                            call_id=call_id,
                            event=response["type"],
                            payload=response,
                        )
                    if response["type"] == "session.created":
                        session_id = response["session"]["id"]
                    if response["type"] == "session.updated":
                        logger.info("session.updated", call_id=call_id)
                    if response["type"] == "response.audio.delta" and response.get(
                        "delta"
                    ):
                        try:
                            audio_payload = base64.b64encode(
                                base64.b64decode(response["delta"])
                            ).decode("utf-8")
                            audio_delta = {
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {"payload": audio_payload},
                            }
                            await websocket.send_json(audio_delta)
                        except Exception as e:
                            logger.error(
                                "audio.process_error", call_id=call_id, error=str(e)
                            )
                    if response["type"] == "conversation.item.created":
                        transcripts.append(response)
                        logger.info(
                            "conversation.item", call_id=call_id, item=response
                        )
                        content = None
                        role = None
                        if isinstance(response.get("message"), dict):
                            role = response["message"].get("role")
                            content = response["message"].get("content")
                        elif "content" in response:
                            content = response.get("content")
                            role = response.get("role") or response.get("speaker")
                        if role == "assistant" and speech_start_time is not None:
                            latency = time.monotonic() - speech_start_time
                            latencies.append(latency)
                            if CURRENT_METRICS is not None:
                                CURRENT_METRICS["latencies"].append(latency)
                            speech_start_time = None
                        if content:
                            if contains_disallowed_topic(content):
                                logger.warning(
                                    "topic.disallowed",
                                    call_id=call_id,
                                    content=content,
                                )
                                derailment_count += 1
                                guardrail_rejects += 1
                                if CURRENT_METRICS is not None:
                                    CURRENT_METRICS["guardrail_rejects"] += 1
                                if openai_ws.open:
                                    await openai_ws.send(
                                        json.dumps({"type": "response.cancel"})
                                    )
                                if derailment_count >= 3:
                                    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                                    vr = VoiceResponse()
                                    vr.hangup()
                                    client.calls(call_id).update(twiml=str(vr))
                                    if openai_ws.open:
                                        await openai_ws.close()
                                    await websocket.close()
                                    break
                                continue
                            intent = None
                            try:
                                llm_output = intent_guard.parse(
                                    content if isinstance(content, str) else json.dumps(content)
                                )
                                intent = llm_output.intent
                            except Exception as exc:
                                logger.warning(
                                    "intent.validation_failed",
                                    call_id=call_id,
                                    error=str(exc),
                                    content=content,
                                )
                                guardrail_rejects += 1
                                if CURRENT_METRICS is not None:
                                    CURRENT_METRICS["guardrail_rejects"] += 1
                            if intent:
                                if session["state"] == "awaiting_greeting":
                                    if intent == "greeting":
                                        session["state"] = "awaiting_date"
                                        await send_session_update(openai_ws, session)
                                    else:
                                        logger.warning(
                                            "state.violation",
                                            call_id=call_id,
                                            state=session["state"],
                                            intent=intent,
                                        )
                                        derailment_count += 1
                                        if derailment_count >= 3:
                                            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                                            vr = VoiceResponse()
                                            vr.hangup()
                                            client.calls(call_id).update(twiml=str(vr))
                                            if openai_ws.open:
                                                await openai_ws.close()
                                            await websocket.close()
                                            break
                                elif session["state"] == "awaiting_date":
                                    if intent == "ask_date":
                                        session["state"] = "complete"
                                        await send_session_update(openai_ws, session)
                                    elif intent != "greeting":
                                        logger.warning(
                                            "state.violation",
                                            call_id=call_id,
                                            state=session["state"],
                                            intent=intent,
                                        )
                                        derailment_count += 1
                                        if derailment_count >= 3:
                                            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                                            vr = VoiceResponse()
                                            vr.hangup()
                                            client.calls(call_id).update(twiml=str(vr))
                                            if openai_ws.open:
                                                await openai_ws.close()
                                            await websocket.close()
                                            break
                    if response["type"] == "input_audio_buffer.speech_started":
                        logger.info("speech.start", call_id=call_id)
                        speech_start_time = time.monotonic()

                        # Send clear event to Twilio
                        await websocket.send_json({"streamSid": stream_sid, "event": "clear"})

                        logger.info("speech.cancel", call_id=call_id)

                        # Send cancel message to OpenAI
                        interrupt_message = {"type": "response.cancel"}
                        await openai_ws.send(json.dumps(interrupt_message))
            except Exception as e:
                logger.error("send_to_twilio.error", call_id=call_id, error=str(e))

        try:
            await asyncio.gather(receive_from_twilio(), send_to_twilio())
        finally:
            stop_ts = datetime.utcnow().isoformat()
            global CURRENT_CALL_ID
            CURRENT_CALL_ID = None
            logger.info(
                "call.completed",
                call_id=call_id,
                start_time=start_ts,
                stop_time=stop_ts,
                outcome=transcripts,
            )

            # Persist transcript to file and record summary in the database
            transcript_dir = os.path.join(os.path.dirname(__file__), "transcripts")
            os.makedirs(transcript_dir, exist_ok=True)
            transcript_file = os.path.join(
                transcript_dir, f"{call_id}_{session_id or 'session'}.json"
            )
            try:
                with open(transcript_file, "w", encoding="utf-8") as f:
                    json.dump(transcripts, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                logger.error("transcript.save_failed", call_id=call_id, error=str(exc))
                transcript_file = None

            # Append plain text transcript for QA review
            qa_file = os.path.join(transcript_dir, f"call_{call_id}.txt")
            try:
                with open(qa_file, "a", encoding="utf-8") as f:
                    for item in transcripts:
                        role = None
                        content = None
                        if isinstance(item.get("message"), dict):
                            role = item["message"].get("role")
                            content = item["message"].get("content")
                        else:
                            role = item.get("role") or item.get("speaker")
                            content = item.get("content")
                        if content:
                            prefix = "GPT-4o" if role == "assistant" else "ASR"
                            f.write(f"{prefix}: {content}\n")
            except Exception as exc:
                logger.error("qa_transcript.save_failed", call_id=call_id, error=str(exc))

            try:
                duration = (
                    datetime.fromisoformat(stop_ts)
                    - datetime.fromisoformat(start_ts)
                ).total_seconds()
            except Exception:
                duration = 0.0

            save_call_summary(
                call_id=call_id,
                duration=duration,
                outcome="completed",
                scheduled_time=datetime.fromisoformat(start_ts),
                transcript_path=transcript_file,
            )

            metrics = compute_call_metrics(
                transcripts=transcripts,
                start_time=start_ts,
                stop_time=stop_ts,
                guardrail_rejects=guardrail_rejects,
                calendar_errors=CURRENT_METRICS.get("calendar_errors", 0)
                if CURRENT_METRICS
                else 0,
                latencies=latencies,
            )
            write_report(call_id, metrics)

            CURRENT_METRICS = None


async def send_session_update(openai_ws, session):
    """Send session update to OpenAI WebSocket."""
    slots = get_todays_free_slots()
    instructions = SYSTEM_MESSAGE
    if slots:
        formatted = "\n".join(f"- {s}" for s in slots)
        instructions += f"\n\nToday's available slots:\n{formatted}"
    session_update = {
        "type": "session.update",
        "session": {
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": instructions,
            "modalities": ["text", "audio"],
            "temperature": 0.2,
            "state": session.get("state"),
        },
    }
    logger.info("session.update.send", payload=session_update)
    await openai_ws.send(json.dumps(session_update))
