import os
import io
import asyncio
from dotenv import load_dotenv
import numpy as np
import soundfile as sf
import gradio as gr
import sounddevice as sd
import subprocess
import threading
import sys

# Import the one-shot voice pipeline exposed in the demo file  
from demo_mode_GROQ_deepgram_voice_agent import run_voice_agent
from demo_mode_GROQ_deepgram_voice_agent import load_prompt as dg_load_prompt

# Load environment variables (DEEPGRAM_API_KEY, GROQ_API_KEY)
load_dotenv()

# ------------------------------------------------------------------
# Overlay styling for the call log modal
# ------------------------------------------------------------------
overlay_css = """
:root {
  --block-border-color: #1e293b;
  --background-fill-secondary: #1e293b;
}

.dark {
  --block-border-color: #1e293b !important;
  --border-color-primary: #1e293b !important;
  --panel-border-color: #1e293b !important;

}
#call_overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.7);
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
}

#call_overlay_inner {
  background: var(--panel-background-fill, #111827); /* match Gradio dark panel */
  padding: 1.5rem;
  border-radius: 12px;
  width: 40vw;            /* square-ish */
  max-width: 600px;
  height: 40vw;
  max-height: 600px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  align-items: center;
}

#call_header {
  font-size: 1.5rem !important;
  font-weight: 600;
  color: var(--body-text-color, #f9f9f9);
  margin-bottom: 0.5rem;
  text-align: center;
}

#call_header * {
  font-size: 1.5rem !important;
}

#call_subtitle {
  font-size: 0.9rem !important;
  color: var(--body-text-color-subdued, #9ca3af);
  margin-top: 0.5rem;
  text-align: center;
  font-style: italic;
}

#call_subtitle * {
  font-size: 0.9rem !important;
}


#end-call-btn {
  background-color: #e53935;
  color: #ffffff;
  width: 100%;
}

#timer_label {
  font-size: 4rem !important;  /* large timer */
  font-weight: 600;
  color: var(--body-text-color, #f9f9f9);
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0;
  text-align: center;
}

#timer_label * {
  font-size: 4rem !important;
}

/* Warmup status gets smaller font */
#timer_label *:contains("⚡"), 
#timer_label *:contains("🔔"),
#timer_label *:contains("❌"),
#timer_label *:contains("⚠️") {
  font-size: 2rem !important;
}

/* Call timer keeps large font */
#timer_label *:matches("[0-9][0-9]:[0-9][0-9]") {
  font-size: 4rem !important;
}

/* Limit height of the voice dropdown options so it scrolls */
#voice_dropdown ul {
  max-height: 300px;
  overflow-y: auto;
}

/* Highlight selected script column */
#script1_col.selected, #script2_col.selected {
  border: 3px solid var(--primary-500, #3b82f6);
  border-radius: 8px;
  padding: 0.5rem;
}

/* Consistent padding & rounded corners for custom containers */
#voice_container, #script_container {
  padding: 1rem;
  border-radius: 8px;
  background-color: #1e293b !important;
  border: 1px solid var(--block-border-color, #1e293b) !important;
  margin-top: 1rem;
}

/* Rounded corner style for main action buttons */
#script1_btn, #script2_btn, #preview_btn {
  border-radius: 6px !important;
  padding: 0.4rem 0.75rem !important;
}

/* Horizontal gap between script columns */
#script_btns_row > div + div {
  margin-left: 10px;
}

/* Increase top/bottom padding for script container */
#script_container {
  padding: 1.25rem 1rem;
}

/* Space below header row */
#script_header_row {
  margin-bottom: 0.75rem;
}

#voice_container .gr-column > * + * {
  margin-top: 0.75rem;
}

#@keyframes for pulse
@keyframes pulseFade {
  0% { opacity: 1; }
  50% { opacity: 0.6; }
  100% { opacity: 1; }
}

.pulse-close {
  animation: pulseFade 0.7s ease-in-out infinite;
}

#preview_btn,
#test_micro_btn {
  width: 100% !important;
}

#script1_col p, #script2_col p {
  margin-top: 0.75rem;
  margin-bottom: 0.75rem;

  /* Force Gradio block borders to new color */
}
.gr-block, .gr-box, .gr-accordion {
  border-color: #1e293b !important;
}
.gr-accordion .gr-panel {
  background-color: #1e293b !important;
  border-color: #1e293b !important;
}
"""

# Step 2: Audio setup
def list_devices(kind):
    devices = sd.query_devices()
    default_idx = sd.default.device[0 if kind == 'input' else 1]
    result = []
    for i, d in enumerate(devices):
        if d['max_' + kind + '_channels'] > 0:
            prefix = 'Default - ' if i == default_idx else ''
            result.append((i, prefix + d['name']))
    return result

def test_audio(input_device, output_device):
    import time as _t
    try:
        duration_sec = 1.0  # shorter capture
        recording = sd.rec(int(duration_sec * 48000), samplerate=48000, channels=1, device=int(input_device))
        sd.wait()
        if np.max(np.abs(recording)) < 0.01:
            yield "No input detected - check mic permissions or volume."
            return

        sd.play(recording, samplerate=48000, device=int(output_device))
        sd.wait()

        yield "Test passed! You should have heard your recording."
    except Exception as e:
        yield f"Test failed: {str(e)}. Try different devices or check connections."

# Step 3: Voice picker (choices list only – actual components are created inside the layout)
voices = [
    {"name": "Arista-PlayAI"},
    {"name": "Atlas-PlayAI"},
    {"name": "Basil-PlayAI"},
    {"name": "Briggs-PlayAI"},
    {"name": "Calum-PlayAI"},
    {"name": "Celeste-PlayAI"},
    {"name": "Cheyenne-PlayAI"},
    {"name": "Chip-PlayAI"},
    {"name": "Cillian-PlayAI"},
    {"name": "Deedee-PlayAI"},
    {"name": "Fritz-PlayAI"},
    {"name": "Gail-PlayAI"},
    {"name": "Indigo-PlayAI"},
    {"name": "Mamaw-PlayAI"},
    {"name": "Mason-PlayAI"},
    {"name": "Mikail-PlayAI"},
    {"name": "Mitch-PlayAI"},
    {"name": "Quinn-PlayAI"},
    {"name": "Thunder-PlayAI"},
]

# Default voice is now Cora
DEFAULT_VOICE = "Cheyenne-PlayAI"

# Helper to turn model name like "aura-2-thalia-en" into "Thalia"
def display_name(model_name: str) -> str:
    """Nicely format a PlayAI voice name."""
    if model_name.endswith("-PlayAI"):
        return model_name.replace("-PlayAI", "")
    return model_name

def update_voice_details(selected):
    for v in voices:
        if v["name"] == selected:
            details = display_name(v["name"])
            return details, None  # preview audio set via button
    return "", None

def play_preview(voice):
    """Generate a short Groq Play-AI TTS sample for the selected voice."""
    import io, numpy as np, soundfile as sf
    from groq_tts import synthesize_speech

    try:
        wav_bytes = synthesize_speech(
            "Hello, I am the selected Remember voice agent.",
            voice=voice,
        )
        buffer = io.BytesIO(wav_bytes)
        audio_np, sr = sf.read(buffer, dtype="int16")
        return (sr, audio_np)
    except Exception as e:
        print(f"Preview generation failed: {e}")
        return None

def stop_call():
    global current_process, current_pipeline
    
    print("Stopping voice agent...")
    
    # Signal the voice agent to stop
    if current_pipeline:
        try:
            # Set the is_active flag to False to break the main loop
            current_pipeline.is_active = False
            print("Voice agent stop signal sent")
            
            # Try to close the Deepgram connection directly (non-async)
            if hasattr(current_pipeline, 'connection') and current_pipeline.connection:
                try:
                    current_pipeline.connection.finish()
                    print("Deepgram connection closed")
                except Exception as e:
                    print(f"Error closing Deepgram connection: {e}")
            
            # Stop audio streams directly
            try:
                import sounddevice as sd
                sd.stop()
                if hasattr(current_pipeline, 'output_stream') and current_pipeline.output_stream:
                    current_pipeline.output_stream.stop()
                    current_pipeline.output_stream.close()
                print("Audio streams stopped")
            except Exception as e:
                print(f"Error stopping audio: {e}")
                
        except Exception as e:
            print(f"Error stopping voice agent: {e}")
    
    # Clean up references
    current_process = None
    current_pipeline = None
    
    return "⏹️ Call stopped", gr.update(visible=False)

def start_call(input_device, output_device, voice, script):
    def get_device_name(idx):
        devices = sd.query_devices()
        return devices[int(idx)]['name']
    
    global current_process, current_pipeline
    prompt_path = "prompts/system_prompt.txt" if script == "Friend" else "prompts/system_prompt BACKUP.txt"
    
    # Set environment variables for the voice agent
    os.environ["AGENT_VOICE_NAME"] = str(voice)
    os.environ["AGENT_PROMPT_PATH"] = str(prompt_path)
    os.environ["SD_INPUT_DEVICE"] = str(get_device_name(input_device))
    os.environ["SD_OUTPUT_DEVICE"] = str(get_device_name(output_device))
    
    # Import the voice agent function
    try:
        from demo_mode_GROQ_deepgram_voice_agent import DeepgramVoiceAgentPipeline
        
        # Create a thread to run the voice agent
        import threading
        import time as _t
        
        agent_running = threading.Event()
        warmup_complete = threading.Event()
        agent_thread = None
        
        def run_agent():
            global current_pipeline
            try:
                import asyncio
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                pipeline = DeepgramVoiceAgentPipeline()
                current_pipeline = pipeline  # Store reference for stopping
                
                # Add a flag to track when warmup is complete (dial tone has played)
                pipeline.warmup_complete = warmup_complete
                
                agent_running.set()
                # Run the async voice conversation
                loop.run_until_complete(pipeline.start_voice_conversation())
            except Exception as e:
                print(f"Voice agent error: {e}")
            finally:
                agent_running.clear()
                current_pipeline = None
                # Clean up the event loop
                try:
                    loop.close()
                except:
                    pass
        
        # Start the voice agent in a separate thread
        agent_thread = threading.Thread(target=run_agent, daemon=True)
        agent_thread.start()
        
        # Store the thread reference so we can stop it later
        current_process = agent_thread
        
        # Wait for agent to start
        _t.sleep(1)
        
        # Show warmup status while waiting for dial tone
        warmup_start = _t.time()
        print("UI: Starting warmup monitoring...")
        while agent_running.is_set() and agent_thread.is_alive() and not warmup_complete.is_set():
            elapsed = int(_t.time() - warmup_start)
            status_msg = f"⚡ Warming up systems... {elapsed}s"
            print(f"UI: {status_msg}")
            yield status_msg
            _t.sleep(0.5)
            
            # Timeout after 30 seconds of warmup
            if elapsed > 30:
                print("UI: Warmup timeout reached")
                yield "⚠️ Warmup timeout - system may not be ready"
                break
        
        # Check if warmup completed successfully
        if not warmup_complete.is_set():
            print("UI: Warmup failed - event not set")
            yield "❌ Warmup failed - call may not work properly"
            return
        
        print("UI: Warmup completed successfully - starting call timer")
        # Now start the actual call timer (after dial tone)
        yield "🔔 Dial tone complete - Call started!"
        _t.sleep(1)  # Brief pause to show the "Call started" message
        
        call_start_ts = _t.time()
        
        # Keep updating the call timer while the agent is running
        while agent_running.is_set() and agent_thread.is_alive():
            # No automatic termination - only manual stop should end calls
            
            elapsed = int(_t.time() - call_start_ts)
            mins, secs = divmod(elapsed, 60)
            yield f"{mins:02d}:{secs:02d}"
            _t.sleep(1)
            
    except Exception as e:
        yield f"❌ Failed to start call: {e}"
        return

    yield "📞 Call ended"

# ------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------
with gr.Blocks(title="Re:MEMBER AI Voice Agent Demo", theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray", neutral_hue="slate", font=[gr.themes.GoogleFont("Inter")]), css=overlay_css) as demo:
    gr.Markdown("""
    # Re:MEMBER AI Voice Agent Demo
    
    Select options below and start a continuous voice call. 
    
    **📝 Note:** The system will warm up all components (LLM, STT, TTS, VAD) before signaling readiness with a dial tone. Call timer starts after the dial tone.
    """)

    # ------------------------------------------------------------------
    # Overlay modal for call log & End Call
    # ------------------------------------------------------------------
    with gr.Group(elem_id="call_overlay", visible=False) as call_overlay:
        with gr.Column(elem_id="call_overlay_inner"):
            gr.Markdown("### Voice Agent Status", elem_id="call_header")
            timer_label = gr.Markdown(value="⚡ Initializing...", elem_id="timer_label")
            gr.Markdown("*Timer shows warmup progress, then call duration*", elem_id="call_subtitle") 
            end_button = gr.Button("End Call", elem_id="end-call-btn")

    # --- Script selection helpers ---
    def get_system_prompt(script):
        if script == "Gatekeeper":
            return dg_load_prompt("prompts/system_prompt BACKUP.txt")
        return dg_load_prompt("prompts/system_prompt.txt")

    prompt_state = gr.State(get_system_prompt("Gatekeeper"))
    selected_script_state = gr.State("Gatekeeper")  # default selection

    with gr.Accordion("Select and Test Audio Devices", open=True, elem_id="audio_acc") as audio_accordion:
        input_devices = list_devices('input')
        output_devices = list_devices('output')
        default_input_idx = sd.default.device[0]
        default_output_idx = sd.default.device[1]
        input_dropdown = gr.Dropdown(
            choices=[(name, idx) for idx, name in input_devices],
            label="Input Device",
            value=default_input_idx
        )
        output_dropdown = gr.Dropdown(
            choices=[(name, idx) for idx, name in output_devices],
            label="Output Device",
            value=default_output_idx
        )
        gr.Markdown("**Tip:** Ensure microphone permission is granted. Click the 'i' icon next to the URL in your browser if prompted.")
        test_button = gr.Button("Test Microphone to Enable Demo - Say 'Hello'", elem_id="test_micro_btn")
        test_output = gr.Textbox(label="Test Result")

    # --------------- Voice selection container ---------------
    with gr.Group(elem_id="voice_container"):
        with gr.Row():
            with gr.Column(scale=2):
                # Dropdown now shows full voice details (Name - Gender - Characteristics)
                voice_dropdown = gr.Dropdown(
                    choices=[(
                        display_name(v['name']),
                        v['name']
                    ) for v in voices],
                    label="Select Agent Voice",
                    value=DEFAULT_VOICE,
                    elem_id="voice_dropdown",
                )

                preview_button = gr.Button("Preview Voice", elem_id="preview_btn")
                preview_audio = gr.Audio(label="Voice Preview", interactive=False, autoplay=True)

    # ---------------- Script selection container ----------------
    with gr.Group(elem_id="script_container"):
        # Header row with title and selected label side-by-side
        with gr.Row(elem_id="script_header_row"):
            with gr.Column():
                gr.Markdown("### Choose a Call Script", elem_id="script_header")
            with gr.Column():
                selected_label = gr.Markdown("**Selected: Script 1**", elem_id="selected_script_label")

        with gr.Row(elem_id="script_btns_row"):
            # Script 1 (Gatekeeper)
            with gr.Column(elem_id="script1_col"):
                script1_button = gr.Button("Script 1", variant="primary", elem_id="script1_btn")
                gr.Markdown("Persistent yet polite strategy aimed at navigating gatekeepers and securing a brief follow-up with the decision-maker.")

            # Script 2 (Friend)
            with gr.Column(elem_id="script2_col"):
                script2_button = gr.Button("Script 2", variant="secondary", elem_id="script2_btn")
                gr.Markdown("Friendly, conversational approach focused on building rapport and enlisting the person as a champion for our directory.")

    # Callback to handle script selection
    def choose_script(script):
        label = "**Selected: Script 1**" if script == "Gatekeeper" else "**Selected: Script 2**"
        btn1_variant = "primary" if script == "Gatekeeper" else "secondary"
        btn2_variant = "primary" if script == "Friend" else "secondary"
        prompt = get_system_prompt(script)
        return (
            prompt,  # update prompt_state
            script,  # update selected_script_state
            gr.update(value=label),
            gr.update(variant=btn1_variant),
            gr.update(variant=btn2_variant),
        )

    script1_button.click(
        fn=lambda: choose_script("Gatekeeper"),
        outputs=[prompt_state, selected_script_state, selected_label, script1_button, script2_button],
    )

    script2_button.click(
        fn=lambda: choose_script("Friend"),
        outputs=[prompt_state, selected_script_state, selected_label, script1_button, script2_button],
    )

    # Clear preview audio whenever a new voice is selected
    voice_dropdown.change(fn=lambda _: None, inputs=voice_dropdown, outputs=preview_audio)

    # ----- Start Call button at bottom -----
    start_button = gr.Button("Start Call", interactive=False)

    mic_test_event = test_button.click(
        fn=test_audio,
        inputs=[input_dropdown, output_dropdown],
        outputs=test_output,
        queue=False,
    )

    # Client-side pulse and collapse after 3 seconds
    mic_test_event.then(
        js="""
        (msg) => {
          if(msg.startsWith('Test passed')) {
            const acc = document.getElementById('audio_acc');
            if(acc){
              acc.classList.add('pulse-close');
              setTimeout(()=>{
                acc.open = false;
                acc.classList.remove('pulse-close');
              }, 3000);
            }
          }
        }
        """,
        outputs=[],
    )

    # Enable Start button after test passes
    def enable_start(msg):
        if msg.startswith("Test passed"):
            return gr.update(interactive=True)
        return gr.update(interactive=False)

    mic_test_event.then(enable_start, inputs=test_output, outputs=start_button, queue=False)

    # Show the overlay immediately when the call starts (no queue to make it instant)  
    start_button.click(fn=lambda: gr.update(visible=True), outputs=call_overlay, queue=False)
    
    # Also reset the timer display when showing overlay
    start_button.click(fn=lambda: "⚡ Starting systems...", outputs=timer_label, queue=False)

    # Start streaming logs to the overlay textbox
    start_button.click(
        fn=start_call,
        inputs=[input_dropdown, output_dropdown, voice_dropdown, selected_script_state],
        outputs=timer_label,
    )

    end_button.click(fn=stop_call, outputs=[timer_label, call_overlay])
    preview_button.click(fn=play_preview, inputs=voice_dropdown, outputs=preview_audio)

# Ensure voice defaults to 'aura-2-thalia-en' (already in voices list)

# Start call generator
current_process = None
current_pipeline = None

# Launch with share=True to get the temporary public *.gradio.live link
if __name__ == "__main__":
    # The tunnel Gradio spins up stays live for ~72 h
    demo.launch(share=True) 