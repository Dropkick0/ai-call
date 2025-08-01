import argparse
import asyncio
import os
import sys
from demo_mode_deepgram_voice_agent import DeepgramVoiceAgentPipeline

def main(call_id: int):
    # Set environment variable for log tagging
    os.environ["CALL_ID"] = str(call_id)
    pipeline = DeepgramVoiceAgentPipeline()
    asyncio.run(pipeline.start_voice_conversation())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--call-id", type=int, required=True)
    args = parser.parse_args()
    main(args.call_id) 