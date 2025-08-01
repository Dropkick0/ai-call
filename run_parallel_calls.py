import argparse
import os
import subprocess
import sys

MAX_CALLS = 5  # Deepgram PAYG concurrency limit

# To Launch: python run_parallel_calls.py -n 5

def launch(n_calls: int):
    if n_calls > MAX_CALLS:
        print(f"Clamping to max {MAX_CALLS} concurrent calls (Deepgram limit)")
        n_calls = MAX_CALLS
    
    for i in range(1, n_calls + 1):
        # Launch in new cmd window with title
        subprocess.Popen(
            ["start", "cmd", "/k", 
             f"title CALL-{i} && python worker_main.py --call-id {i}"],
            shell=True
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--calls", type=int, default=3,
                        help="Number of parallel calls (1-5)")
    args = parser.parse_args()
    launch(args.calls) 