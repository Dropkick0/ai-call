from app import demo

# Ensure python-dotenv is installed at runtime
try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True) 