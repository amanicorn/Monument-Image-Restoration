import subprocess
import time
from pyngrok import ngrok

# 1. Kill any existing instances on 7860
subprocess.run(["pkill", "-9", "-f", "gradio_app.py"])

# 2. Start the Gradio backend
print("--> Starting Gradio backend...")
proc = subprocess.Popen(["python", "gradio_app.py"])
time.sleep(4)

# 3. Connect ngrok tunnel
print("--> Opening public ngrok tunnel...")
public_url = ngrok.connect(7860)

print("\n" + "=" * 60)
print(f"🚀 YOUR LIVE PUBLIC LINK: {public_url.public_url}")
print("=" * 60 + "\n")

try:
    proc.wait()
except KeyboardInterrupt:
    ngrok.kill()
    proc.terminate()
