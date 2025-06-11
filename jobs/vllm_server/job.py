import wandb
import os
import json
import subprocess
import threading

wandb_config = json.loads(os.environ.get("WANDB_CONFIG"))

model_name = wandb_config.get("model")
artifact_path = wandb_config.get("artifact_path")

if not model_name and not artifact_path:
    raise ValueError("model_name or artifact_path are required")

wandb_api = wandb.Api()

model = wandb_api.artifact(artifact_path).download() if artifact_path else model_name

args = [
    "vllm",
    "serve",
    model,
    "--api-key",
    "token-abc123",
    "--trust-remote-code",
    "--enable-chunked-prefill",
    "--max-num-batched-tokens",
    "1024",
    "--chat-template",
    "chat_template.jinja"
]

print(f"Starting VLLM server with args: {args}")

vllm_server = subprocess.Popen(
    args,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
    universal_newlines=True,
)

# Start threads to capture and log output
def log_output(pipe, prefix):
    for line in pipe:
        print(f"{prefix}: {line.strip()}")

stdout_thread = threading.Thread(
    target=log_output, args=(vllm_server.stdout, "vllm stdout")
)
stderr_thread = threading.Thread(
    target=log_output, args=(vllm_server.stderr, "vllm stderr")
)
stdout_thread.daemon = True
stderr_thread.daemon = True
stdout_thread.start()
stderr_thread.start()

print("VLLM server starting ...")

try:
    vllm_server.wait()
except KeyboardInterrupt:
    print("Received interrupt signal, shutting down...")
    vllm_server.terminate()
    vllm_server.wait()
