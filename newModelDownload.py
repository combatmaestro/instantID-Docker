from huggingface_hub import hf_hub_download
import shutil
import os
import gdown
import zipfile

# === Download from HuggingFace ===
print("📥 Downloading ControlNet config and model...")
hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/config.json",
    local_dir="./checkpoints",
)

hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/diffusion_pytorch_model.safetensors",
    local_dir="./checkpoints",
)

hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ip-adapter.bin",
    local_dir="./checkpoints",
)

hf_hub_download(
    repo_id="latent-consistency/lcm-lora-sdxl",
    filename="pytorch_lora_weights.safetensors",
    local_dir="./checkpoints",
)

print("📥 Downloading antelopev2.zip from Google Drive...")

# Ensure models directory exists
os.makedirs("./models", exist_ok=True)

# Download antelopev2.zip from Google Drive
url = "https://drive.google.com/uc?id=1tQsgEfP1gfQpu3IGeK0jCVMs4i6kD4g0"
output_path = "./models/antelopev2.zip"

print("📥 Downloading antelopev2.zip...")
gdown.download(url, output_path, quiet=False)

# Extract zip
print("🗜️ Extracting antelopev2.zip...")
with zipfile.ZipFile(output_path, 'r') as zip_ref:
    zip_ref.extractall("./models/antelopev2")

# Optional: remove zip after extraction
os.remove(output_path)
print("✅ Done!")

print("✅ All models downloaded and ready.")
