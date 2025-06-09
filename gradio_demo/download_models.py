from huggingface_hub import hf_hub_download
import kagglehub
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
os.makedirs("./checkpoints/insightface", exist_ok=True)

gdown.download(
    url="https://drive.google.com/uc?id=1tQsgEfP1gfQpu3IGeK0jCVMs4i6kD4g0",
    output="./checkpoints/insightface/antelopev2.zip",
    quiet=False
)

# === Unzip to models directory ===
print("🗜️ Extracting antelopev2.zip...")
with zipfile.ZipFile("./checkpoints/insightface/antelopev2.zip", 'r') as zip_ref:
    zip_ref.extractall("./checkpoints/insightface/models")

# Optional: Remove ZIP after extraction
os.remove("./checkpoints/insightface/antelopev2.zip")

print("✅ antelopev2 models ready in ./checkpoints/insightface/models/")
print("✅ All models downloaded and ready.")
