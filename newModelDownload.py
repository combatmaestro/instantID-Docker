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
os.makedirs('./models', exist_ok=True)

# Direct download link
url = "https://huggingface.co/combatmaestro/antelopeV2/resolve/main/antelopev2.zip"
zip_path = "./models/antelopev2.zip"

# Download the zip file
print("📥 Downloading antelopev2.zip from Hugging Face...")
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(zip_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

# Extract it
print("🗜️ Extracting antelopev2.zip...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("./models/antelopev2")

os.remove(zip_path)
print("✅ Model download and extraction complete.")

print("✅ All models downloaded and ready.")
