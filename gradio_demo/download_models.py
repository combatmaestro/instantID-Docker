from huggingface_hub import hf_hub_download
import os
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

# === Download antelopev2.zip using Kaggle curl ===
print("📥 Downloading antelopev2.zip from Kaggle...")

# Target directories
insightface_dir = "./checkpoints/insightface"
zip_path = os.path.join(insightface_dir, "antelopev2.zip")

# Ensure folder exists
os.makedirs(insightface_dir, exist_ok=True)

# Use curl to download (requires kaggle.json properly set up)
curl_command = (
    f"curl -L -o {zip_path} "
    "https://www.kaggle.com/api/v1/datasets/download/dipakbg145198/antelopev2"
)
os.system(curl_command)

# === Unzip the downloaded file ===
print("🗜️ Extracting antelopev2.zip...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(insightface_dir)

# Optional: Remove ZIP after extraction
os.remove(zip_path)

print("✅ All models downloaded and extracted successfully.")
