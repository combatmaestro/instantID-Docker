from huggingface_hub import hf_hub_download
import kagglehub
import shutil
import os

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

# === Download antelopev2 via kagglehub ===
print("📥 Downloading antelopev2 model via kagglehub...")

# This will download and extract into a cache dir like ~/.kagglehub/
path = kagglehub.dataset_download("dipakbg145198/antelopev2")

print("📦 antelopev2 files downloaded to:", path)

# === Copy files to the project directory ===
target_dir = "./checkpoints/insightface/models"
os.makedirs(target_dir, exist_ok=True)

for file in os.listdir(path):
    source_file = os.path.join(path, file)
    target_file = os.path.join(target_dir, file)
    shutil.copy2(source_file, target_file)

print("✅ antelopev2 model copied to:", target_dir)
print("✅ All models downloaded and ready.")
