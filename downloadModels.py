#!/usr/bin/env python3
"""
Complete setup script for InstantID API
This script handles model downloading, caching, and environment setup
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
import requests
from huggingface_hub import hf_hub_download
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InstantIDSetup:
    """Complete setup for InstantID API"""
    
    def __init__(self, cache_dir: str = "./model_cache"):
        self.cache_dir = Path(cache_dir)
        self.models_dir = self.cache_dir / "models" / "antelopev2"
        self.checkpoints_dir = self.cache_dir / "checkpoints"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.antelopev2_models = {
            "1k3d68.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/1k3d68.onnx",
            "2d106det.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/2d106det.onnx",
            "genderage.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/genderage.onnx", 
            "scrfd_10g_bnkps.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_10g_bnkps.onnx"
        }
    
    def check_system_requirements(self):
        """Check system requirements"""
        logger.info("Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            raise Exception("Python 3.8 or higher is required")
        
        # Check if CUDA is available
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            logger.info(f"CUDA available: {cuda_available}")
            if cuda_available:
                logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        except ImportError:
            logger.warning("PyTorch not installed yet")
        
        logger.info("✓ System requirements check passed")
    
    def install_dependencies(self):
        """Install Python dependencies"""
        logger.info("Installing Python dependencies...")
        
        requirements = [
            "fastapi==0.104.1",
            "uvicorn[standard]==0.24.0", 
            "torch==2.0.1",
            "torchvision==0.15.2",
            "diffusers==0.24.0",
            "transformers==4.35.2",
            "accelerate==0.25.0",
            "insightface==0.7.3",
            "opencv-python==4.8.1.78",
            "Pillow==10.1.0",
            "numpy==1.24.3",
            "pydantic==2.5.0",
            "python-multipart==0.0.6",
            "aiofiles==23.2.0",
            "huggingface-hub==0.19.4",
            "requests==2.31.0",
            "gdown==4.7.1",
            "tqdm==4.66.1",
            "python-dotenv==1.0.0"
        ]
        
        try:
            for req in requirements:
                logger.info(f"Installing {req}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            
            logger.info("✓ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            raise
    
    def download_file_with_progress(self, url: str, destination: Path) -> bool:
        """Download file with progress tracking"""
        try:
            logger.info(f"Downloading {destination.name}...")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0 and downloaded % (5 * 1024 * 1024) == 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Progress: {progress:.1f}% ({downloaded/1024/1024:.1f}MB)")
            
            logger.info(f"✓ Downloaded {destination.name} ({downloaded/1024/1024:.1f}MB)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {str(e)}")
            if destination.exists():
                destination.unlink()
            return False
    
    def download_antelopev2_models(self) -> bool:
        """Download AntelopeV2 face analysis models"""
        logger.info("Downloading AntelopeV2 models...")
        
        for model_name, url in self.antelopev2_models.items():
            model_path = self.models_dir / model_name
            
            if model_path.exists() and model_path.stat().st_size > 0:
                logger.info(f"✓ {model_name} already exists")
                continue
            
            success = self.download_file_with_progress(url, model_path)
            if not success:
                return False
        
        logger.info("✓ AntelopeV2 models downloaded successfully")
        return True
    
    def download_instantid_models(self) -> bool:
        """Download InstantID models from HuggingFace"""
        try:
            logger.info("Downloading InstantID models from HuggingFace...")
            
            # Create ControlNet directory
            controlnet_dir = self.checkpoints_dir / "ControlNetModel"
            controlnet_dir.mkdir(exist_ok=True)
            
            # Download files
            files_to_download = [
                ("ip-adapter.bin", "ip-adapter.bin"),
                ("ControlNetModel/config.json", "ControlNetModel/config.json"),
                ("ControlNetModel/diffusion_pytorch_model.safetensors", "ControlNetModel/diffusion_pytorch_model.safetensors")
            ]
            
            for hf_filename, local_filename in files_to_download:
                local_path = self.checkpoints_dir / local_filename
                
                if local_path.exists() and local_path.stat().st_size > 0:
                    logger.info(f"✓ {local_filename} already exists")
                    continue
                
                logger.info(f"Downloading {local_filename}...")
                hf_hub_download(
                    repo_id="InstantX/InstantID",
                    filename=hf_filename,
                    local_dir=str(self.checkpoints_dir),
                    local_dir_use_symlinks=False
                )
                logger.info(f"✓ Downloaded {local_filename}")
            
            logger.info("✓ InstantID models downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download InstantID models: {str(e)}")
            return False
    
    def download_pipeline_file(self) -> bool:
        """Download the InstantID pipeline file"""
        pipeline_url = "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/community/pipeline_stable_diffusion_xl_instantid.py"
        pipeline_path = Path("pipeline_stable_diffusion_xl_instantid.py")
        
        if pipeline_path.exists():
            logger.info("✓ Pipeline file already exists")
            return True
        
        logger.info("Downloading InstantID pipeline file...")
        return self.download_file_with_progress(pipeline_url, pipeline_path)
    
    def create_env_file(self):
        """Create .env file with configuration"""
        env_content = f"""# InstantID API Configuration
MODEL_CACHE_DIR={self.cache_dir}
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
CUDA_VISIBLE_DEVICES=0

# HuggingFace Cache
HF_HOME={self.cache_dir}/huggingface
TRANSFORMERS_CACHE={self.cache_dir}/transformers
DIFFUSERS_CACHE={self.cache_dir}/diffusers
"""
        
        with open(".env", "w") as f:
            f.write(env_content)
        
        logger.info("✓ Created .env file")
    
    def verify_installation(self) -> bool:
        """Verify that all models are downloaded correctly"""
        logger.info("Verifying installation...")
        
        required_files = [
            self.models_dir / "1k3d68.onnx",
            self.models_dir / "2d106det.onnx", 
            self.models_dir / "genderage.onnx",
            self.models_dir / "scrfd_10g_bnkps.onnx",
            self.checkpoints_dir / "ip-adapter.bin",
            self.checkpoints_dir / "ControlNetModel" / "config.json",
            self.checkpoints_dir / "ControlNetModel" / "diffusion_pytorch_model.safetensors",
            Path("pipeline_stable_diffusion_xl_instantid.py")
        ]
        
        missing_files = []
        total_size = 0
        
        for file_path in required_files:
            if not file_path.exists() or file_path.stat().st_size == 0:
                missing_files.append(str(file_path))
            else:
                total_size += file_path.stat().st_size
        
        if missing_files:
            logger.error(f"Missing files: {missing_files}")
            return False
        
        logger.info(f"✓ All models verified (Total: {total_size/1024/1024/1024:.2f}GB)")
        return True
    
    async def setup_complete(self):
        """Complete setup process"""
        try:
            logger.info("=" * 50)
            logger.info("InstantID API Setup Starting")
            logger.info("=" * 50)
            
            # Step 1: Check system requirements
            self.check_system_requirements()
            
            # Step 2: Install dependencies
            self.install_dependencies()
            
            # Step 3: Download pipeline file
            pipeline_success = self.download_pipeline_file()
            if not pipeline_success:
                raise Exception("Failed to download pipeline file")
            
            # Step 4: Download models
            antelopev2_success = self.download_antelopev2_models()
            if not antelopev2_success:
                raise Exception("Failed to download AntelopeV2 models")
            
            instantid_success = self.download_instantid_models()
            if not instantid_success:
                raise Exception("Failed to download InstantID models")
            
            # Step 5: Create environment file
            self.create_env_file()
            
            # Step 6: Verify installation
            if not self.verify_installation():
                raise Exception("Installation verification failed")
            
            logger.info("=" * 50)
            logger.info("✓ InstantID API Setup Complete!")
            logger.info("=" * 50)
            logger.info("To start the API server, run:")
            logger.info("  python main.py")
            logger.info("Or:")
            logger.info("  uvicorn main:app --host 0.0.0.0 --port 8000")
            logger.info("=" * 50)
            
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            return False

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="InstantID API Setup")
    parser.add_argument("