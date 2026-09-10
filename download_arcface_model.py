#!/usr/bin/env python3
"""
Download ArcFace ONNX model for face recognition.
This script downloads a working ArcFace model that's compatible with your pipeline.
"""
import os
import urllib.request
from pathlib import Path

def download_arcface_model():
    """Download ArcFace ONNX model from a reliable source."""
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "embedder_arcface.onnx"
    
    if model_path.exists():
        print(f"✅ ArcFace model already exists at: {model_path}")
        print(f"📊 File size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        return
    
    print("🔍 Searching for available ArcFace models...")
    
    # List of potential model sources (in order of preference)
    model_sources = [
        {
            "name": "ArcFace ResNet50 (Hugging Face)",
            "url": "https://huggingface.co/onnx-community/arcface-onnx/resolve/main/arcface.onnx",
            "size_mb": "~63",
            "description": "Good balance of accuracy and speed"
        },
        {
            "name": "ArcFace from py-feat",
            "url": "https://huggingface.co/py-feat/arcface_r50/resolve/main/onnx/model.onnx",
            "size_mb": "~63", 
            "description": "ResNet50 backbone, well-tested"
        },
        {
            "name": "ONNX Model Zoo ArcFace",
            "url": "https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx",
            "size_mb": "~236",
            "description": "ResNet100 backbone, higher accuracy"
        }
    ]
    
    print("\n📋 Available models:")
    for i, source in enumerate(model_sources, 1):
        print(f"{i}. {source['name']} ({source['size_mb']} MB)")
        print(f"   {source['description']}")
    
    print(f"\n🚀 Attempting to download the first available model...")
    
    for source in model_sources:
        try:
            print(f"\n⬇️  Trying: {source['name']}")
            print(f"    URL: {source['url']}")
            print(f"    Expected size: {source['size_mb']} MB")
            
            # Download with progress
            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) / total_size)
                    mb_downloaded = (block_num * block_size) / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    print(f"\r    Progress: {percent:.1f}% ({mb_downloaded:.1f}/{total_mb:.1f} MB)", end="", flush=True)
            
            urllib.request.urlretrieve(source["url"], model_path, reporthook=show_progress)
            print()  # New line after progress
            
            # Verify the download
            if model_path.exists() and model_path.stat().st_size > 1024:  # At least 1KB
                file_size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"✅ Successfully downloaded: {source['name']}")
                print(f"📊 File size: {file_size_mb:.1f} MB")
                print(f"💾 Saved to: {model_path}")
                
                # Test if it's a valid ONNX model
                try:
                    import onnxruntime as ort
                    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                    input_shape = sess.get_inputs()[0].shape
                    output_shape = sess.get_outputs()[0].shape
                    print(f"🔍 Model validation successful!")
                    print(f"   Input shape: {input_shape}")
                    print(f"   Output shape: {output_shape}")
                    print(f"\n🎉 ArcFace model ready! You can now run your face recognition system.")
                    return
                except Exception as e:
                    print(f"⚠️  Model validation failed: {e}")
                    model_path.unlink()  # Delete invalid file
                    continue
            else:
                print(f"❌ Download failed or file too small")
                if model_path.exists():
                    model_path.unlink()
                continue
                
        except Exception as e:
            print(f"❌ Failed to download from {source['name']}: {e}")
            if model_path.exists():
                model_path.unlink()
            continue
    
    # If all downloads failed, provide manual instructions
    print(f"\n❌ Automatic download failed for all sources.")
    print(f"\n📖 Manual download options:")
    print(f"1. Visit: https://huggingface.co/onnx-community/arcface-onnx")
    print(f"2. Download the .onnx file")
    print(f"3. Save it as: {model_path}")
    print(f"\n📖 Alternative sources:")
    print(f"- InsightFace official models: https://github.com/deepinsight/insightface")
    print(f"- ONNX Model Zoo: https://github.com/onnx/models")
    print(f"- OpenVINO Model Zoo: https://github.com/openvinotoolkit/open_model_zoo")

if __name__ == "__main__":
    print("🤖 ArcFace Model Downloader")
    print("=" * 50)
    
    # Check if onnxruntime is available
    try:
        import onnxruntime as ort
        print("✅ ONNXRuntime is available")
    except ImportError:
        print("❌ ONNXRuntime not found. Install with: pip install onnxruntime")
        exit(1)
    
    download_arcface_model()