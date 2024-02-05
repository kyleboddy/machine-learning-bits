# neat file to run various CUDA / pytorch / tensorflow availabilities on your new or reconfigured system
import subprocess
import sys

def check_cuda():
    try:
        # Check NVIDIA driver version to infer CUDA installation
        nvcc_version = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        if "release" in nvcc_version:
            cuda_version = nvcc_version.split("release")[1].split(",")[0].strip()
            print(f"CUDA is installed with version: {cuda_version}")
        else:
            print("CUDA is installed, but version could not be determined.")
    except Exception as e:
        print(f"CUDA check failed: {e}")

def check_tensorflow():
    try:
        import tensorflow as tf
        print(f"TensorFlow is installed with version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"TensorFlow GPU Devices: {[gpu.name for gpu in gpus]}")
        else:
            print("TensorFlow is installed, but no GPU devices were found.")
    except ImportError:
        print("TensorFlow is not installed.")
    except Exception as e:
        print(f"An error occurred while checking TensorFlow: {e}")

def check_pytorch():
    try:
        import torch
        print(f"PyTorch is installed with version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"PyTorch CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("PyTorch is installed, but CUDA is not available.")
    except ImportError:
        print("PyTorch is not installed.")
    except Exception as e:
        print(f"An error occurred while checking PyTorch: {e}")

if __name__ == "__main__":
    check_cuda()
    check_tensorflow()
    check_pytorch()
