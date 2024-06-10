import torch

def main():
    if torch.cuda.is_available():
        print("CUDA is available!")
        version = torch.version.cuda
        print(f"CUDA version: {version}")
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device: {current_device}")
        device_name = torch.cuda.get_device_name(current_device)
        print(f"CUDA device name: {device_name}")
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    main()
