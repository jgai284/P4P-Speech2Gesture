FROM nvidia/cuda:12.2-cudnn8-runtime-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Copy files or directory from the host machine into Docker image.
COPY . /app

RUN apt-get update && apt-get install -y python3-pip

# Install pycasper
RUN mkdir /pycasper \
    && git clone https://github.com/chahuja/pycasper /pycasper \
    && rm -rf /app/src/pycasper \
    && ln -s /pycasper/pycasper /app/src/pycasper

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

EXPOSE 80

ENTRYPOINT ["python", "src/train.py"]
CMD ["-batch_size", "32", "-cpk", "speech2gesture", "-early_stopping", "0", "-exp", "1", "-fs_new", "[15, 15]", "-gan", "1", "-loss", "L1Loss", "-model", "Speech2Gesture_G", "-note", "speech2gesture", "-num_epochs", "100", "-overfit", "0", "-render", "0", "-save_dir", "save/speech2gesture/oliver", "-stop_thresh", "3", "-tb", "1", "-window_hop", "5"]
