############################################ Virtual environment ############################################

FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Copy files or directory from the host machine into Docker image.
COPY . /app

# Install pycasper
RUN mkdir /pycasper && \
    git clone https://github.com/chahuja/pycasper /pycasper && \
    rm -rf /app/src/pycasper && \
    ln -s /pycasper/pycasper /app/src/pycasper

# Install Python dependencies into .venv
RUN python3 -m venv .venv && \
    . .venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Set the entrypoint and default command
ENTRYPOINT ["/app/.venv/bin/python", "src/train.py"]
CMD ["-batch_size", "32", "-cpk", "speech2gesture", "-early_stopping", "0", "-exp", "1", "-fs_new", "[15, 15]", "-gan", "1", "-loss", "L1Loss", "-model", "Speech2Gesture_G", "-note", "speech2gesture", "-num_epochs", "100", "-overfit", "0", "-render", "0", "-save_dir", "save/speech2gesture/oliver", "-stop_thresh", "3", "-tb", "1", "-window_hop", "5"]

############################################ No virtual environment ############################################

# FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# # Set the working directory in the container
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && \
#     apt-get install -y \
#     git \
#     python3 \
#     python3-pip && \
#     rm -rf /var/lib/apt/lists/*

# # Copy files or directory from the host machine into Docker image.
# COPY . /app

# # Install pycasper
# RUN mkdir /pycasper && \
#     git clone https://github.com/chahuja/pycasper /pycasper && \
#     rm -rf /app/src/pycasper && \
#     ln -s /pycasper/pycasper /app/src/pycasper

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Expose the necessary port
# EXPOSE 80

# # Set the entrypoint and default command
# ENTRYPOINT ["python3", "src/train.py"]
# CMD ["-batch_size", "32", "-cpk", "speech2gesture", "-early_stopping", "0", "-exp", "1", "-fs_new", "[15, 15]", "-gan", "1", "-loss", "L1Loss", "-model", "Speech2Gesture_G", "-note", "speech2gesture", "-num_epochs", "100", "-overfit", "0", "-render", "0", "-save_dir", "save/speech2gesture/oliver", "-stop_thresh", "3", "-tb", "1", "-window_hop", "5"]
