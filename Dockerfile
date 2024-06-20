FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy files or directory from the host machine into Docker image.
# COPY <src> <dest>
COPY  . /app

# Install dependencies
# No need to activate virtual environment in a docker container
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

CMD ["python", "src/train.py", "-batch_size", "32", "-cpk", "speech2gesture", "-early_stopping", "0", "-exp", "1", "-fs_new", "[15, 15]", "-gan", "1", "-loss", "L1Loss", "-model", "Speech2Gesture_G", "-note", "speech2gesture", "-num_epochs", "100", "-overfit", "0", "-render", "0", "-save_dir", "save/speech2gesture/oliver", "-stop_thresh", "3", "-tb", "1", "-window_hop", "5"]
