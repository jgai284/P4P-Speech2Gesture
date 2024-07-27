# Lab workstation setup

[Full instruction](https://github.com/UoA-CARES/essential-gpu-docker)

1. Download [Docker](https://www.docker.com/)
2. Create your own Dockerfile ([example](Dockerfile))
3. Build Docker image
4. Push to Docker hub
5. Start Docker container

```sh
docker run --gpus all -v /media/myuser1/Storage/PATSDATASET/speaker/pats/data:/app/data jamesgai207/gesture_generation:latest python src/train.py -path2data '/app/data' -path2outdata '/app/data' -batch_size 32 -cpk speech2gesture -early_stopping 0 -exp 1 -fs_new '[15, 15]' -gan 1 -input_modalities '["audio/log_mel_400"]' -loss L1Loss -modalities '["pose/normalize", "audio/log_mel_400"]' -model Speech2Gesture_G -note speech2gesture -num_epochs 100 -overfit 0 -render 0 -save_dir save/speech2gesture/oliver -speaker '["oliver"]' -stop_thresh 3 -tb 1 -window_hop 5
```

### R10 IP address:

`130.216.239.210`

### P6000-1 IP address:

`130.216.238.91`

### Username:

`myuser1`

### Password:

`pass1`

# Docker utility

### Dockerfile:

A set of instructions to create a Docker image

### Docker Image:

A blueprint of an application and its dependencies

### Docker layer:

A Docker image consists of layers indicated by the commands in Dockerfile

### Docker Container:

A running instance of a Docker image, which provides an isolated environment for the application to run

# Docker commands

Build image:

```sh
docker build -t <image-name>:<image-tag> -f Dockerfile .
```

Start container

```sh
docker run -it <image-name>:<image-tag> -path2data /app/data -path2outdata /app/data
```

Push to Docker hub

1. Tag the image with repo:

```sh
docker tag <image-name>:<image-tag> <repo-name>:<image-tag>
```

2. Push the image:

```sh
docker push <image-name>:<image-tag>
```

Mount directory to container

```sh
docker run -d -v <host_path>:<container_path> <image-id>
```

# Useful links

[Docker for beginners](https://docker-curriculum.com/)

[WSL storage transferation](https://needlify.com/post/how-to-move-wsl-distributions-including-docker-images-to-new-locations-on-windows-6412384cbd14c)
