#!bin/bash

parent_dir=$(dirname "$(pwd)")

docker run -it -d \
    -p 2255:22 \
    --name lexicon3d-container \
    --runtime=nvidia \
    -v $parent_dir:/workspace \
    --shm-size=256gb \
    -v ~/.bashrc:/root/.bashrc \
    okamoto/lexicon3d 

    # -v /mnt/nfsshare1/datasets-share/nerf/kandao_movie/KD_20240228_193649_MP4:/workspace/NeRF-tutorial/datasets 