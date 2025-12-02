###########################################################
# This is the dockerfile providing the enviroment 
# from which to use the LLM extractions from
###########################################################

# Get NVIDIA CUDA image with Ubuntu 22.04 base
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG USERNAME=snakerun
ARG USER_UID= # Your UID
ARG USER_GID= # Your GID

RUN groupadd -g ${USER_GID} ${USERNAME} \
    && useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash ${USERNAME}

USER ${USERNAME}
ENV HOME=/home/${USERNAME}
ENV USER=${USERNAME}
ENV PATH=$HOME/miniforge/bin:$PATH
WORKDIR /home/${USERNAME}/workspace

# Environment setup
ENV HOME=/home/${USERNAME}
ENV MINIFORGE_DIR=$HOME/miniforge
ENV PATH=$MINIFORGE_DIR/bin:$PATH

# Install Miniforge (includes Mamba)
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -O /tmp/miniforge.sh \
    && /bin/bash /tmp/miniforge.sh -b -p $MINIFORGE_DIR \
    && rm /tmp/miniforge.sh

# Ensure PATH is available in *all* shell types (interactive, non-login, etc.)
ENV PATH="/home/${USERNAME}/miniforge/bin:${PATH}"

# Initialize conda and auto-activate base environment for interactive shells
RUN echo ". $MINIFORGE_DIR/etc/profile.d/conda.sh" >> $HOME/.bashrc \
    && echo "conda activate base" >> $HOME/.bashrc

# Create cache directories
ENV XDG_CACHE_HOME=$HOME/.cache
ENV SNAKEMAKE_HOME=$HOME/.snakemake
RUN mkdir -p $XDG_CACHE_HOME $SNAKEMAKE_HOME

RUN pip install --no-cache-dir snakemake==9.3.4

CMD ["bash"]

# build from the project folder with
#   docker build -t name:12.4.1 .

# to run docker: 
#   docker run -it --rm --gpus all -w /workspace/workflow -v $(pwd):/workspace -u uid:gid  name:12.4.1
# replace -u with your uid and gid look up via id

# a shell should be available after docker run in which snakemake can be run, e.g.
#   snakemake -n