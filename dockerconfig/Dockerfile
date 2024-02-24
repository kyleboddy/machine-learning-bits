# Use an NVIDIA CUDA base image
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Set non-interactive installation mode for apt-get and initial PATH
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/usr/local/cuda/bin:${PATH}

# Set the workspace folder and MAX_JOBS for ninja/flash-attn
WORKDIR /workspace
ENV MAX_JOBS=6

# Remove the installation of r-base from the system packages section
RUN apt-get update && apt-get install -y \
    openssh-server sudo vim git curl wget ffmpeg libsm6 libxext6 libxrender-dev \
    python3-pip python3-dev build-essential libopencv-dev libssl-dev zlib1g-dev \
    cudnn9-cuda-12 libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev \
    libswresample-dev libswscale-dev python3-venv btop nvtop nano net-tools htop \
    zip unzip git-lfs tmux screen glances cmake software-properties-common \
    && add-apt-repository ppa:ondrej/php \
    && apt-get update \
    && apt-get install -y php8.0 php8.0-cli php8.0-common php8.0-mysql php8.0-xml \
    php8.0-mbstring php8.0-curl php8.0-gd libmariadb-dev libmariadb-dev-compat \
    ninja-build jq python3-tk libpng-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/miniconda \
    && rm /tmp/miniconda.sh

# Set PATH to include Miniconda for the Dockerfile session
ENV PATH=/opt/miniconda/bin:${PATH}

# Add PATH setting for Conda and CUDA to .bashrc and .profile to set the PATH permanently
RUN echo "export PATH=/opt/miniconda/bin:/usr/local/cuda/bin:${PATH}" >> /root/.bashrc && \
    echo "export PATH=/opt/miniconda/bin:/usr/local/cuda/bin:${PATH}" >> /root/.profile


# Set up Conda and append conda-forge to the channels
RUN conda config --append channels conda-forge

# Install ezc3d using conda
RUN conda install -c conda-forge ezc3d

# Install NodeJS
RUN apt-get update && apt-get install -y ca-certificates curl gnupg \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /usr/share/keyrings/nodesource.gpg \
    && NODE_MAJOR=20 \
    && echo "deb [signed-by=/usr/share/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update && apt-get install nodejs -y \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set up SSH for remote connections
RUN mkdir /var/run/sshd \
    && echo 'root:root' | chpasswd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd
EXPOSE 22

# Install Python packages and database connectors using Conda as the handler
RUN /opt/miniconda/bin/pip install --no-cache-dir opencv-python ipykernel xgboost lightgbm pandas scikit-learn scipy numpy seaborn matplotlib build nvitop ffmpeg-python Jinja2 imageio[ffmpeg] mysql-connector-python PyMySQL torch transformers datasets deepspeed bitsandbytes apex exllamav2 tokenizers sentencepiece optuna ultralytics supervision mariadb

# Install dlib and tensorflow after the main install
RUN /opt/miniconda/bin/pip install --no-cache-dir dlib tensorflow

# Install face-recognition and flash-attn after the tensorflow install
RUN /opt/miniconda/bin/pip install --no-cache-dir face-recognition flash-attn

# Install R and R packages using Conda, this part is painful
RUN conda install -c r r-base \
    && conda install -c r r-ggplot2 r-dplyr r-tidyr r-shiny r-caret r-randomForest r-rmarkdown r-keras r-tensorflow r-imager r-xgboost r-lightgbm r-h2o r-RMySQL r-RMariaDB
	
# Reinforce PATH at the end for paranoia's sake
ENV PATH=/opt/miniconda/bin:/usr/local/cuda/bin:${PATH}

# Reset the frontend variable and start the SSH service
ENV DEBIAN_FRONTEND=
CMD ["/usr/sbin/sshd", "-D"]
