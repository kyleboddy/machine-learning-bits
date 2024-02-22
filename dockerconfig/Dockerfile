# Use an NVIDIA CUDA base image
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Set non-interactive installation mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Set the workspace folder
WORKDIR /workspace

# Set MAX_JOBS environment variable for ninja/flash-attn issues
ENV MAX_JOBS=6

# Install SSH server, development tools, libraries, R, PHP, and database connectors
RUN apt-get update && apt-get install -y \
    openssh-server \
    sudo \
    vim \
    git \
    curl \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    r-base \
    python3-pip \
    python3-dev \
    build-essential \
    libopencv-dev \
    libssl-dev \
    zlib1g-dev \
    cudnn9-cuda-12 \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libswresample-dev \
    libswscale-dev \
    python3-venv \
    btop \
    nvtop \
    nano \
    net-tools \
    htop \
    zip \
    unzip \
    git-lfs \
    tmux \
    screen \
    glances \
    cmake \
    software-properties-common \
    && add-apt-repository ppa:ondrej/php \
    && apt-get update \
    && apt-get install -y \
    php8.0 \
    php8.0-cli \
    php8.0-common \
    php8.0-mysql \
    php8.0-xml \
    php8.0-mbstring \
    php8.0-curl \
    php8.0-gd \
    libmariadb-dev \
    libmariadb-dev-compat \
    ninja-build \
    jq \
    python3-tk \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Get nvcc in PATH
ENV PATH=/usr/local/cuda/bin:${PATH}

# Install NodeJS and all of its BS
RUN apt-get update && apt-get install -y ca-certificates curl gnupg \
    # Download and add the NodeSource GPG key
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /usr/share/keyrings/nodesource.gpg \
    # Set the Node.js major version
    && NODE_MAJOR=20 \
    # Add the NodeSource repository
    && echo "deb [signed-by=/usr/share/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" > /etc/apt/sources.list.d/nodesource.list \
    # Update the package list with the new repository and install Node.js
    && apt-get update && apt-get install nodejs -y \
    # Clean up
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set up SSH for remote connections
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd
EXPOSE 22

# Install Python packages and database connectors
RUN pip3 install --no-cache-dir \
    opencv-python \
    ipykernel \
    xgboost \
    lightgbm \
    pandas \
    scikit-learn \
    scipy \
    numpy \
    seaborn \
    matplotlib \
    build \
    nvitop \
    ffmpeg-python \
    Jinja2 \
    imageio[ffmpeg] \
    mysql-connector-python \
    PyMySQL \
    torch \
    transformers \
    datasets \
    deepspeed \
    bitsandbytes \
    apex \
    exllamav2 \
    tokenizers \
    sentencepiece \
    optuna \
    ultralytics \
    supervision \
    mariadb

# Install more Python packages
RUN pip3 install --no-cache-dir \
    dlib \
    tensorflow

# Install MORE Python packages
RUN pip3 install --no-cache-dir \
    face-recognition \
    flash-attn

# Install R packages and database connectors
RUN R -e "install.packages(c('ggplot2', 'dplyr', 'tidyr', 'shiny', 'caret', 'randomForest', 'rmarkdown', 'keras', 'tensorflow', 'EBImage', 'imager', 'xgboost', 'lightgbm', 'h2o', 'RMySQL', 'RMariaDB'), repos='http://cran.rstudio.com/')"

# Reset the frontend variable to its default state
ENV DEBIAN_FRONTEND=

# Get nvcc in PATH
ENV PATH=/usr/local/cuda/bin:${PATH}

# Start the SSH service
CMD ["/usr/sbin/sshd", "-D"]
