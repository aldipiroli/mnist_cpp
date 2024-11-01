# FROM tensorflow/tensorflow:latest-gpu
FROM nvcr.io/nvidia/tensorflow:24.10-tf2-py3

ENV DEBIAN_FRONTEND="noninteractive" \
	TERM="xterm"
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install apt packages
RUN apt-get update && apt-get install -y -qq --no-install-recommends  \
	build-essential \
	isort \
	meld \
	zsh \
	libsm6 \
	libxext6 \
	libxrender-dev \
	libcap-dev \
	libgl1-mesa-glx \
	libusb-1.0-0 \
	libglvnd-dev \
	libgl1-mesa-dev \
	libegl1-mesa-dev \
	libx11-6 \
	libgtk2.0-dev \
	zip \
	sudo \ 
	ninja-build \ 
	gettext \ 
	cmake \
	g++ \
	wget \
	unzip \
	curl \
	build-essential \
	librsvg2-dev \ 
	xclip\
	clang \
	clang-tidy \ 
	ca-certificates \
	software-properties-common \
	libopencv-dev \
	libopencv-core-dev \
	libopencv-imgproc-dev \
	libopencv-video-dev \
	libopencv-highgui-dev \
	libopencv-videoio-dev \
	libopencv-contrib-dev

# ----------- Install tensorflow for C (https://www.tensorflow.org/install/lang_c) -------------------------
RUN curl -fsSL -o /tmp/libtensorflow-gpu-linux-x86_64.tar.gz https://storage.googleapis.com/tensorflow/versions/2.18.0/libtensorflow-gpu-linux-x86_64.tar.gz
RUN tar -C /usr/local -xzf /tmp/libtensorflow-gpu-linux-x86_64.tar.gz
RUN ldconfig /usr/local/lib
RUN rm /tmp/libtensorflow-gpu-linux-x86_64.tar.gz


# --------- Install Buildifier -------------------
# RUN git clone https://github.com/bazelbuild/buildtools.git && cd buildtools/ && bazel build //buildifier

# ----------- Install neovim ------------ 
# compile neovim
RUN git clone https://github.com/neovim/neovim && cd neovim && make CMAKE_BUILD_TYPE=RelWithDebInfo && sudo make install

# Install Node
RUN curl -fsSL https://deb.nodesource.com/setup_23.x -o nodesource_setup.sh
RUN bash nodesource_setup.sh
RUN apt-get install -y nodejs && \
	rm -rf /var/lib/apt/lists/*

# ------- ohmyzsh-----------------
# Install ohmyzsh (source: https://github.com/deluan/zsh-in-docker)
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.2.1/zsh-in-docker.sh)" -- \
	-t robbyrussell \
	-p git -p ssh-agent -p fzf -p zsh-autosuggestions -p zsh-syntax-highlighting -p fast-syntax-highlighting 

# -------- Add USER -----------
# Source: https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
ARG USERNAME={USERNAME}
ARG USER_UID={USER_UID}
ARG USER_GID={USER_GID}

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
	&& useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
	#
	# [Optional] Add sudo support. Omit if you don't need to install software after connecting.
	&& apt-get update \
	&& apt-get install -y sudo \
	&& echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
	&& chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

CMD [ "zsh" ]

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
WORKDIR /home/$USERNAME/workspace/mnist_cpp
