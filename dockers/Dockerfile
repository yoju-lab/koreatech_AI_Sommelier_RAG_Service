FROM python:3.11

# 기본 도구 설치
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://astral.sh/uv/install.sh | sh

# 작업 디렉토리 설정
WORKDIR /apps

# Clone the Git repository. Here we dynamically specify the repository name using the variable defined earlier.
RUN git clone https://github.com/yoju-lab/koreatech_LangChain_withRags koreatech_LangChain_withRags

# Changes the working directory to /app/${REPO_NAME}. This uses the variable to dynamically set the directory path.
WORKDIR /apps/koreatech_LangChain_withRags

RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir -r ./requirements.txt

RUN rm -rf .git
