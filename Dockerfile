FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

RUN apt-get update \
 && apt-get install -y vim git build-essential ffmpeg\
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# その後、キャッシュをクリアし、不要なファイルを削除してイメージのサイズを減らします。
# CC is 90 for H100
ARG TCNN_CUDA_ARCHITECTURES=90
# ninja（高速なビルドシステム）
RUN pip install ninja "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

WORKDIR /root

RUN apt-get update \
 && apt-get install -y tmux \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip install gsplat \
                mlflow 

COPY requirements.txt /root
RUN pip install -r requirements.txt


RUN apt-get update && apt-get install -y openssh-server \
    && mkdir /var/run/sshd \
    && echo 'root:123' | chpasswd \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/PermitUserEnvironment no/PermitUserEnvironment yes/' /etc/ssh/sshd_config

# 環境変数の設定を.bashrcに追加
RUN echo "export PATH=${PATH}" >> /root/.bashrc \
 && echo "export PATH=${PATH}" >> /root/.profile


    # sed -i 's/a/b'でaをbに置き換える

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]