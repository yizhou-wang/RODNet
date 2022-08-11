# Use nvidia/cuda image
FROM gpuci/miniconda-cuda:10.2-devel-ubuntu18.04

# Copy
COPY ./install_rodnet.sh /install_rodnet.sh
RUN chmod +x ./install_rodnet.sh

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub

# install anacond
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git rsync nano && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


