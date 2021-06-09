FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt upgrade -y
RUN apt install build-essential -y
RUN apt install git -y
RUN apt install python3 python3-pip -y
RUN apt install iverilog verilator gtkwave -y
RUN pip3 install -U pip
RUN pip3 install pytest pytest-pythonpath
RUN pip3 install torch torchvision
RUN mkdir /home/nngen/
WORKDIR "/home/nngen"
RUN git clone https://github.com/NNgen/nngen.git
RUN cd nngen && python3 setup.py install && cd ../
