FROM ubuntu:18.04
RUN apt update && apt upgrade -y
RUN apt install build-essential -y
RUN apt install git -y
RUN apt install python3 python3-pip -y
RUN apt install iverilog verilator gtkwave -y
RUN pip3 install jinja2 pyverilog veriloggen numpy onnx
RUN pip3 install torch torchvision
RUN pip3 install pytest pytest-pythonpath
RUN mkdir /home/nngen/
WORKDIR "/home/nngen"
RUN git clone https://github.com/NNgen/nngen.git
RUN cd nngen && python3 setup.py install && cd ../
