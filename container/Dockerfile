FROM twongjirad/larbys-pytorch:6.12.04_cuda8.0_opencv3.2_pytorch0.4_larcv2nutufts

MAINTAINER taritree.wongjirad@tufts.edu

RUN  apt-get update && apt-get -y install vim emacs wget perl python python-pip python-dev libzmq-dev libzmq-dbg python-zmq python-zmq-dbg \
  	  mesa-common-dev freeglut3 freeglut3-dev libglew-dev libeigen3-dev && apt-get autoremove -y && apt-get clean -y \
     && pip install numpy \
     && pip install msgpack-numpy
