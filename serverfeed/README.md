# serverfeed

Utility for feeding images to the network for training. Built on ZMQ sockets.

Basically, a client class gets images from a ZMQ PULL socket.
The data comes from N workers reading larcv files and sending the images through a ZMQ PUSH socket.

For dual-flow data tests show a transfer of about 200 ms per event (batchsize 4).

Right now tailored for dual-flow training. Later, will try to generalize for all larcv-based feeding.


