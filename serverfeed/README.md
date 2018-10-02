# serverfeed

Utility for feeding images to the network for training. Built on ZMQ sockets.

Two implementations.

* a larcv2-only setup, based on `ThreadFillerIO` in `larcv/app/ThreadIO` and `dataloader3` in `python/larcv/dataloader3.py`
* a larcv1 or larcv2 setup where user is simply provided `larcv::IOManager` (after entry loaded)

The second method is intended to be agnostic about if we are larcv 1 or larcv 2.  The user provides a function that

```
def load_data( io ):
 # note: io is an instance of the `larcv::IOManager` class found in both larcv 1 and larcv 2 frameworks
 # data = {} # we want to fill this dictionary if key=string name and  value=numpy_arrays
 # ...
 # profit!!
 return data
```

The two methods use different socket connections as well. The first methods
* PUSH (worker) and PULL (client)
* REP-REQ between client-server
* REP-REQ between client-worker


For dual-flow data tests show a transfer of about <200 ms per event (batchsize 4) with either method.

*NOTE* the first method, threadio, seems to have bugs (likely due to way I setup.)
Seems to be some kind of race condition where images in an event do no match with one anohter.

*NOTE* the second method, using `larcvworker` does not seem to have similar problems.


