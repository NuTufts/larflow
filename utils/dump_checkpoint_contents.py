import os,sys

# pytorch
import torch

def unpack_checkpoint( checkpointfile, map_location=None ):
    checkpoint = torch.load( checkpointfile, map_location=map_location )
    print type(checkpoint)
    nelem = 0
    for k,v in checkpoint.items():
        print k,type(v)
        if k=="state_dict":
            print "STATE_DICT CONTENTS"
            for s,t in checkpoint[k].items():
                print "  ",s,type(t)
                if type(t) is torch.Tensor:
                    print "     grad=",t.grad
                    print "     shape=",t.shape
                    nelem += reduce( lambda x,y: x*y, t.shape )
        elif k=="optimizer":
            print "OPTIMIZER CONTENTS"
            for s,t in checkpoint[k].items():
                print "  ",s,type(t)
                if s=="state":
                    for u,v in t.items():
                        print "      ",u,type(v)
    print "number of model elements: ",nelem
    print "mem of model elements: ",nelem*8/1e9," GB"
                



chkpt = sys.argv[1]
map_location = {}
for x in range(0,6):
    map_location["cuda:%d"%(x)] = "cpu"
print map_location
unpack_checkpoint( chkpt, map_location=map_location )
raw_input()
