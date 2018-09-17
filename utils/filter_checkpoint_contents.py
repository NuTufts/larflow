import os,sys

# pytorch
import torch

def filter_checkpoint( checkpointfile, outputfile, map_location=None ):
    """ reduces state dictionary by moving elements to cpu and destroying optimizer contents"""
    checkpoint = torch.load( checkpointfile, map_location=map_location )
    del checkpoint["optimizer"]
    torch.save( checkpoint, outputfile )



chkpt = sys.argv[1]
map_location = {}
for x in range(0,6):
    map_location["cuda:%d"%(x)] = "cpu"
print map_location
filter_checkpoint( chkpt, "output_filtered_checkpoint.tar", map_location="cpu" )

print "made output_filtered_checkpoint.tar"
