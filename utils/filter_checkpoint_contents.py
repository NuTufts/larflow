import os,sys

# pytorch
import torch

def filter_checkpoint( checkpointfile, outputfile, map_location=None ):
    """ reduces state dictionary by moving elements to cpu and destroying optimizer contents"""
    checkpoint = torch.load( checkpointfile, map_location=map_location )
    del checkpoint["optimizer"]
    torch.save( checkpoint, outputfile )



chkpt = sys.argv[1]
filter_checkpoint( chkpt, "output_filtered_checkpoint.tar", map_location={"cuda:0":"cpu"} )

print "made output_filtered_checkpoint.tar"
