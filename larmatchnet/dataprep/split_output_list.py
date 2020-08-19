import os,sys

inputlist = "kps_output.list"
stride = 20
nparts = 10
for p in xrange(6,nparts):
    listout = "kps_trainlist_p%02d.list"%(p)
    start = p*stride+1
    end   = (p+1)*stride
    os.system("sed -n %d,%dp %s > %s"%(start,end,inputlist,listout))
    
