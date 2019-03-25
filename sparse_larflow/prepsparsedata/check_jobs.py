import os,sys
import ROOT as rt
#from larcv import larcv

def checkfile( filepath ):
    t = rt.TChain("sparseimg_larflow_tree")
    t.Add(filepath)
    nentries = t.GetEntries()
    if nentries<20:
        return False
    return True


if __name__ == "__main__":

    outdir="/cluster/tufts/wongjiradlab/twongj01/ubdl/larflow/sparse_larflow/workdir"
    folders = os.listdir(outdir)

    inputlistname = "inputlists/mcc9mar_bnbcorsika.list"
    inputfile = open(inputlistname,'r')
    inputlist = inputfile.readlines()
    
    oklistname = "processedok.list"
    oklist = open(oklistname,'w')

    errlistname = "err_processing.list"
    errlist = open(errlistname,'w')

    rerunlistname = "rerun_processing.list"
    rerunlist = open(rerunlistname,'w')
    
    for ijob,larcvname in enumerate(inputlist):
        larcvname = larcvname.strip()
        workdir=outdir+"/sparsifyjobid%04d"%(ijob)
        sparseout = os.path.basename( larcvname ).replace("larcvtruth","sparselarflowy2u")
        sparsepath = workdir + "/" + sparseout
        
        if not os.path.exists(sparsepath):
            print sparsepath," does not exist"
            print >> rerunlist,ijob,larcvname.strip()
            continue
            
        ok = checkfile(sparsepath)

        if ok:
            print "ok=",ok," : ",sparsepath
            print >> oklist,sparsepath.strip()
        else:
            print >> errlist,sparsepath.strip()
            print >> rerunlist,ijob,larcvname.strip()
                
