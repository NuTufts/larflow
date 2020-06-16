import os,sys
import ROOT as rt
#from larcv import larcv

def checkfile( filepath, separate_flows=False ):
    
    if separate_flows:
        ty2u = rt.TChain("sparseimg_sparsecropy2u_tree")
        ty2v = rt.TChain("sparseimg_sparsecropy2v_tree")

        ty2u.Add(filepath)
        ty2v.Add(filepath)
        try:
            nentriesy2u = ty2u.GetEntries()
            nentriesy2v = ty2v.GetEntries()
        except:
            nentriesy2u = 0
            nentriesy2v = 0

        if nentriesy2u<1 or nentriesy2v<1:
            return False
        if nentriesy2u != nentriesy2v:
            return False

    else:
        tdual = rt.TChain("sparseimg_sparsecropdual_tree")
        tdual.Add(filepath)
        try:
            nentries = tdual.GetEntries()
        except:
            nentries = 0

        if nentries<1:
            return False

    return True


if __name__ == "__main__":

    # set this to true to reset the files, or check the whole list
    # otherwise, will only check un-finished jobs
    checkinputlist = True

    outdir="/cluster/tufts/wongjiradlab/twongj01/ubdl/larflow/sparse_larflow/workdir"
    folders = os.listdir(outdir)

    inputlistname = "inputlists/mcc9mar_bnbcorsika.list"
    inputfile = open(inputlistname,'r')
    inputlist = inputfile.readlines()
    
    oklistname = "processedok.list"
    if checkinputlist:
        oklist = open(oklistname,'w')
    else:
        oklist = open(oklistname,'a')

    errlistname = "err_processing.list"
    errlist = open(errlistname,'w')

    rerunlistname = "rerun_processing.list"
    # read list
    rerunfile = open(rerunlistname,'r')
    rerunlist = rerunfile.readlines()
    rerunfile.close()

    # write next jobs to run here
    rerunfile = open("rerun_processing_tmp.list",'w')
    
    nrerun = 0
    flist = rerunlist
    if checkinputlist:
        flist = inputlist

    for iline,larcvname in enumerate(flist):
        if checkinputlist:
            ijob = iline
            larcvname = larcvname.strip()
        else:
            ijob = int(larcvname.split()[0].strip())
            larcvname = larcvname.split()[1].strip()

        workdir=outdir+"/cropped_sparsifyjobid%04d"%(ijob)
        sparseout = os.path.basename( larcvname ).replace("larcvtruth","sparsecroplarflow")
        sparsepath = workdir + "/" + sparseout
        
        if not os.path.exists(sparsepath):
            print sparsepath," does not exist"
            print >> rerunfile,ijob,larcvname.strip()
            nrerun += 1
            continue
            
        ok = checkfile(sparsepath, separate_flows=False)

        if ok:
            print "ok=",ok," : ",sparsepath
            print >> oklist,sparsepath.strip()
        else:
            print >> errlist,sparsepath.strip()
            print >> rerunfile,ijob,larcvname.strip()
            nrerun += 1


    print "Number to rerun: ",nrerun
    os.system("mv rerun_processing_tmp.list rerun_processing.list")
    
