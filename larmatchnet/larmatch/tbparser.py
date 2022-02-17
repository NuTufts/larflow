import os,sys,array
from math import sqrt
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import ROOT as rt
    
def extract_values(logdir,filename,nave=100):
    ploss_eventfiles = os.popen("find %s/ -name *.out.* | grep %s"%(logdir,filename))
    loss_eventfiles = ploss_eventfiles.readlines()

    dat = []
    for l in loss_eventfiles:
        print("parsing: ",l.strip())
        logpath = l.strip()
        for summary in summary_iterator(logpath):
            #print(summary)
            for v in summary.summary.value:
                #print(v.tag,": ",summary.step,v.simple_value)
                dat.append( (summary.step, v.simple_value) )

    print("extracted ",len(dat)," points")
    dat.sort()
    print("first: ",dat[0]," last: ",dat[-1])
    npts = int(len(dat)/nave)
    g = rt.TGraphErrors( npts )
    for n in range(npts):
        valave = 0.0
        xx = 0.0
        stepave = 0.0
        nused = 0.0
        for (step,val) in dat[n*nave:(n+1)*nave]:
            valave += val
            xx += val*val
            stepave += step
            nused += 1.0
        valave /= float(nused)
        xx /= float(nused)
        stepave /= float(nused)
        sig = sqrt( xx - valave*valave )
        g.SetPoint(n,stepave,valave)
        g.SetPointError(n,0.0,sig)
    
    return dat,g

if __name__ == "__main__":


    rundir="arxiv/larmatch_lossweights_nossnet/runs/"
    tfile = rt.TFile("extracted_tensorboard.root","recreate")
    nave_train = 50
    nave_valid = 50
    for task in ["kp","lm"]:
        train_loss,tloss = extract_values(rundir,"data_train_loss_"+task,nave=nave_train)
        tloss.Write("gtrain_loss_"+task)

        valid_loss,vloss = extract_values(rundir,"data_valid_loss_"+task,nave=nave_valid)
        vloss.Write("gvalid_loss_"+task)

    for c in ["lm_all","lm_neg","lm_pos"]:
        train_posexamples,train_posex = extract_values(rundir,"data_train_larmatch_accuracy_"+c,nave=nave_train)
        train_posex.Write("gtrain_lm_acc_"+c)
        valid_posexamples,valid_posex = extract_values(rundir,"data_valid_larmatch_accuracy_"+c,nave=nave_valid)
        valid_posex.Write("gvalid_lm_acc_"+c)

    for c in ["kp_nu","kp_trackstart","kp_trackend","kp_shower","kp_michel","kp_delta"]:
        train_posexamples,train_posex = extract_values(rundir,"data_train_kp_accuracy_"+c,nave=nave_train)
        train_posex.Write("gtrain_kp_acc_"+c)
        valid_posexamples,valid_posex = extract_values(rundir,"data_valid_kp_accuracy_"+c,nave=nave_valid)
        valid_posex.Write("gvalid_kp_acc_"+c)
    
