import os,sys,array
from math import sqrt
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import ROOT as rt
    
def extract_values(logdir,filename,nave=100):
    logdir="runs"
    ploss_eventfiles = os.popen("find %s/ -name *.out.* | grep %s"%(logdir,filename))
    loss_eventfiles = ploss_eventfiles.readlines()

    dat = []
    for l in loss_eventfiles:
        print "parsing: ",l.strip()
        logpath = l.strip()
        for summary in summary_iterator(logpath):
            for v in summary.summary.value:
                #print v.tag,v.simple_value
                dat.append( (summary.step, v.simple_value) )

    print "extracted ",len(dat)," points"
    dat.sort()
    npts = int(len(dat)/nave)
    g = rt.TGraphErrors( npts )
    for n in xrange(npts):
        valave = 0.0
        xx = 0.0
        for (step,val) in dat[n*nave:(n+1)*nave]:
            valave += val
            xx += val*val
        valave /= float(nave)
        xx /= float(nave)
        sig = sqrt( xx - valave*valave )
        g.SetPoint(n,n*nave,valave)
        g.SetPointError(n,0.0,sig)
    
    return dat,g

if __name__ == "__main__":


    tfile = rt.TFile("extracted_tensorboard.root","recreate")

    train_loss,tloss = extract_values("runs","data_train_loss_total")
    tloss.Write("gtrain_loss")

    valid_loss,vloss = extract_values("runs","data_valid_loss_total")
    vloss.Write("gvalid_loss")

    train_posexamples,train_posex = extract_values("runs","data_train_accuracy_pos_correct")
    train_posex.Write("gtrain_acc_pos_examples")
    valid_posexamples,valid_posex = extract_values("runs","data_valid_accuracy_pos_correct")
    valid_posex.Write("gvalid_acc_pos_examples")

    train_negexamples,train_negex = extract_values("runs","data_train_accuracy_neg_correct")
    train_negex.Write("gtrain_acc_neg_examples")
    valid_negexamples,valid_negex = extract_values("runs","data_valid_accuracy_neg_correct")
    valid_negex.Write("gvalid_acc_neg_examples")
    
