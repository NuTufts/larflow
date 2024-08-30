import os,sys
import uproot
import numpy as np
import xgboost as xgb

# make file list
data_dir="/cluster/tufts/wongjiradlabnu/nutufts/data/v3dev_shower_mcanalysis/mcc9_v40_NC_Pi0_run3b/larflowreco/ana/"

root_files = []
flist = os.popen("find %s -type f"%(data_dir))
for f in flist:
    f = f.strip()
    #print(f)
    root_files.append(f+":nushowerbuilder_mcana_tree")


# shorten list for debug
#root_files = root_files[:20]

nfiles = len(root_files)
itrain = int(0.8*nfiles)
ivalid = int(0.9*nfiles)

train_files = root_files[:itrain]
valid_files = root_files[itrain+1:ivalid]
test_files  = root_files[ivalid+1:]

varlist = ["recofragment_cosine",
           "recofragment_dist2vtx",
           "recofragment_impactpar",
           "recofragment_pixsum"]
label = "groundtruth_outcome"

print("attempt to make variable trees")
print("[enter] to start")
input()
#train_data = uproot.concatenate(train_files,expressions=varlist+[label])
valid_data = uproot.concatenate(valid_files,expressions=varlist+[label])
valid_v = [ np.expand_dims( valid_data[x].to_numpy(), axis=1  ) for x in varlist ]
valid_X = np.concatenate( valid_v, axis=1 )
valid_Y = valid_data[label].to_numpy()
print("validation data shape: ",valid_X.shape," ",valid_Y.shape)

bst = xgb.XGBClassifier()
bst.load_model('test.model')

valid_X_true = valid_X[ valid_Y==1 ]
print("valid_X_true shape: ",valid_X_true.shape)

valid_X_false = valid_X[ valid_Y==0 ]
print("valid_X_false.shape: ",valid_X_false.shape)

pred_Y_true  = bst.predict_proba(valid_X_true)
pred_Y_false = bst.predict_proba(valid_X_false)
print("pred_Y_true.shape (prob): ",pred_Y_true.shape)
#print(pred_Y_true)

acc_true  = float((pred_Y_true[:,1]>=0.5).sum())/float(pred_Y_true.shape[0])
acc_false = float((pred_Y_false[:,1]<0.5).sum())/float(pred_Y_false.shape[0])

print("number correct when true shower fragment: ",(pred_Y_true==1).sum()," out of ",pred_Y_true.shape)
print("number correct when false shower fragment: ",(pred_Y_false==0).sum()," out of ",pred_Y_false.shape)
print("true shower fragment accuracy: ",acc_true)
print("false shower fragment accuracy: ",acc_false)

prob_Y_full = bst.predict_proba(valid_X)


# save to a root file with uproot!

out = uproot.recreate("xgb_validout_v1.4.0.root")

# make a ttree
out["modelout"] = { "score":prob_Y_full[:,1], "label":valid_Y }
out.close()
print("done?")

