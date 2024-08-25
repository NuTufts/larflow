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
train_data = uproot.concatenate(train_files,expressions=varlist+[label])
valid_data = uproot.concatenate(valid_files,expressions=varlist+[label])

print("made data. now convert to numpy arrays")

# convert to numpy, concat, then turn into DMatrix for xg boost
train_v = [ np.expand_dims( train_data[x].to_numpy(), axis=1  ) for x in varlist ]
train_X = np.concatenate( train_v, axis=1 )
train_Y = train_data[label].to_numpy()
print("training data shape: ",train_X.shape," ",train_Y.shape)

valid_v = [ np.expand_dims( valid_data[x].to_numpy(), axis=1  ) for x in varlist ]
valid_X = np.concatenate( valid_v, axis=1 )
valid_Y = valid_data[label].to_numpy()
print("validation data shape: ",valid_X.shape," ",valid_Y.shape)

dtrain = xgb.DMatrix(train_X, label=train_Y)

# create model instance
bst = xgb.XGBClassifier(max_depth=6, eta=0.3,
                        subsample=0.5,
                        n_estimators=1000,
                        objective='binary:logistic', eval_metric=['logloss',"error"])

evalset = [ (train_X, train_Y), (valid_X, valid_Y) ]

# fit model
print("training ... ")
bst.fit(train_X, train_Y, eval_set=evalset, verbose=True)

# get eval matrics
results = bst.evals_result()
# results is a dict
for k,v in results.items():
    print(k,": ",type(v))

print("save mode")
bst.save_model('test.model')

# make predictions
print("make predictions")
preds = bst.predict(valid_X)
# preds is a numpy array, dtype int64
print(preds.shape," ",preds.dtype)
print(preds[:10])

ncorrect = ( preds.astype(np.float64)*valid_Y.astype(np.float64) ).sum()
print("number correct: ",ncorrect)
print("accuracy: ",ncorrect/float(valid_Y.shape[0])*100.0,"%")




