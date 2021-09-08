import os,time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import ROOT as rt
from larflow import larflow
from ctypes import c_int

class larmatchDataset(torch.utils.data.Dataset):
    def __init__(self, filelist=None, filefolder=None, random_access=True, npairs=50000, verbose=False):
        """
        Parameters:
        """
        file_v = rt.std.vector("string")()
        if filelist is not None:
            if type(filelist) is not list:
                raise RuntimeError("filelist argument was not passed a list")
            for f in filelist:
                file_v.push_back(f)
        elif filefolder is not None:
            if not os.path.exists(filefolder):
                raise RuntimeError("filefolder argument points to location that does not exist")
            flist = os.listdir(filefolder)
            for f in flist:
                if ".root" not in f:
                    continue
                fpath = filefolder+"/"+f
                file_v.push_back(fpath)
        else:
            raise RuntimeError("must provide a filelist or a folder path")

        self.loaders = {"kps":larflow.keypoints.LoaderKeypointData(file_v),
                        "affinity":larflow.keypoints.LoaderAffinityField(file_v)}
        self.exclude_neg_examples = False
        self.loaders["kps"].exclude_false_triplets( self.exclude_neg_examples )        
        self.nentries = self.loaders["kps"].GetEntries()
        self.random_access = random_access

        self._current_entry  = 0
        self._nloaded        = 0
        self._verbose = False
        self.npairs = npairs
                                 

    def __getitem__(self, idx):

        ientry = self._current_entry
        data    = {"entry":ientry,
                   "tree_entry":int(ientry)%int(self.loaders["kps"].GetEntries())}
    

        if self._verbose:
            # if verbose, we'll output some timinginfo
            t_start = time.time()
            tio     = time.time()

        # get data from match trees            
        for name,loader in self.loaders.items():
            nbytes = loader.load_entry(data["tree_entry"])
            if self._verbose:
                print("nbytes: ",nbytes," for tree[",name,"] entry=",data['tree_entry'])

        if self._verbose:
            dtio = time.time()-tio

        nfilled = c_int()
        nfilled.value = 0
        
        # the sparse image comes from the KPS loader
        spdata_v    = [ self.loaders["kps"].triplet_v[0].make_sparse_image( p ) for p in range(3) ]
        # sample the possible spacepoint matches
        matchdata   = self.loaders["kps"].sample_data( self.npairs, nfilled, True )
        # get the particle affinity field data
        pafdata = self.loaders["affinity"].get_match_data( matchdata["matchtriplet"], self.exclude_neg_examples )
        
        # add the contents to the data dictionary
        data.update(matchdata)
        data.update(pafdata)

        # prepare data dictionary
        npts_per_plane = [0,0,0]
        batch_tot_per_plane = [0,0,0]
        batch_npts_per_plane = []
        # separate the sparse charge image matrix into coordinates and features (the charge)
        for p in range(3):
            data["coord_%d"%(p)] = spdata_v[p][:,0:2].astype( dtype=np.int32 )
            # reshape and scale feature (i.e. pixel intensities)            
            data["feat_%d"%(p)]  = np.clip( spdata_v[p][:,2].reshape( (spdata_v[p].shape[0], 1) )/40.0, 0, 10.0 )
            npts_per_plane[p] = spdata_v[p].shape[0]
            batch_tot_per_plane[p] += npts_per_plane[p]
        batch_npts_per_plane.append(npts_per_plane)
        # split the spacepoint match information into the 3-plane sparse matrxi indices 
        data["matchpairs"]     = matchdata["matchtriplet"][:,0:3].astype( dtype=np.long )
        data["larmatchlabels"] = matchdata["matchtriplet"][:,3].astype( dtype=np.long )
        data["npairs"]         = nfilled.value
        # resetting the topological weights for ssnet
        nboundary = np.sum(data["ssnet_top_weight"][data["ssnet_top_weight"]==10.0])
        nvertex   = np.sum(data["ssnet_top_weight"][data["ssnet_top_weight"]==100.0])        
        data["ssnet_top_weight"][ data["ssnet_top_weight"]==10.0 ]  = 2.0
        data["ssnet_top_weight"][ data["ssnet_top_weight"]==100.0 ] = 5.0

        self._nloaded += 1
        self._current_entry += 1
        
        if self._verbose:
            tottime = time.time()-t_start            
            print("[load larmatch kps single-batch sample]")
            print("  io time: %.3f secs"%(dtio))
            print("  tot time: %.3f secs"%(tottime))
            
        return data

    def __len__(self):
        return self.nentries

    def print_status(self):
        print("worker: entry=%d nloaded=%d"%(self._current_entry,self._nloaded))

    def collate_fn(batch):
        #print("[larmatchDataset::collate_fn] batch: ",type(batch)," len=",len(batch))
        return batch
    
            
if __name__ == "__main__":

    import time

    niter = 10
    batch_size = 4
    test = larmatchDataset( filefolder="/home/twongjirad/working/data/larmatch_training_data/", random_access=True )
    #test = larmatchDataset( filelist=["larmatchtriplet_ana_trainingdata_testfile.root"])
    print("NENTRIES: ",len(test))
    
    loader = torch.utils.data.DataLoader(test,batch_size=batch_size,collate_fn=larmatchDataset.collate_fn)

    start = time.time()
    for iiter in range(niter):
        batch = next(iter(loader))
        print("====================================")
        for ib,data in enumerate(batch):
            print("ITER[%d]:BATCH[%d]"%(iiter,ib))
            print(" keys: ",data.keys())
            for name,d in data.items():
                if type(d) is np.array:
                    print("  ",name,": ",d.shape)
                else:
                    print("  ",name,": ",type(d))
    end = time.time()
    elapsed = end-start
    sec_per_iter = elapsed/float(niter)
    print("sec per iter: ",sec_per_iter)
    
    loader.dataset.print_status()
