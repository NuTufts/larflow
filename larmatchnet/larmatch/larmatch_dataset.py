import os,time,copy,sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import ROOT as rt
from larflow import larflow
from ctypes import c_int


class larmatchDataset(torch.utils.data.Dataset):
    def __init__(self, filelist=None, filefolder=None, txtfile=None,
                 random_access=True, npairs=None, load_truth=False,
                 triplet_limit=2500000, normalize_inputs=True,
                 num_triplet_samples=None,
                 return_constant_sample=False,
                 use_keypoint_sampler=False,
                 verbose=False):
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
        elif txtfile is not None:
            f = open(txtfile,'r')
            ll = f.readlines()
            for l in ll:
                l = l.strip()
                if os.path.exists(l):
                    file_v.push_back(l)
            f.close()
        else:
            raise RuntimeError("must provide a list of paths (filelist), folder path (filefolder), or textfile with paths (txtfile)")

        self.tree = rt.TChain("larmatchtrainingdata")
        for ifile in range(file_v.size()):
            self.tree.Add( file_v.at(ifile) )
        self.nentries = self.tree.GetEntries()
        self.load_truth = load_truth

        # store parameters
        self.random_access = random_access
        self.partition_index = 0
        self.num_partitions = 1
        self.start_index = 0
        self.end_index = self.nentries

        self._current_entry  = self.start_index
        self._nloaded        = 0
        self._verbose = False
        self._random_access = random_access
        self._random_entry_list = None
        self._max_num_tries = 10
        self._triplet_limit = triplet_limit
        self._normalize_inputs = normalize_inputs
        self._num_triplet_samples = num_triplet_samples
        self._return_constant_sample = return_constant_sample
        self._constant_sample = None
        self._use_keypoint_sampler = use_keypoint_sampler
        if self._random_access:
            self._rng = np.random.default_rng(None)
            self._random_entry_list = self._rng.choice( self.nentries, size=self.nentries )

        self.triplet_array_names = ["matchtriplet_v",
                                    "larmatch_weight",
                                    "larmatch_label",
                                    "ssnet_truth",
                                    "ssnet_top_weight",
                                    "ssnet_class_weight",
                                    "keypoint_truth",
                                    "keypoint_weight"]
                                 

    def __getitem__(self, idx):
        
        if self._return_constant_sample and self._constant_sample is not None:
            return copy.deepcopy(self._constant_sample)
        
        worker_info = torch.utils.data.get_worker_info()

        okentry = False
        num_tries = 0

        while not okentry:

            if self._random_access and self._current_entry>=self.nentries:
                # if random access and we've used all possible indices we reset the shuffled index list
                print("reset shuffled indices list")
                self._random_entry_list = self._rng.choice( self.nentries, size=self.nentries )
                self._current_entry = 0 # current entry in shuffled list

            ientry = int(self._current_entry)
        
            if self._random_access:
                # if random access, get next entry in shuffled index list
                data    = {"entry":ientry,
                           "tree_entry":self._random_entry_list[ientry]}
            else:
                # else just take the next entry
                data    = {"entry":ientry,
                           "tree_entry":int(ientry)%int(self.nentries)}

            if self._num_triplet_samples is not None:
                # we want to sample
                if not self._use_keypoint_sampler:
                    # unguided sampling
                    okentry = self.get_data_from_tree( data, sample_spacepoints=True )
                else:
                    # guided sampling by keypoint truth
                    okentry = self.get_data_from_tree( data, sample_spacepoints=False )
                    self.keypoint_sampler( data, self._num_triplet_samples, int(self._num_triplet_samples/2) )
            else:
                okentry = self.get_data_from_tree( data, sample_spacepoints=False )

            # increment the entry index
            self._current_entry += 1
                
            # increment the number of attempts we've made
            num_tries += 1
            if num_tries>=self._max_num_tries:
                raise RuntimeError("Tried the max num of times (%d) to get an acceptable piece of data"%(num_tries))

        # xlist = np.unique( data["voxcoord"], axis=0, return_counts=True )
        # indexlist = xlist[0]
        # counts = xlist[-1]
        # hasdupe = False
        # for i in range(indexlist.shape[0]):
        #     if counts[i]>1:
        #         print(i," ",indexlist[i,:]," counts=",counts[i])
        #         hasdupe = True
        # if hasdupe:
        #     raise("[larvoxel_dataset::__getitem__] Dupe introduced somehow in batch-index=%d"%(ibatch)," arr=",data["voxcoord"].shape)

        self._nloaded += 1
        #print("data lodaded ntries: ",num_tries)
        #sys.stdout.flush()
        #print("data: ",data.keys())
        if self._return_constant_sample and self._constant_sample is None:
            self._constant_sample = data
        
        return copy.deepcopy(data)


    def __len__(self):
        return self.nentries

    # get data from match trees
    def get_data_from_tree( self, data, sample_spacepoints=False ):

        if self._verbose:
            dtio = time.time()-tio
        
        nbytes = self.tree.GetEntry(data["tree_entry"])
        if self._verbose:
            print("nbytes: ",nbytes," for tree[",name,"] entry=",data['tree_entry'])

        #print("get_data_dict_from_voxelarray_file: ",self.voxeldata_tree.coord_v.at(0).tonumpy().shape)
        # get the image data
        nimgs = self.tree.coord_v.size()
        for p in range(nimgs):
            data["coord_%d"%(p)] = self.tree.coord_v.at(p).tonumpy()
            data["feat_%d"%(p)]  = np.expand_dims( self.tree.feat_v.at(p).tonumpy(), 1 )
            # renormalize feats
            if self._normalize_inputs:
                data["feat_%d"%(p)] /= 50.0
                data["feat_%d"%(p)] = np.clip( data["feat_%d"%(p)], 0, 10.0 )
        # get the 2D-3D correspondence data
        data["matchtriplet_v"] = self.tree.matchtriplet_v.at(0).tonumpy()
        ntriplets = data["matchtriplet_v"].shape[0]
        
        if ntriplets>self._triplet_limit:
            print("num triplets above the limit: ",data["matchtriplet_v"].shape)
            data = None
            return False

        print("larmatch_label: ",self.tree.larmatch_label_v.at(0).tonumpy().shape)
        if sample_spacepoints:
            sample = np.arange(ntriplets)
            np.random.shuffle(sample)            
            if self._num_triplet_samples<ntriplets:
                sample = sample[:self._num_triplet_samples]
            elif self._num_triplet_samples>ntriplets:
                nfilled = 0
                while nfilled<self._num_triplet_samples:
                    x = np.arange(ntriplets)
                    np.random.shuffle(x) 
                    nadd = ntriplets
                    if int(nfilled+ntriplets)>self._num_triplet_samples:
                        nadd = self._num_triplet_samples-nfilled
                    sample[nfilled:nfilled+nadd] = x[:nadd]
                    nfilled += nadd

            data["matchtriplet_v"] = data["matchtriplet_v"][sample[:self._num_triplet_samples],:]
            data["ntriplets"] = int(self._num_triplet_samples)

            if self.load_truth:
                data["larmatch_truth"]  = self.tree.larmatch_truth_v.at(0).tonumpy()[sample]
                data["larmatch_weight"] = self.tree.larmatch_weight_v.at(0).tonumpy()[sample]
                data["larmatch_label"]  = self.tree.larmatch_label_v.at(0).tonumpy()[sample]
                data["ssnet_truth"]   = self.tree.ssnet_truth_v.at(0).tonumpy().astype(np.long)[sample]
                data["ssnet_top_weight"] = self.tree.ssnet_top_weight_v.at(0).tonumpy()[sample]
                data["ssnet_class_weight"] = self.tree.ssnet_class_weight_v.at(0).tonumpy()[sample]
                data["keypoint_truth"]  = np.transpose( self.tree.kp_truth_v.at(0).tonumpy()[sample,:], (1,0) )
                data["keypoint_weight"] = np.transpose( self.tree.kp_weight_v.at(0).tonumpy()[sample,:], (1,0) )
                print("post sampling: ",data["larmatch_label"].shape)
        else:
            data["ntriplets"] = int(ntriplets)
            if self.load_truth:
                data["larmatch_truth"]  = self.tree.larmatch_truth_v.at(0).tonumpy()
                data["larmatch_weight"] = self.tree.larmatch_weight_v.at(0).tonumpy()
                data["larmatch_label"]  = self.tree.larmatch_label_v.at(0).tonumpy()                
                data["ssnet_truth"]   = self.tree.ssnet_truth_v.at(0).tonumpy().astype(np.long)
                data["ssnet_top_weight"] = self.tree.ssnet_top_weight_v.at(0).tonumpy()
                data["ssnet_class_weight"] = self.tree.ssnet_class_weight_v.at(0).tonumpy()
                data["keypoint_truth"]  = np.transpose( self.tree.kp_truth_v.at(0).tonumpy(), (1,0) )
                data["keypoint_weight"] = np.transpose( self.tree.kp_weight_v.at(0).tonumpy(), (1,0) )

        print("ntriplets: ",data["ntriplets"])
        
        if self._verbose:
            tottime = time.time()-t_start            
            print("[larmatchDataset::get_data_tree entry=%d loaded]"%(data["tree_entry"]))
            print("  io time: %.3f secs"%(dtio))
            print("  tot time: %.3f secs"%(tottime))
            
        return True


    def print_status(self):
        print("worker: entry=%d nloaded=%d"%(self._current_entry,self._nloaded))

    def set_partition(self,partition_index,num_partitions):
        self.partition_index = partition_index
        self.num_partitions = num_partitions
        self.start_index = int(self.partition_index*self.nentries)/int(self.num_partitions)
        self.end_index   = int((self.partition_index+1)*self.nentries)/int(self.num_partitions)
        self._current_entry = self.start_index

    def collate_fn(batch):
        #print("[larmatchDataset::collate_fn] batch: ",type(batch)," len=",len(batch))
        #print(batch)
        return batch
    

    def keypoint_sampler(self,data, NALLSAMPLE, NTOTSAMPLE):        
        """
        Takes the output of 'get_data_from_tree' and downsamples to emphasize keypoint spacepoints
        """
        # Keypoint sampler
        idxlist = np.arange( len(data["matchtriplet_v"]) )
        kptruth = data["keypoint_truth"]

        # positive sample index
        sampleidx = np.zeros(NTOTSAMPLE,dtype=np.int64)

        # all sample index
        kpallidx = np.zeros(NALLSAMPLE,dtype=np.int64)
        kpweightall = np.zeros( (6,NALLSAMPLE) )

        NUSED = 0
        class_limits = {0:2000,1:5000,2:5000,3:2000,4:2000,5:5000}

        for c in [0,3,4,1,2,5]:
            #abovezero = (kptruth[c]>0).sum()
            #above1sig = (kptruth[c]>0.66).sum()            
            #above2sig = (kptruth[c]>0.10).sum()
            above3sig = (kptruth[c]>0.05).sum()
            #print("class[",c,"] : ",abovezero," ",above3sig," ",above2sig," ",above1sig)
            classidx = idxlist[ kptruth[c]>0.05 ]
            fill_limit = class_limits[c]
            if c==5:
                fill_limit = np.minimum( above3sig, NTOTSAMPLE-NUSED )
            if above3sig<fill_limit:
                nfilled = classidx.shape[0]
                if NUSED+nfilled>NTOTSAMPLE:
                    nfilled = NTOTSAMPLE-NUSED
            else:
                np.random.shuffle(classidx)
                nfilled = class_limits[c]
                if NUSED+nfilled>NTOTSAMPLE:
                    nfilled = NTOTSAMPLE-NUSED
            if nfilled>classidx.shape[0]:
                nfilled = classidx.shape[0]
            #print("class[",c,"] abovezero=",abovezero," filled ",nfilled," NUSED=",NUSED)
            sampleidx[NUSED:NUSED+nfilled][:] = classidx[:nfilled]
            if nfilled>0:
                kpweightall[c,NUSED:NUSED+nfilled] = 0.5/nfilled
                kpweightall[c,0:NUSED] = 0.5/(NALLSAMPLE-nfilled)
                kpweightall[c,NUSED+nfilled:] = 0.5/(NALLSAMPLE-nfilled)
            else:
                kpweightall[c,:] = 1.0/float(NALLSAMPLE)
        
                #print("class[",c,"] abovezero=",abovezero," filled ",nfilled," NUSED=",NUSED)
                #print("class[",c,"] minweight=",np.min(kpweightall[c])," maxweight=",np.max(kpweightall[c]), 
                #      " sum=",np.sum(kpweightall[c]))
            NUSED += nfilled

        kpallidx[:NUSED] = sampleidx[:NUSED]
        if NALLSAMPLE-NUSED>0:
            np.random.shuffle( idxlist )
            kpallidx[NUSED:] = idxlist[:NALLSAMPLE-NUSED]

        # downsample all tensors
        data["matchtriplet_v"]  = data["matchtriplet_v"][kpallidx,:]
        data["larmatch_truth"]  = data["larmatch_truth"][kpallidx]
        data["larmatch_weight"] = data["larmatch_weight"][kpallidx]
        data["ssnet_truth"]     = data["ssnet_truth"][kpallidx]
        data["ssnet_top_weight"]   = data["ssnet_top_weight"][kpallidx]
        data["ssnet_class_weight"] = data["ssnet_class_weight"][kpallidx]
        data["keypoint_truth"]     = data["keypoint_truth"][:,kpallidx[:]]
        data["keypoint_weight"]    = kpweightall
        return

            
                
    
if __name__ == "__main__":

    """
    Example of loading and reading the data.

    Branches in the ROOT tree we are retrieveing

*............................................................................*
*Br    4 :coord_v   : Int_t coord_v_                                         *
*Entries :        5 : Total  Size=       2960 bytes  File Size  =        130 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.06     *
*............................................................................*
*Br    5 :coord_v.ndims : Int_t ndims[coord_v_]                              *
*Entries :        5 : Total  Size=        785 bytes  File Size  =        136 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.35     *
*............................................................................*
*Br    6 :coord_v.shape : vector<int> shape[coord_v_]                        *
*Entries :        5 : Total  Size=        935 bytes  File Size  =        223 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.50     *
*............................................................................*
*Br    7 :coord_v.data : vector<int> data[coord_v_]                          *
*Entries :        5 : Total  Size=    4476222 bytes  File Size  =    1261439 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   3.55     *
*............................................................................*
*Br    8 :feat_v    : Int_t feat_v_                                          *
*Entries :        5 : Total  Size=       2957 bytes  File Size  =        129 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.06     *
*............................................................................*
*Br    9 :feat_v.ndims : Int_t ndims[feat_v_]                                *
*Entries :        5 : Total  Size=        778 bytes  File Size  =        135 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.36     *
*............................................................................*
*Br   10 :feat_v.shape : vector<int> shape[feat_v_]                          *
*Entries :        5 : Total  Size=        868 bytes  File Size  =        201 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.36     *
*............................................................................*
*Br   11 :feat_v.data : vector<float> data[feat_v_]                          *
*Entries :        5 : Total  Size=    2238711 bytes  File Size  =     986523 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   2.27     *
*............................................................................*
*Br   12 :matchtriplet_v : Int_t matchtriplet_v_                             *
*Entries :        5 : Total  Size=       3093 bytes  File Size  =        137 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.06     *
*............................................................................*
*Br   13 :matchtriplet_v.ndims : Int_t ndims[matchtriplet_v_]                *
*Entries :        5 : Total  Size=        794 bytes  File Size  =        143 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.06     *
*............................................................................*
*Br   14 :matchtriplet_v.shape : vector<int> shape[matchtriplet_v_]          *
*Entries :        5 : Total  Size=        864 bytes  File Size  =        177 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.25     *
*............................................................................*
*Br   15 :matchtriplet_v.data : vector<int> data[matchtriplet_v_]            *
*Entries :        5 : Total  Size=   56939279 bytes  File Size  =   18744441 *
*Baskets :        5 : Basket Size=      32000 bytes  Compression=   3.04     *
*............................................................................*
*Br   16 :larmatch_truth_v : Int_t larmatch_truth_v_                         *
*Entries :        5 : Total  Size=       3131 bytes  File Size  =        136 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br   17 :larmatch_truth_v.ndims : Int_t ndims[larmatch_truth_v_]            *
*Entries :        5 : Total  Size=        788 bytes  File Size  =        132 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.01     *
*............................................................................*
*Br   18 :larmatch_truth_v.shape : vector<int> shape[larmatch_truth_v_]      *
*Entries :        5 : Total  Size=        818 bytes  File Size  =        150 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   19 :larmatch_truth_v.data : vector<int> data[larmatch_truth_v_]        *
*Entries :        5 : Total  Size=        813 bytes  File Size  =        149 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   20 :larmatch_weight_v : Int_t larmatch_weight_v_                       *
*Entries :        5 : Total  Size=       3166 bytes  File Size  =        137 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br   21 :larmatch_weight_v.ndims : Int_t ndims[larmatch_weight_v_]          *
*Entries :        5 : Total  Size=        795 bytes  File Size  =        133 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.01     *
*............................................................................*
*Br   22 :larmatch_weight_v.shape : vector<int> shape[larmatch_weight_v_]    *
*Entries :        5 : Total  Size=        825 bytes  File Size  =        151 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   23 :larmatch_weight_v.data : vector<float> data[larmatch_weight_v_]    *
*Entries :        5 : Total  Size=        820 bytes  File Size  =        150 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   24 :ssnet_truth_v : Int_t ssnet_truth_v_                               *
*Entries :        5 : Total  Size=       3074 bytes  File Size  =        133 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br   25 :ssnet_truth_v.ndims : Int_t ndims[ssnet_truth_v_]                  *
*Entries :        5 : Total  Size=        767 bytes  File Size  =        129 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.01     *
*............................................................................*
*Br   26 :ssnet_truth_v.shape : vector<int> shape[ssnet_truth_v_]            *
*Entries :        5 : Total  Size=        797 bytes  File Size  =        147 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   27 :ssnet_truth_v.data : vector<int> data[ssnet_truth_v_]              *
*Entries :        5 : Total  Size=        792 bytes  File Size  =        146 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   28 :ssnet_weight_v : Int_t ssnet_weight_v_                             *
*Entries :        5 : Total  Size=       3109 bytes  File Size  =        134 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br   29 :ssnet_weight_v.ndims : Int_t ndims[ssnet_weight_v_]                *
*Entries :        5 : Total  Size=        774 bytes  File Size  =        130 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.01     *
*............................................................................*
*Br   30 :ssnet_weight_v.shape : vector<int> shape[ssnet_weight_v_]          *
*Entries :        5 : Total  Size=        804 bytes  File Size  =        148 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   31 :ssnet_weight_v.data : vector<float> data[ssnet_weight_v_]          *
*Entries :        5 : Total  Size=        799 bytes  File Size  =        147 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   32 :kp_truth_v : Int_t kp_truth_v_                                     *
*Entries :        5 : Total  Size=       3033 bytes  File Size  =        130 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br   33 :kp_truth_v.ndims : Int_t ndims[kp_truth_v_]                        *
*Entries :        5 : Total  Size=        746 bytes  File Size  =        126 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.01     *
*............................................................................*
*Br   34 :kp_truth_v.shape : vector<int> shape[kp_truth_v_]                  *
*Entries :        5 : Total  Size=        776 bytes  File Size  =        144 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   35 :kp_truth_v.data : vector<float> data[kp_truth_v_]                  *
*Entries :        5 : Total  Size=        771 bytes  File Size  =        143 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   36 :kp_weight_v : Int_t kp_weight_v_                                   *
*Entries :        5 : Total  Size=       3052 bytes  File Size  =        131 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.08     *
*............................................................................*
*Br   37 :kp_weight_v.ndims : Int_t ndims[kp_weight_v_]                      *
*Entries :        5 : Total  Size=        753 bytes  File Size  =        127 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.01     *
*............................................................................*
*Br   38 :kp_weight_v.shape : vector<int> shape[kp_weight_v_]                *
*Entries :        5 : Total  Size=        783 bytes  File Size  =        145 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br   39 :kp_weight_v.data : vector<float> data[kp_weight_v_]                *
*Entries :        5 : Total  Size=        778 bytes  File Size  =        144 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*

    """
    
    import time
    from larcv import larcv
    larcv.load_pyutil()
    larcv.SetPyUtil()

    niter = 10
    batch_size = 4
    test = larmatchDataset( filelist=["test.root"])
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
                if type(d) is np.ndarray:
                    print("  ",name,"-[array]: ",d.shape)
                else:
                    print("  ",name,"-[non-array]: ",type(d))
            print(data['coord_0'].shape)
            print(np.unique(data['coord_0'],axis=0).shape)
    end = time.time()
    elapsed = end-start
    sec_per_iter = elapsed/float(niter)
    print("sec per iter: ",sec_per_iter)
    loader.dataset.print_status()

    
