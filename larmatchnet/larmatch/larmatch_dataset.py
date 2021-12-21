import os,time,copy
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
        if self._random_access:
            self._rng = np.random.default_rng(None)
            self._random_entry_list = self._rng.choice( self.nentries, size=self.nentries )        
                                 

    def __getitem__(self, idx):
        

        
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

            self.get_data_from_tree( data )

            # increment the entry index
            self._current_entry += 1
                
            # increment the number of attempts we've made
            num_tries += 1
            if num_tries>=self._max_num_tries:
                raise RuntimeError("Tried the max num of times (%d) to get an acceptable piece of data"%(num_tries))
            if data is None:
                okentry = False
            else:
                okentry = True

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
        #print("data: ",data.keys())
        
        return copy.deepcopy(data)


    def __len__(self):
        return self.nentries

    # get data from match trees
    def get_data_from_tree( self, data ):

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
        # get the 2D-3D correspondence data
        data["matchtriplet_v"] = self.tree.matchtriplet_v.at(0).tonumpy()

        if self.load_truth:
            data["larmatch_truth"]  = self.tree.larmatch_truth_v.at(0).tonumpy()
            data["larmatch_weight"] = self.tree.larmatch_weight_v.at(0).tonumpy()
            data["ssnet_truth"]   = self.tree.ssnet_truth_v.at(0).tonumpy().astype(np.long)
            data["ssnet_weight"] = self.tree.ssnet_top_weight_v.at(0).tonumpy()
            data["ssnet_weight"] *= self.tree.ssnet_class_weight_v.at(0).tonumpy()
            data["keypoint_truth"]  = np.transpose( self.tree.kp_truth_v.at(0).tonumpy(), (1,0) )
            data["keypoint_weight"] = np.transpose( self.tree.kp_weight_v.at(0).tonumpy(), (1,0) )
        
        if self._verbose:
            tottime = time.time()-t_start            
            print("[larmatchDataset::get_data_tree entry=%d loaded]"%(data["tree_entry"]))
            print("  io time: %.3f secs"%(dtio))
            print("  tot time: %.3f secs"%(tottime))
            
        return None


    def print_status(self):
        print("worker: entry=%d nloaded=%d"%(self._current_entry,self._nloaded))

    def set_partition(self,partition_index,num_partitions):
        self.partition_index = partition_index
        self.num_partitions = num_partitions
        self.start_index = int(self.partition_index*self.nentries)/int(self.num_partitions)
        self.end_index   = int((self.partition_index+1)*self.nentries)/int(self.num_partitions)
        self._current_entry = self.start_index

    def collate_fn(batch):
        print("[larmatchDataset::collate_fn] batch: ",type(batch)," len=",len(batch))
        #print(batch)
        return batch
    
            
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
    test = larmatchDataset( filelist=["temp.root"])
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
            print(data['coord_0'].shape)
            print(np.unique(data['coord_0'],axis=0).shape)
    end = time.time()
    elapsed = end-start
    sec_per_iter = elapsed/float(niter)
    print("sec per iter: ",sec_per_iter)
    
    loader.dataset.print_status()
