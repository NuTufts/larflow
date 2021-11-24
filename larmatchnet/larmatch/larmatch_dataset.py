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
                 random_access=True, npairs=50000, verbose=False):
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
        print("data: ",data.keys())
        
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
        nimgs = self.tree.coord_v.size()
        for p in range(nimgs):
            data["coord_%d"%(p)] = self.tree.coord_v.at(p).tonumpy()
            data["feat_%d"%(p)]  = self.tree.feat_v.at(p).tonumpy()

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
        print(batch)
        return batch
    
            
if __name__ == "__main__":

    import time

    niter = 1
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
