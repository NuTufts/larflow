import os,time,copy
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import ROOT as rt
from larflow import larflow
from ctypes import c_int


class larvoxelClassDataset(torch.utils.data.Dataset):

    _pdg2index = {11:0,
                  -11:0,
                  22:1,
                  13:2,
                  -13:2,
                  2212:3,
                  211:4,
                  -211:4,
                  321:5,
                  -321:5}
    
    def pdgcode_to_classindex(pdgcode):
        """
        """
        if pdgcode in larvoxelClassDataset._pdg2index:
            return larvoxelClassDataset._pdg2index[pdgcode]
        raise ValueError("[larvoxelClassDataset] unrecognized pdg code: %d"%(pdgcode))
    
    def __init__(self, filelist=None, filefolder=None, txtfile=None,
                 random_access=True, load_truth=False, load_meta_data=False,
                 triplet_limit=2500000, normalize_inputs=True,
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

        self.tree = rt.TChain("larvoxelpidtrainingdata")
        for ifile in range(file_v.size()):
            self.tree.Add( file_v.at(ifile) )
        self._nentries = self.tree.GetEntries()
        self._load_truth = load_truth

        # store parameters
        self.partition_index = 0
        self.num_partitions = 1
        self.start_index = 0
        self.end_index = self._nentries

        self._current_entry  = self.start_index
        self._nloaded        = 0
        self._verbose = False
        self._random_access = random_access
        self._random_entry_list = None
        self._max_num_tries = 10
        self._triplet_limit = triplet_limit
        self._normalize_inputs = normalize_inputs
        self._load_meta_data = load_meta_data
        if self._random_access:
            self._rng = np.random.default_rng(None)
            self._random_entry_list = self._rng.choice( self._nentries, size=self._nentries )        
                                 

    def __getitem__(self, idx):
        
        worker_info = torch.utils.data.get_worker_info()

        okentry = False
        num_tries = 0

        while not okentry:

            if self._random_access and self._current_entry>=self._nentries:
                # if random access and we've used all possible indices we reset the shuffled index list
                print("reset shuffled indices list")
                self._random_entry_list = self._rng.choice( self._nentries, size=self._nentries )
                self._current_entry = 0 # current entry in shuffled list

            ientry = int(self._current_entry)
        
            if self._random_access:
                # if random access, get next entry in shuffled index list
                data    = {"entry":ientry,
                           "tree_entry":self._random_entry_list[ientry]}
            else:
                # else just take the next entry
                data    = {"entry":ientry,
                           "tree_entry":int(ientry)%int(self._nentries)}

            okentry = self.get_data_from_tree( data )

            # increment the entry index
            self._current_entry += 1
                
            # increment the number of attempts we've made
            num_tries += 1
            if num_tries>=self._max_num_tries:
                raise RuntimeError("Tried the max num of times (%d) to get an acceptable piece of data"%(num_tries))

        self._nloaded += 1
        
        return copy.deepcopy(data)


    def __len__(self):
        return self._nentries

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
        data["coord"] = self.tree.coord_v.at(0).tonumpy()
        data["feat"]  = self.tree.feat_v.at(0).tonumpy()
        # renormalize feats
        if self._normalize_inputs:
            data["feat"] /= 50.0
            data["feat"] = np.clip( data["feat"], 0, 10.0 )
            
        if self._load_truth:
            data["pid"] = self.tree.pid_v.at(0).tonumpy()
            data["pid"][0] = larvoxelClassDataset.pdgcode_to_classindex( data["pid"][0] )

        if self._load_meta_data:
            data["truemom"] = self.tree.mom_v.at(0).tonumpy()
            data["truepos"] = self.tree.detpos_v.at(0).tonumpy()
        
        if self._verbose:
            tottime = time.time()-t_start            
            print("[larvoxelDataset::get_data_tree entry=%d loaded]"%(data["tree_entry"]))
            print("  io time: %.3f secs"%(dtio))
            print("  tot time: %.3f secs"%(tottime))
            
        return True


    def print_status(self):
        print("worker: entry=%d nloaded=%d"%(self._current_entry,self._nloaded))

    def set_partition(self,partition_index,num_partitions):
        self.partition_index = partition_index
        self.num_partitions = num_partitions
        self.start_index = int(self.partition_index*self._nentries)/int(self.num_partitions)
        self.end_index   = int((self.partition_index+1)*self._nentries)/int(self.num_partitions)
        self._current_entry = self.start_index

    def collate_fn(batch):
        return batch
    
            
if __name__ == "__main__":

    """
    Example of loading and reading the data.

    Branches in the ROOT tree we are retrieveing

root [2] larvoxelpidtrainingdata->Print()
******************************************************************************
*Tree    :larvoxelpidtrainingdata: LArMatch training data                                 *
*Entries :       17 : Total =         1440156 bytes  File  Size =     516083 *
*        :          : Tree compression factor =   2.78                       *
******************************************************************************
*Br    0 :run       : run/I                                                  *
*Entries :       17 : Total  Size=        634 bytes  File Size  =        113 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.39     *
*............................................................................*
*Br    1 :subrun    : subrun/I                                               *
*Entries :       17 : Total  Size=        649 bytes  File Size  =        116 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.38     *
*............................................................................*
*Br    2 :event     : event/I                                                *
*Entries :       17 : Total  Size=        644 bytes  File Size  =        129 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.23     *
*............................................................................*
*Br    3 :pid       : pid/I                                                  *
*Entries :       17 : Total  Size=        634 bytes  File Size  =        141 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.11     *
*............................................................................*
*Br    4 :shower0_or_track1 : shower0_or_track1/I                            *
*Entries :       17 : Total  Size=        704 bytes  File Size  =        137 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.25     *
*............................................................................*
*Br    5 :geant_trackid : geant_trackid/I                                    *
*Entries :       17 : Total  Size=        684 bytes  File Size  =        167 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.00     *
*............................................................................*
*Br    6 :ke        : ke/F                                                   *
*Entries :       17 : Total  Size=        629 bytes  File Size  =        156 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.00     *
*............................................................................*
*Br    7 :coord_v   : Int_t coord_v_                                         *
*Entries :       17 : Total  Size=       3099 bytes  File Size  =        175 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.35     *
*............................................................................*
*Br    8 :coord_v.ndims : Int_t ndims[coord_v_]                              *
*Entries :       17 : Total  Size=        964 bytes  File Size  =        173 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   2.10     *
*............................................................................*
*Br    9 :coord_v.shape : vector<int> shape[coord_v_]                        *
*Entries :       17 : Total  Size=       1442 bytes  File Size  =        325 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   2.59     *
*............................................................................*
*Br   10 :coord_v.data : vector<int> data[coord_v_]                          *
*Entries :       17 : Total  Size=     707329 bytes  File Size  =     157852 *
*Baskets :       11 : Basket Size=      32000 bytes  Compression=   4.48     *
*............................................................................*
*Br   11 :feat_v    : Int_t feat_v_                                          *
*Entries :       17 : Total  Size=       3096 bytes  File Size  =        174 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.36     *
*............................................................................*
*Br   12 :feat_v.ndims : Int_t ndims[feat_v_]                                *
*Entries :       17 : Total  Size=        957 bytes  File Size  =        172 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   2.10     *
*............................................................................*
*Br   13 :feat_v.shape : vector<int> shape[feat_v_]                          *
*Entries :       17 : Total  Size=       1435 bytes  File Size  =        328 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   2.56     *
*............................................................................*
*Br   14 :feat_v.data : vector<float> data[feat_v_]                          *
*Entries :       17 : Total  Size=     707312 bytes  File Size  =     349847 *
*Baskets :       11 : Basket Size=      32000 bytes  Compression=   2.02     *
*............................................................................*
*Br   15 :pid_v     : Int_t pid_v_                                           *
*Entries :       17 : Total  Size=       3021 bytes  File Size  =        173 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.36     *
*............................................................................*
*Br   16 :pid_v.ndims : Int_t ndims[pid_v_]                                  *
*Entries :       17 : Total  Size=        950 bytes  File Size  =        170 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   2.12     *
*............................................................................*
*Br   17 :pid_v.shape : vector<int> shape[pid_v_]                            *
*Entries :       17 : Total  Size=       1240 bytes  File Size  =        217 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   3.00     *
*............................................................................*
*Br   18 :pid_v.data : vector<int> data[pid_v_]                              *
*Entries :       17 : Total  Size=       1235 bytes  File Size  =        277 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   2.35     *
*............................................................................*
*Br   19 :detpos_v  : Int_t detpos_v_                                        *
*Entries :       17 : Total  Size=       3094 bytes  File Size  =        176 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.35     *
*............................................................................*
*Br   20 :detpos_v.ndims : Int_t ndims[detpos_v_]                            *
*Entries :       17 : Total  Size=        971 bytes  File Size  =        170 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   2.14     *
*............................................................................*
*Br   21 :detpos_v.shape : vector<int> shape[detpos_v_]                      *
*Entries :       17 : Total  Size=       1261 bytes  File Size  =        220 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   2.97     *
*............................................................................*
*Br   22 :detpos_v.data : vector<float> data[detpos_v_]                      *
*Entries :       17 : Total  Size=       1820 bytes  File Size  =        436 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   2.79     *
*............................................................................*
*Br   23 :mom_v     : Int_t mom_v_                                           *
*Entries :       17 : Total  Size=       3037 bytes  File Size  =        173 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.36     *
*............................................................................*
*Br   24 :mom_v.ndims : Int_t ndims[mom_v_]                                  *
*Entries :       17 : Total  Size=        950 bytes  File Size  =        170 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   2.12     *
*............................................................................*
*Br   25 :mom_v.shape : vector<int> shape[mom_v_]                            *
*Entries :       17 : Total  Size=       1240 bytes  File Size  =        218 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   2.99     *
*............................................................................*
*Br   26 :mom_v.data : vector<float> data[mom_v_]                            *
*Entries :       17 : Total  Size=       1799 bytes  File Size  =        579 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   2.10     *
*............................................................................*
    """
    
    import time
    from larcv import larcv
    larcv.load_pyutil()
    larcv.SetPyUtil()

    niter = 3
    batch_size = 4
    test = larvoxelClassDataset( filelist=["testdata/testdata.root"],
                                 random_access=False,
                                 load_truth=True)
    print("NENTRIES: ",len(test))
    
    loader = torch.utils.data.DataLoader(test,batch_size=batch_size,                                         
                                         collate_fn=larvoxelClassDataset.collate_fn)

    start = time.time()
    for iiter in range(niter):
        batch = next(iter(loader))
        print("====================================")
        for ib,data in enumerate(batch):
            print("ITER[%d]:BATCH[%d]"%(iiter,ib))
            print(" keys: ",data.keys())
            for name,d in data.items():
                if  name in ["entry","tree_entry"]:
                    print("  ",name,": ",d)
                elif type(d) is np.ndarray:
                    print("  ",name,": ",d.shape)
                else:
                    print("  ",name,": ",type(d))
            print(" pid: ",data["pid"])
            print(" coord: ",data["coord"][:10,:])
    end = time.time()
    elapsed = end-start
    sec_per_iter = elapsed/float(niter)
    print("sec per iter: ",sec_per_iter)
    
    loader.dataset.print_status()
