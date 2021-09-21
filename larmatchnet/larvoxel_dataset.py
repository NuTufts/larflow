import os,time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import ROOT as rt
from larflow import larflow

class larvoxelDataset(torch.utils.data.Dataset):
    def __init__(self, filelist=None, filefolder=None, txtfile=None,
                 random_access=True, verbose=False,
                 voxelize=True, voxelsize_cm=1.0, is_voxeldata=False):
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

        self.is_voxeldata = is_voxeldata
        if not is_voxeldata:
            self.loader = larflow.keypoints.LoaderKeypointData( file_v )
            self.nentries = self.loader.GetEntries()
        else:
            print("Using files which already contain voxel data arrays")
            self.voxeldata_tree = rt.TChain("larvoxeltrainingdata")
            for ifile in range(file_v.size()):
                self.voxeldata_tree.Add( file_v.at(ifile) )
            self.nentries = self.voxeldata_tree.GetEntries()
            
        self.random_access = random_access
        self.partition_index = 0
        self.num_partitions = 1
        self.start_index = 0
        self.end_index = self.nentries

        self._current_entry  = self.start_index
        self._nloaded        = 0
        self._verbose = False

        self._voxelize = voxelize
        self._voxelsize_cm = voxelsize_cm
        if self._voxelize and not self.is_voxeldata:
            self.voxelizer = larflow.voxelizer.VoxelizeTriplets()
            self.voxelizer.set_voxel_size_cm( self._voxelsize_cm )
                                 

    def __getitem__(self, idx):

        ientry = self._current_entry
        data    = {"entry":ientry,
                   "tree_entry":int(ientry)%int(self.nentries)}

        if not self.is_voxeldata:
            data = self.get_data_dict_from_triplet_file( data )
        else:
            data = self.get_data_dict_from_voxelarray_file( data )
            
        return data

    def get_data_dict_from_triplet_file(self, data):    

        # get data from match trees            
        nbytes = self.loader.load_entry(data["tree_entry"])
        if self._verbose:
            print("nbytes: ",nbytes," for tree[",name,"] entry=",data['tree_entry'])

        if self._verbose:
            dtio = time.time()-tio

        if not self._voxelize:
            # get the spacepoints            
            matchdata   = self.loader.triplet_v[0].make_spacepoint_charge_array()
        else:
            # voxelize the data
            self.voxelizer.make_voxeldata( self.loader.triplet_v[0] )
            voxdata = self.voxelizer.get_full_voxel_labelset_dict( self.loader )
            origin_x = self.voxelizer.get_origin()[0]
            origin_y = self.voxelizer.get_origin()[1]
            origin_z = self.voxelizer.get_origin()[2]
            pos = np.zeros( (voxdata["voxcoord"].shape[0], 6 ), dtype=np.float32 )
            pos[:,0:3] = voxdata["voxcoord"]
            pos[:,0] += origin_x/self._voxelsize_cm
            pos[:,1] += origin_x/self._voxelsize_cm 
            pos[:,2] += origin_x/self._voxelsize_cm
            pos *= self._voxelsize_cm
            
            pos[:,3:] = np.clip( voxdata["voxfeat"]/40.0, 0, 10.0 )

            matchdata = {"spacepoint_t":pos,
                         "truetriplet_t":voxdata["voxlabel"]}
            data.update(voxdata)
        
        # add the contents to the data dictionary
        data.update(matchdata)

        self._nloaded += 1
        self._current_entry += 1
        if self._current_entry>=self.end_index:
            self._current_entry = self.start_index
            
        if self._verbose:
            tottime = time.time()-t_start            
            print("[larennetDataset entry=%d loaded]"%(data["tree_entry"]))
            print("  io time: %.3f secs"%(dtio))
            print("  tot time: %.3f secs"%(tottime))
            
        return data

    def get_data_dict_from_voxelarray_file(self, data):
        
        # get data from match trees            
        nbytes = self.voxeldata_tree.GetEntry(data["tree_entry"])
        if self._verbose:
            print("nbytes: ",nbytes," for tree[",name,"] entry=",data['tree_entry'])

        if self._verbose:
            dtio = time.time()-tio

            
        data = {}
        data["voxcoord"] = self.voxeldata_tree.coord_v.at(0).tonumpy()
        data["voxfeat"]  = self.voxeldata_tree.feat_v.at(0).tonumpy()
        data["ssnet_labels"] =  self.voxeldata_tree.ssnet_truth_v.at(0).tonumpy()
        data["kplabel"] =  self.voxeldata_tree.kp_truth_v.at(0).tonumpy()
        data["voxlabel"] = self.voxeldata_tree.larmatch_truth_v.at(0).tonumpy()
        data["voxlmweight"] = self.voxeldata_tree.larmatch_weight_v.at(0).tonumpy()
        data["kpweight"]    = self.voxeldata_tree.kp_weight_v.at(0).tonumpy()
        data["ssnet_weight"] = self.voxeldata_tree.ssnet_weight_v.at(0).tonumpy()

        self._nloaded += 1
        self._current_entry += 1
        if self._current_entry>=self.end_index:
            self._current_entry = self.start_index
            
        if self._verbose:
            tottime = time.time()-t_start            
            print("[larvoxelDataset::get_data_dict_from_voxelarray_file entry=%d loaded]"%(data["tree_entry"]))
            print("  io time: %.3f secs"%(dtio))
            print("  tot time: %.3f secs"%(tottime))
            
        return data
    
    def __len__(self):
        return self.nentries

    def print_status(self):
        print("worker: entry=%d nloaded=%d"%(self._current_entry,self._nloaded))

    def set_partition(self,partition_index,num_partitions):
        self.partition_index = partition_index
        self.num_partitions = num_partitions
        self.start_index = int(self.partition_index*self.nentries)/int(self.num_partitions)
        self.end_index   = int((self.partition_index+1)*self.nentries)/int(self.num_partitions)
        self._current_entry = self.start_index

    def collate_fn(batch):
        print("[larvoxelDataset::collate_fn] batch: ",type(batch)," len=",len(batch))
        return batch
    
            
if __name__ == "__main__":

    import time

    niter = 10
    batch_size = 1
    test = larvoxelDataset( filelist=["larmatchtriplet_ana_trainingdata_testfile.root"])
    print("NENTRIES: ",len(test))
    
    loader = torch.utils.data.DataLoader(test,batch_size=batch_size,collate_fn=larvoxelDataset.collate_fn)

    start = time.time()
    for iiter in range(niter):
        batch = next(iter(loader))
        print("====================================")
        for ib,data in enumerate(batch):
            print("ITER[%d]:BATCH[%d]"%(iiter,ib))
            print(" keys: ",data.keys())
            for name,d in data.items():
                if type(d) is np.ndarray:
                    print("  ",name,": ",d.shape)
                else:
                    print("  ",name,": ",type(d))
    end = time.time()
    elapsed = end-start
    sec_per_iter = elapsed/float(niter)
    print("sec per iter: ",sec_per_iter)
