import os,time
import copy
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import ROOT as rt
from larflow import larflow
#import MinkowskiEngine as ME

class LMDataset(torch.utils.data.Dataset):
    def __init__(self, filelist=None, filefolder=None, txtfile=None,
                 random_access=True, verbose=False,
                 voxelize=True, voxelsize_cm=1.0,
                 max_num_voxels=1e6,
                 is_voxeldata=False):
        """
        Parameters:
        is_voxeldata [bool] if true, assumes ROOT file containts trees with larcv::NumpyArray[Int/Float] classes.
                              These classes are easy to transform into numpy arrays.
                            if false (default), assumes ROOT file contains 3D spacepoint data which we
                              need to convert into numpy array format.
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
        self._random_access = random_access
        self._random_entry_list = None
        if self._random_access:
            self._rng = np.random.default_rng(None)
            self._random_entry_list = self._rng.choice( self.nentries, size=self.nentries )

        self._voxelize = voxelize
        self._voxelsize_cm = voxelsize_cm
        self._max_num_voxels = max_num_voxels
        self._max_num_tries = 10
        if self._voxelize and not self.is_voxeldata:
            self.voxelizer = larflow.voxelizer.VoxelizeTriplets()
            self.voxelizer.set_voxel_size_cm( self._voxelsize_cm )


    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()


        okentry = False
        num_tries = 0

        while not okentry:

            if self._random_access and self._current_entry>=self.nentries:
                # if random access and we've used all possible indices we reset the shuffled index list
                print("reset choices")
                self._random_entry_list = self._rng.choice( self.nentries, size=self.nentries )
                self._current_entry = 0

            ientry = int(self._current_entry)

            if self._random_access:
                # if random access, get next entry in shuffled index list
                data    = {"entry":ientry,
                           "tree_entry":self._random_entry_list[ientry]}
            else:
                # else just take the next entry
                data    = {"entry":ientry,
                           "tree_entry":int(ientry)%int(self.nentries)}

            if not self.is_voxeldata:
                data = self.get_data_dict_from_triplet_file( data )
            else:
                data = self.get_data_dict_from_voxelarray_file( data )

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

        xlist = np.unique( data["voxcoord"], axis=0, return_counts=True )
        indexlist = xlist[0]
        counts = xlist[-1]
        hasdupe = False
        for i in range(indexlist.shape[0]):
            if counts[i]>1:
                print(i," ",indexlist[i,:]," counts=",counts[i])
                hasdupe = True
        if hasdupe:
            raise("[larvoxel_dataset::__getitem__] Dupe introduced somehow in batch-index=%d"%(ibatch)," arr=",data["voxcoord"].shape)

        self._nloaded += 1

        return copy.deepcopy(data)

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

            # check for failures in the data
            if voxdata["voxcoord"].shape[0]>=self._max_num_voxels:
                # don't return giant events
                return None

            origin_x = self.voxelizer.get_origin()[0]
            origin_y = self.voxelizer.get_origin()[1]
            origin_z = self.voxelizer.get_origin()[2]
            pos = np.zeros( (voxdata["voxcoord"].shape[0], 6 ), dtype=np.float32 )
            pos[:,0:3] = voxdata["voxcoord"]
            pos[:,0] += origin_x/self._voxelsize_cm
            pos[:,1] += origin_x/self._voxelsize_cm
            pos[:,2] += origin_x/self._voxelsize_cm
            pos *= self._voxelsize_cm

            pos[:,3:] = np.clip( voxdata["voxfeat"]/150.0, 0, 10.0 )

            matchdata = {"spacepoint_t":pos,
                         "truetriplet_t":voxdata["voxlabel"]}
            data.update(voxdata)

        # add the contents to the data dictionary
        data.update(matchdata)


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

        #print("get_data_dict_from_voxelarray_file: ",self.voxeldata_tree.coord_v.at(0).tonumpy().shape)
        data["voxcoord"] = self.voxeldata_tree.coord_v.at(0).tonumpy()

        # check for failures in the data
        if data["voxcoord"].shape[0]>=self._max_num_voxels:
            # don't return giant events
            return None

        data["voxfeat"]  = self.voxeldata_tree.feat_v.at(0).tonumpy()
        #print("prenorm: ",data["voxfeat"].shape,", ",np.min(data["voxfeat"])," to ",np.max(data["voxfeat"]))
        # normalize features
        data["voxfeat"] = np.clip( data["voxfeat"]/150.0, 0, 10.0 )
        #print("postnorm: ",np.min(data["voxfeat"])," to ",np.max(data["voxfeat"]))

        data["ssnet_labels"] =  self.voxeldata_tree.ssnet_truth_v.at(0).tonumpy().astype(np.int)
        #data["kplabel"] =  self.voxeldata_tree.kp_truth_v.at(0).tonumpy()
        data["voxlabel"] = self.voxeldata_tree.larmatch_truth_v.at(0).tonumpy()
        data["voxlmweight"] = self.voxeldata_tree.larmatch_weight_v.at(0).tonumpy()
        #data["kpweight"]    = self.voxeldata_tree.kp_weight_v.at(0).tonumpy()
        data["ssnet_weights"] = self.voxeldata_tree.ssnet_weight_v.at(0).tonumpy()

        if self._verbose:
            tottime = time.time()-t_start
            print("[LMDataset::get_data_dict_from_voxelarray_file entry=%d loaded]"%(data["tree_entry"]))
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
        """
        """
        npoints = 0
        batch_size = len(batch)
        tree_entries = [ x["tree_entry"] for x in batch ]
        npoints_v = []
        print("[LMDataset::collate_fn] batch len=%d tree entries="%(batch_size),tree_entries)

        for ib,data in enumerate(batch):
            print("batch[%d]: "%(ib),data["voxcoord"].shape)
            npoints += data["voxcoord"].shape[0]
            npoints_v.append( data["voxcoord"].shape[0] )
        print("[LMDataset::collate_fn] batch: ",type(batch)," len=",len(batch)," nvoxels=",npoints)

        # check for duplicates in individual entries in batch
        for ib,data in enumerate(batch):
            xlist = np.unique( data["voxcoord"], axis=0, return_counts=True )
            indexlist = xlist[0]
            counts = xlist[-1]
            hasdupe = False
            for i in range(indexlist.shape[0]):
                if counts[i]>1:
                    print(i," ",indexlist[i,:]," counts=",counts[i])
                    hasdupe = True
            if hasdupe:
                raise("Dupe introduced somehow in batch-index=%d"%(ib)," arr=",data["voxcoord"].shape)


        v = ["tree_entry","voxcoord","voxfeat","voxlabel","ssnet_labels","kplabel","voxlmweight","ssnet_weights","kpweight"]

        print("batch_size: ",batch_size)
        for ib in range(batch_size):
            print("batch ",ib," --------------------")
            for vv in v[1:]:
                print("  ",vv,": ",batch[ib][vv].shape)

        batch_dict = {"tree_entry":tree_entries,
                      "voxcoord":[],
                      "voxfeat":[]}
        for k in batch[0].keys():
            print(k,": ",type(batch[0][k]))
            if type(batch[0][k]) is not np.ndarray:
                continue
            arr = batch[0][k]
            if k in ["spacepoint_t","truetriplet_t"]:
                continue
            elif k=="voxcoord":
                batch_dict[k] = np.zeros( (npoints,arr.shape[1]+1), dtype=arr.dtype )
                #batch_dict["voxcoord"].append( arr )
            elif k=="voxfeat":
                batch_dict[k] = np.zeros( (npoints,arr.shape[1]), dtype=arr.dtype )
                #batch_dict["voxfeat"].append( arr )
            elif len(arr.shape)>1:
                batch_dict[k] = np.zeros( (1,arr.shape[0],npoints), dtype=arr.dtype )
            else:
                batch_dict[k] = np.zeros( (1,npoints), dtype=arr.dtype )
            print(k," array=",arr.shape," batch array: ",batch_dict[k].shape)
        #coords, feats = ME.utils.sparse_collate(batch_dict["voxcoord"], batch_dict["voxfeat"])

        npoints = 0
        for ib,data in enumerate(batch):

            n = data["voxcoord"].shape[0]
            batch_dict["voxcoord"][npoints:npoints+n,0]  = ib
            batch_dict["voxcoord"][npoints:npoints+n,1:] = data["voxcoord"]
            batch_dict["voxfeat"][npoints:npoints+n,:] = data["voxfeat"]

            for k in v[3:]:
                arr = data[k]
                if len(arr.shape)>1:
                    batch_dict[k][0,:,npoints:npoints+n] = arr
                else:
                    batch_dict[k][0,npoints:npoints+n] = arr

            npoints += n

        #coords = batch_dict["voxcoord"]
        #feats  = batch_dict["voxfeat"]
        #A = ME.SparseTensor(features=torch.from_numpy(feats), coordinates=torch.from_numpy(coords))
        #print("sparsetensor: ",A)
        #batch_dict["sparsetensor"] = A

        return batch_dict


if __name__ == "__main__":

    import time
    import MinkowskiEngine as ME

    niter = 100
    batch_size = 1
    #testfile="larvoxeldata_bnb_nu_traindata_1cm_0000.root"
    #testfile="larvoxeldata_bnb_nu_traindata_1cm_0047.root"
    testfile="testdata/larvoxeldata_bnb_nu_traindata_3mm_bnbnu_0000.root"

    import ROOT as rt
    out = rt.TFile("temp_larvoxel_dataset_hist.root",'recreate')
    h = rt.TH1F("hnvoxels","num voxels; number of voxels in event;",20,0,4e6)


    DEVICE=torch.device("cpu")
    test = LMDataset( filelist=[testfile],
                            is_voxeldata=True,
                            random_access=False )
    print("NENTRIES: ",len(test))

    loader = torch.utils.data.DataLoader(test,batch_size=batch_size,collate_fn=LMDataset.collate_fn)

    max_nvoxels = 0

    start = time.time()
    for iiter in range(niter):

        print("====================================")
        #for ib,data in enumerate(batch):
        print("ITER[%d]"%(iiter))

        batch = next(iter(loader))

        print("batch keys: ",batch.keys())
        print(batch["tree_entry"])

        for name,d in batch.items():
            if type(d) is np.ndarray:
                print("  ",name,": ",d.shape)
            else:
                print("  ",name,": ",type(d))

        print("iteration batch check ------------")
        xlist = np.unique( batch["voxcoord"], axis=0, return_counts=True )
        indexlist = xlist[0]
        counts = xlist[-1]
        hasdupe = False
        for i in range(indexlist.shape[0]):
            if counts[i]>1:
                print(i," ",indexlist[i,:]," ",counts[i])
                hasdupe = True
        if hasdupe:
            raise("iteration has dupe! -------------")

        coords = batch["voxcoord"]
        feats  = batch["voxfeat"]
        A = ME.SparseTensor(features=torch.from_numpy(feats).to(DEVICE),
                            coordinates=torch.from_numpy(coords).to(DEVICE))

        print("sparsetensor: ",A.shape)

        if coords.shape[0]>max_nvoxels:
            max_nvoxels = coords.shape[0]
            maxiter = iiter
        h.Fill( coords.shape[0] )

        #xinput = batch["sparsetensor"]
        #print("after sparse collate")
        #print(xinput)
        print("====================================")

    end = time.time()
    elapsed = end-start
    sec_per_iter = elapsed/float(niter)
    print("sec per iter: ",sec_per_iter)
    print("max nvoxels: ",max_nvoxels)
    print("max iter: ",maxiter)
    h.Write()
    out.Close()
