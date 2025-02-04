import os,sys

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Example of defining a data loader class
class LArMatchHitHDF5Dataset(Dataset):
    #The columns in the dataset
    COLUMNS = [
        "lmfeatures", #(C,N) C=48 usually
        "lmscores",   #(N,)
        "ssnet",      #(6,N)
        "paf",        #(3,N)
        "kpscores",   #(5,N)
        "pos",        #(3,N)
        "pixvals"]    #(3,N)

    TRUTH_COLUMNS = [
        "instanceids"
    ]

    
    def __init__(self, file_paths=None,
                 load_training_labels=True,
                 load_from_cachefile=None,
                 apply_max_filter=False,
                 max_num_spacepoints=10000):
        """
        Class constructor

        inputs
        ------
        file_paths: either a (1) [string] pointing to the path of textfile containing paths to input h5 files along with events per entry
                         or (2) a [list of strings] pointing to paths to input h5 files to include in the dataset.
        load_from_cachefile [string] path to cache textfile
        apply_max_filter [bool] enforce a max number of points. will randomly sample.
        max_num_spacepoints [int] if apply_max_filter=True, will only return around this many points (good for visualization)
        """
        if file_paths is None and load_from_cachefile is None:
            print("to specify input files, you must provide a value to one of the keyword arguments: ")
            print("  file_paths: a list of file paths to include in the dataset")
            print("  load_from_cachefile: a text file with a list of files with number of entries information. make this with make_cache_file")
        self.file_paths = file_paths
        self.load_from_cachefile = load_from_cachefile
        self.max_num_spacepoints=max_num_spacepoints
        self.apply_max_filter=apply_max_filter
        self.load_training_labels = load_training_labels

        self.COLS = LArMatchHitHDF5Dataset.COLUMNS
        if self.load_training_labels:
            self.COLS += LArMatchHitHDF5Dataset.TRUTH_COLUMNS

        self.nlength = 0
        if self.load_from_cachefile is not None:
            with open(self.load_from_cachefile,'r') as fcache:
                ll = fcache.readlines()
                self.nlength = int(ll[-1].strip().split()[-1])
            print("using cache to set number of entries in dataset to ",self.nlength)
        else:
            print("Make entry table for list of files (len=",len(self.file_paths),")")
            self.make_entry_table()


    def make_entry_table(self):
        # we have to scan the files to map out which file has which indices        
        self.dataset_lengths = []
        self.cumulative_lengths = [0]
        
        if self.file_paths is not None:
            print("MAKE_ENTRY_TABLE: Loading from list of file paths")
            for file_path in self.file_paths:
                with h5py.File(file_path, 'r') as hf:
                    # because each entry has its own column, we can infer the number of entries
                    nkeys = len(hf.keys())
                    length = nkeys // len(self.COLS)  # Divide by number of columns in each entry
                    print("nkeys=",nkeys," length=",length," for ",file_path)
                    self.dataset_lengths.append(length)
                    self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)
            self.nlength = self.cumulative_lengths[-1]
        elif self.load_from_cachefile is not None:
            print("MAKE_ENTRY_TABLE: Loading from cached list of files")
            self.file_paths = []
            with open(self.load_from_cachefile,'r') as fcache:
                ll = fcache.readlines()
                for l in ll:
                    linfo = l.strip().split()
                    fname   = linfo[0]
                    flength = int(linfo[1])
                    cdf     = int(linfo[2])
                    self.file_paths.append(fname)
                    self.dataset_lengths.append(flength)
                    self.cumulative_lengths.append(cdf)
            self.nlength = self.cumulative_lengths[-1]
        
        
    def __len__(self):
        #return self.cumulative_lengths[-1]        
        return self.nlength


    def __getitem__(self, idx):

        if not hasattr(self,'dataset_lengths'):
            print("call make_entry_table from __getitem__")
            self.make_entry_table()
        
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        local_idx = idx - self.cumulative_lengths[file_idx]
        
        entry_data = {}
        with h5py.File(self.file_paths[file_idx], 'r') as hf:
            for col in self.COLS:
                #print("retrieve key=",f'{col}_{local_idx}')
                arr = np.array(hf[f'{col}_{local_idx}'])
                if len(arr.shape)==1:
                    arr = np.expand_dims(arr, 0)
                entry_data[col] = arr

        # here we have a chance to modify the data
        # do we subsample to limit the number of spacepoints?
        # do we crop around the neutrino vertex or crop within some box
        # do we mask out the ghost and cosmic spacepoints?
        npts = entry_data['pos'].shape[-1]

        # add index of entry
        entry_data["idx"] = idx
        entry_data["npts"] = npts

        if self.apply_max_filter and npts>self.max_num_spacepoints:
            frac = float(self.max_num_spacepoints)/float(npts)
            ptfilter = np.random.random( npts ) < frac
            for col in self.COLS:
                entry_data[col] = entry_data[col][:,ptfilter[:]]

        
        return entry_data

    def make_cache_file(self,cache_file_name):
        print("Dumping fileset length into into a cache file: ",cache_file_name)
        with open(cache_file_name,'w') as f:
            for i,fname in enumerate(self.file_paths):
                print(fname,' ',self.dataset_lengths[i],' ',self.cumulative_lengths[i+1],file=f)

    def collate_fn(batch):
        """
        Simple pass-through
        """
        return batch

    def make_batch_sparse_tensors( batchdata, device=None, verbose=False, triplet_key="matchtriplet_v" ):
        # convert wire plane data, in numpy form into ME.SparseTensor form
        # data comes back as numpy arrays.
        # we need to move it to DEVICE and then form MinkowskiEngine SparseTensors
        # needs to be done three times: one for each wire plane of the detector
        # params
        # batchdata list
        try:
            import MinkowskiEngine as ME
        except:
            print("could not load MinkowskiEngine library")
            sys.exit(0)

        wireplane_sparsetensors = []
    
        nb = len(batchdata)
        print("Batchsize: ",nb)

        npts = 0
        for data in batchdata:
            npts += data[triplet_key].shape[0]
        print("Drawn total spacepoints: ",npts)

        for p in range(3):
            if verbose:
                print("plane ",p)
                for b,data in enumerate(batchdata):
                    print(" coord plane[%d] batch[%d]"%(p,b),": ",data["coord_%d"%(p)].shape)

            coord_v = [ torch.from_numpy(data["coord_%d"%(p)]).to(device) for data in batchdata ]
            feat_v  = [ torch.from_numpy(data["feat_%d"%(p)]).to(device) for data in batchdata ]

            # hack make random matrix
            # coord_v = []
            # feat_v = []
            # for b in range(config["BATCH_SIZE"]):
            #     fake_coord = np.random.randint( 0, high=1004, size=(200000,2) )
            #     coord_v.append( torch.from_numpy(fake_coord).to(DEVICE) )
            #     fake_feat  = np.random.rand( 200000, 1 )
            #     feat_v.append( torch.from_numpy(fake_feat.astype(np.float32)).to(DEVICE) )

            for x in coord_v:
                x.requires_grad = False
        
            coords, feats = ME.utils.sparse_collate(coord_v, feat_v)
            if verbose:
                print(" coords: ",coords.shape)
                print(" feats: ",feats.shape)
            wireplane_sparsetensors.append( ME.SparseTensor(features=feats, coordinates=coords) )

            # we also need the metadata associating possible 3d spacepoints

        # collect the wire image locations they project to for the batch
        matchtriplet_v = []
        for b,data in enumerate(batchdata):
            matchtriplet_v.append( torch.from_numpy(data["matchtriplet_v"]).to(device) )
            if verbose:
                print("batch ",b," matchtriplets: ",matchtriplet_v[b].shape)

        # collect the image coordinates for each spacepoint for the batch
        query_v = []
        for p in range(3):
            plane_query = [ data['query_coord_%d'%(p)] for data in batchdata ]
            query_v.append( torch.from_numpy( np.concatenate( plane_query, axis=0 ) ).to(device) )
            print("plane [",p,"] query coords shape: ",query_v[p].shape)

        return wireplane_sparsetensors, matchtriplet_v, query_v


# Usage example
def get_data_loader(file_paths, batch_size=2, num_workers=1, shuffle=True,
                    load_from_cachefile=False,
                    collate_for_training=False,
                    max_num_spacepoints=100000,
                    apply_max_filter=False):
    if not load_from_cachefile:
        xpaths = []
        if type(file_paths) is str:
            if os.path.exists(file_paths) and os.path.isfile(file_paths):
                # treat as text file
                print("Loading data files from text file: ",file_paths)
                with open(file_paths) as f:
                    flines = f.readlines()
                    for l in flines:
                        l = l.strip()
                        if os.path.exists(l):
                            xpaths.append(l.strip())
                        else:
                            raise RuntimeError("bad input file path: ",l)
            elif os.path.exists(file_paths) and os.path.isdir(file_paths):
                print("Loading data files from directory: ",file_paths)
                raise RuntimeError("Not yet implemented")
        elif type(file_paths) is list:
            print("Loading data files from list of file paths")
            xpaths = file_paths
        dataset = LArMatchHDF5Dataset(file_paths=xpaths, collate_for_training=collate_for_training,
                                    max_num_spacepoints=max_num_spacepoints,
                                    apply_max_filter=apply_max_filter)
    else:
        dataset = LArMatchHDF5Dataset(load_from_cachefile=file_paths, collate_for_training=collate_for_training, 
                                    max_num_spacepoints=max_num_spacepoints,
                                    apply_max_filter=apply_max_filter)
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=LArMatchHitHDF5Dataset.collate_fn)
    



    
