import os,sys

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Example of defining a data loader class
class LArMatchHDF5Dataset(Dataset):
    #The columns in the dataset
    COLUMNS = [
        "matchtriplet",
        "match_weight",
        "spacepoints",
        "positive_indices",
        "ssnet_label",
        "ssnet_top_weight",
        "ssnet_class_weight",
        "kplabel",
        "kplabel_weight",
        "kpshift",
        "paf_label",
        "paf_weight",
        "origin_label",
        "keypoint_truth_kptype_pdg_trackid",
        "keypoint_truth_pos",
        "wireimage_plane0",
        "wireimage_plane1",
        "wireimage_plane2"]

    COLLATE_FOR_TRAINING = False
    
    def __init__(self, file_paths=None, collate_for_training=False, load_from_cachefile=None):
        if file_paths is None and load_from_cachefile is None:
            print("to specify input files, you must provide a value to one of the keyword arguments: ")
            print("  file_paths: a list of file paths to include in the dataset")
            print("  load_from_cachefile: a text file with a list of files with number of entries information. make this with make_cache_file")
        self.file_paths = file_paths
        self.dataset_lengths = []
        self.cumulative_lengths = [0]
        LArMatchHDF5Dataset.collate_for_training = collate_for_training
        
        # we have to scan the files to map out which file has which indices
        if file_paths is not None:
            print("LOADING FROM LIST OF FILE PATHS")
            for file_path in file_paths:
                with h5py.File(file_path, 'r') as hf:
                    # because each entry has its own column, we can infer the number of entries
                    nkeys = len(hf.keys())
                    length = nkeys // len(LArMatchHDF5Dataset.COLUMNS)  # Divide by number of columns in each entry
                    print("length=",length," for ",file_path)
                    self.dataset_lengths.append(length)
                    self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)
        elif load_from_cachefile is not None:
            print("LOADING FROM CACHED LIST OF FILES")            
            self.file_paths = []
            with open(load_from_cachefile,'r') as fcache:
                ll = fcache.readlines()
                for l in ll:
                    linfo = l.strip().split()
                    fname   = linfo[0]
                    flength = int(linfo[1])
                    cdf     = int(linfo[2])
                    self.file_paths.append(fname)
                    self.dataset_lengths.append(flength)
                    self.cumulative_lengths.append(cdf)
        
    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        local_idx = idx - self.cumulative_lengths[file_idx]
        
        entry_data = {}
        with h5py.File(self.file_paths[file_idx], 'r') as hf:
            for col in LArMatchHDF5Dataset.COLUMNS:
                #print("retrieve key=",f'{col}_{local_idx}')
                entry_data[col] = np.array(hf[f'{col}_{local_idx}'])

        # here we have a chance to modify the data
        # do we subsample to limit the number of spacepoints?
        # do we crop around the neutrino vertex or crop within some box
        # do we mask out the ghost and cosmic spacepoints?
        
        return entry_data

    def make_cache_file(self,cache_file_name):
        print("Dumping fileset length into into a cache file: ",cache_file_name)
        with open(cache_file_name,'w') as f:
            for i,fname in enumerate(self.file_paths):
                print(fname,' ',self.dataset_lengths[i],' ',self.cumulative_lengths[i+1],file=f)

    def collate_fn(batch):
        #print("[larmatchDataset::collate_fn] batch: ",type(batch)," len=",len(batch))
        #print(batch)
        if LArMatchHDF5Dataset.collate_for_training:
            rebatch = []
            for batchdata in batch:
                rebatchdata = {}
                rebatchdata['matchtriplet_v']   = batchdata['matchtriplet']
                rebatchdata['larmatch_truth']   = batchdata['matchtriplet'][:,3]
                rebatchdata['larmatch_weight']  = batchdata['match_weight']
                rebatchdata['ssnet_truth']      = batchdata['ssnet_label']
                rebatchdata['ssnet_weight']     = batchdata['ssnet_class_weight']*batchdata['ssnet_top_weight']
                rebatchdata['keypoint_truth']   = np.transpose( batchdata['kplabel'], (1,0) )
                rebatchdata['keypoint_weight']  = np.transpose( batchdata['kplabel_weight'], (1,0) )
                rebatchdata['positive_indices'] = batchdata['positive_indices']
                rebatchdata['paf_label']        = np.expand_dims( np.transpose( batchdata['paf_label'],  (1,0) ), 0 )
                rebatchdata['paf_weight']       = batchdata['paf_weight']
                for p in range(3):
                    rebatchdata['coord_%d'%(p)] = batchdata['wireimage_plane%d'%(p)][:,:2].astype(np.int64)
                    feat_t = np.expand_dims( batchdata['wireimage_plane%d'%(p)][:,2].astype(np.float32), 1 )
                    # normalize feature data
                    feat_t -= 50.0 # center around mip values
                    feat_t /= 200.0 # scale
                    feat_t = np.clip( feat_t, -5.0, 5.0 )
                    rebatchdata['feat_%d'%(p)] = feat_t
                rebatch.append( rebatchdata )
            return rebatch
        else:
            return batch

# Usage example
def get_data_loader(file_paths, batch_size=2, num_workers=1, shuffle=True,
                    load_from_cachefile=False,
                    collate_for_training=False):
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
        dataset = LArMatchHDF5Dataset(file_paths=xpaths, collate_for_training=collate_for_training)
    else:
        dataset = LArMatchHDF5Dataset(load_from_cachefile=file_paths, collate_for_training=collate_for_training)
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=LArMatchHDF5Dataset.collate_fn)
    



    
