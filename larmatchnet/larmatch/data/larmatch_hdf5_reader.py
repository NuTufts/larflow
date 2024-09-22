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
        "keypoint_truth_pos"]

    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.dataset_lengths = []
        self.cumulative_lengths = [0]
        # we have to scan the files to map out which file has which indices
        for file_path in file_paths:
            with h5py.File(file_path, 'r') as hf:
                # because each entry has its own column, we can infer the number of entries
                nkeys = len(hf.keys())
                length = nkeys // len(LArMatchHDF5Dataset.COLUMNS)  # Divide by number of columns in each entry
                print("length=",length," for ",file_path)
                self.dataset_lengths.append(length)
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)
        
    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        local_idx = idx - self.cumulative_lengths[file_idx]
        
        entry_data = {}
        with h5py.File(self.file_paths[file_idx], 'r') as hf:
            for col in LArMatchHDF5Dataset.COLUMNS:
                print("retrieve key=",f'{col}_{local_idx}')
                entry_data[col] = np.array(hf[f'{col}_{local_idx}'])
        
        return entry_data

    def collate_fn(batch):
        #print("[larmatchDataset::collate_fn] batch: ",type(batch)," len=",len(batch))
        #print(batch)
        return batch

# Usage example
def get_data_loader(file_paths, batch_size=2, num_workers=1, shuffle=True):
    dataset = LArMatchHDF5Dataset(file_paths)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=LArMatchHDF5Dataset.collate_fn)

