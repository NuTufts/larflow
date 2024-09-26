import os,sys,time
import larmatch
from larmatch.data.larmatch_hdf5_reader import LArMatchHDF5Dataset, get_data_loader

#dataset = LArMatchHDF5Dataset(load_from_cachefile="cache_list_larmatch_training_dataset.txt")

#N=4
niters = 20
batch_size=4
Nmax = int(1.5*batch_size)

results = {}
for N in range(0,Nmax+1):

    dataloader = get_data_loader( 'cache_list_larmatch_validation_dataset.txt', batch_size=batch_size,
                                  num_workers=N, shuffle=True,
                                  load_from_cachefile=True,
                                  collate_for_training=True)
    data_iter = iter(dataloader)
    
    tstart = time.time()
    for iiter in range(niters):
        print("[N=",N,"] iter ",iiter)
        batch = next(data_iter)

    tend = time.time()
    tload = float(tend-tstart)
    print("elapsed: ",tload)
    print("time per batch: ",tload/float(niters))
    results[N] = tload/float(niters)

for N in range(0,Nmax):
    print("workers N=",N,": ",results[N]," sec/batch")
