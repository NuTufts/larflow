import os,sys
import ROOT as rt
from ROOT import std
import numpy

from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow
import os,sys,argparse,time
import random
import torch
import math
import torch.nn as nn
import torch.optim as optim
import pickle

import matplotlib.pyplot as plt
from SpatialEmbed import SpatialEmbed, spatialembed_loss, post_process
from particle_list import *
from SpatialEmbedVisualization import *

parser = argparse.ArgumentParser("Read TTree")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-d', "--input-directory", type=str, help='TTree input directory')
group.add_argument("-f", "--input-tree-file", type=str,help="Input TTreee file [required]")
parser.add_argument("-s", "--save", type=str, help="Model save path")
parser.add_argument("-p", "--visualize",required=False,type=int,help="Percentage [0-100] of events to visualize")
parser.add_argument("-u", "--vis-uneven", action='store_true', help="Visualize all cases where num instances is uneven amongst planes")
parser.add_argument("-v", "--verbose", action='store_true', help="Verbose")
parser.add_argument("-l", "--track_loss", type=int, help="Loss tracking step")

args = parser.parse_args()

if (args.visualize) and (args.visualize < 0 or args.visualize > 100):
    raise ValueError('Visualize percentage must be between 0-100.')

if (args.visualize != None) or args.vis_uneven:
    visualization_setup(particle_names)

if args.input_tree_file:
    directory_files = [args.input_tree_file]
    train_files = directory_files
    test_files = []
    validation_files = []
else:
    directory_files = [os.path.join(args.input_directory, file) for file in os.listdir(args.input_directory)]
    random.shuffle(directory_files)
    train_split = int(len(directory_files) * 0.8)
    test_split = train_split + int(math.ceil(len(directory_files) * 0.1))
    train_files = directory_files[:train_split]
    test_files = directory_files[train_split:test_split]
    validation_files = directory_files[test_split:]

loss_tracking = []

class_loss = []
seed_loss = []
sigma_smooth_loss = []

IMG_WIDTH  = 3456
IMG_HEIGHT = 1008
IMG_BUFF = 15
NUM_TYPES = len(particle_list)

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SpatialEmbed(features_per_layer=NUM_TYPES).to(device)
model = model.train()

optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = spatialembed_loss

events = 0

loops = 100
loops_per_image = 1
loss = 0.0
total_inner_count = 0
for loop in range(0, loops):
    print "Loop ", loop+1, "/", loops

    for filenum, filename in enumerate(train_files):
        print "File ", filenum+1, "/", len(train_files)
        print "    ", filename

        f = rt.TFile.Open(filename)
        tree = f.Get('trainingdata')

        for entry in tree:

            print "======== EVENT {} ========".format(events+1)

            coord_plane0_t = entry.DataBranch.coord_t_pyarray(0)
            coord_plane1_t = entry.DataBranch.coord_t_pyarray(1)
            coord_plane2_t = entry.DataBranch.coord_t_pyarray(2)

            feat_plane0_t = entry.DataBranch.feat_t_pyarray(0)
            feat_plane1_t = entry.DataBranch.feat_t_pyarray(1)
            feat_plane2_t = entry.DataBranch.feat_t_pyarray(2)

            coord_dict = {0: coord_plane0_t, 1: coord_plane1_t, 2: coord_plane2_t}
            feat_dict = {0: feat_plane0_t, 1: feat_plane1_t, 2: feat_plane2_t}
            
            print "Num instances (plane 0, 1, 2): ({}, {}, {})".format(entry.DataBranch.num_instances_plane(0), \
                                                                    entry.DataBranch.num_instances_plane(1), \
                                                                    entry.DataBranch.num_instances_plane(2))

            instances_uneven = False
            if (entry.DataBranch.num_instances_plane(0) != entry.DataBranch.num_instances_plane(1)) or \
                (entry.DataBranch.num_instances_plane(0) != entry.DataBranch.num_instances_plane(2)):
                instances_uneven = True

            # Visualize, check correctness
            if (args.visualize != None) or (args.vis_uneven):
                visualize(entry, coord_dict, events, IMG_WIDTH, IMG_HEIGHT, IMG_BUFF, args.visualize, \
                        args.vis_uneven, instances_uneven, particle_names)

            # Train on each plane
            for iterate in range(loops_per_image):
                for plane in range(3):
                    print "Plane: ", plane

                    optimizer.zero_grad()

                    coord_plane = torch.from_numpy(coord_dict[plane]).float().to(device)
                    feat_plane =  torch.from_numpy(feat_dict[plane]).float().to(device)

                    offsets, seeds = model.forward_features(coord_plane, feat_plane, 1)


                    # get positions of instance pixels per instance
                    instances = entry.DataBranch.get_instance_binaries(plane)
                    instances = torch.Tensor(instances).float().to(device)
                    instances.requires_grad_(True)

                    num_instances = entry.DataBranch.num_instances_plane(plane)

                    # Get each combined labeled class map, for each class, arranged in matrix
                    # Get the indices of each type
                    class_maps = []
                    types = []
                    for key in particle_list:
                        class_maps.append(entry.DataBranch.get_class_map(plane, key, 1))
                        types.append(list(entry.DataBranch.type_indices(plane, key, 1)))
                    class_maps = torch.Tensor(class_maps).float().to(device)
                    class_maps.requires_grad_(False)

                    offsets = offsets.to(device)
                    seeds = seeds.to(device)
                    
                    offsets.requires_grad_(True)
                    seeds.requires_grad_(True)

                    loss_tot = criterion(coord_plane, offsets, seeds, instances, class_maps, num_instances, types, device, verbose=args.verbose, iterator=iterate)
                    
                    loss = loss_tot[0]

                    if (args.track_loss):
                        if (total_inner_count % args.track_loss) == 0:
                            loss_tracking.append(numpy.double(loss.detach().to('cpu')))
                            class_loss.append(loss_tot[1].detach().to('cpu'))
                            seed_loss.append(loss_tot[2].detach().to('cpu'))
                            sigma_smooth_loss.append(loss_tot[3].detach().to('cpu'))

                    loss.backward()
                    optimizer.step()

                    total_inner_count += 1

            events += 1
            # if events == 1:
            #     exit()

# save_dict = {'class_loss': class_loss,
#              'total_loss': loss_tracking,
#              'seed_loss': seed_loss,
#              'sigma_loss': sigma_smooth_loss,
#              "xaxis": [args.track_loss*i for i, elem in enumerate(loss_tracking)]}
# pickle.dump(save_dict, open('loss.pkl', 'wb'))

if args.save:
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train': train_files,
        'test': test_files,
        'validate': validation_files,
        'loss_tracking': loss_tracking,
        'class_loss': class_loss,
        'seed_loss': seed_loss,
        'sigma_smooth_loss': sigma_smooth_loss
    }
    torch.save(state, args.save)

print 'End'
