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
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from SpatialEmbed import SpatialEmbed, spatialembed_loss, post_process
from particle_list import *
from SpatialEmbedVisualization import *

parser = argparse.ArgumentParser("Read TTree")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-d', "--input-directory", type=str, help='TTree input directory')
group.add_argument("-f", "--input-tree-file", type=str,help="Input TTreee file [required]")
parser.add_argument("-p", "--visualize",required=False,type=int,help="Percentage [0-100] of events to visualize")
parser.add_argument("-u", "--vis-uneven", action='store_true', help="Visualize all cases where num instances is uneven amongst planes")
parser.add_argument("-v", "--verbose", action='store_true', help="Verbose")
args = parser.parse_args()

if (args.visualize) and (args.visualize < 0 or args.visualize > 100):
    raise ValueError('Visualize percentage must be between 0-100.')

if args.input_tree_file:
    directory_files = [args.input_tree_file]
else:
    directory_files = [os.path.join(args.input_directory, file) for file in os.listdir(args.input_directory)]

if (args.visualize != None) or args.vis_uneven:
    visualization_setup(particle_names)

IMG_WIDTH  = 3456
IMG_HEIGHT = 1008
IMG_BUFF = 15
NUM_TYPES = len(particle_list)

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SpatialEmbed(features_per_layer=NUM_TYPES).to(device)
model = model.train()

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = spatialembed_loss

events = 0
for filename in directory_files:
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
        for iterate in range(5000):
            for plane in range(1):
                if args.verbose:
                    print "Plane: ", plane

                # if iterate==1:
                #     x, y, dummy = zip(*coord_dict[plane])

                #     plt.plot(x, y, '.', markersize=5, color='black')
                #     for inst_idx in xrange(entry.DataBranch.num_instances_plane(plane)):
                #         inst_xs, inst_ys = zip(*entry.DataBranch.instance(plane, inst_idx))
                #         type_inst = abs(entry.DataBranch.typeof_instance(plane, inst_idx))
                #         plt.plot(inst_xs, inst_ys, '.', markersize=7, color='red')
                #     plt.show()
                #     exit(1)

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


                loss = criterion(coord_plane, offsets, seeds, instances, class_maps, num_instances, types, device, verbose=True, iterator=iterate)
                
                loss.backward()
                optimizer.step()


        coord_plane = torch.from_numpy(coord_dict[0]).float().to(device)
        feat_plane =  torch.from_numpy(feat_dict[0]).float().to(device)

        offsets, seeds = model.forward_features(coord_plane, feat_plane, 1)
        print(numpy.shape(post_process(coord_plane.to('cpu'), offsets.detach().cpu(), seeds.detach().cpu())))

        events += 1
        if events == 1:
            exit()

print 'End'
