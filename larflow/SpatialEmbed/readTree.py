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
import torch.optim as optim

import matplotlib.pyplot as plt
from SpatialEmbed import SpatialEmbed, spatialembed_loss

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

particle_list = [11, 13, 2212, 22, 111, 211]
particle_names = {
    11: "electron",
    13: "muon",
    2212: "proton",
    22: "photon",
    111: "pi_0",
    211: "pi_+/-"
}
colors = {}

if (args.visualize != None) or args.vis_uneven:
    colormap = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
    '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']

    keys = particle_names.keys()
    keys.sort()
    for idx, key in enumerate(keys):
        colors[key] = colormap[idx%len(colormap)]

IMG_WIDTH  = 3456
IMG_HEIGHT = 1008
IMG_BUFF = 15

events = 0

if args.input_tree_file:
    directory_files = [args.input_tree_file]
else:
    directory_files = [os.path.join(args.input_directory, file) for file in os.listdir(args.input_directory)]


NUM_TYPES = len(particle_list)
model = SpatialEmbed(features_per_layer=NUM_TYPES)
optimizer = optim.Adam(model.parameters(), lr=0.01)


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
            for plane, coord_t in coord_dict.items(): # random for each plane
                if ((args.visualize != None) and (random.random()*100 < args.visualize)) or \
                    (args.vis_uneven and instances_uneven):

                    x, y, dummy = zip(*coord_t)
                    plt.ylim(0, IMG_WIDTH + IMG_BUFF)
                    plt.xlim(0, IMG_HEIGHT + IMG_BUFF)
                    plt.title("Event {}, plane {}".format(events, plane))

                    plt.plot(x, y, '.', markersize=5, color='black')
                    for inst_idx in xrange(entry.DataBranch.num_instances_plane(plane)):
                        inst_xs, inst_ys = zip(*entry.DataBranch.instance(plane, inst_idx))
                        type_inst = abs(entry.DataBranch.typeof_instance(plane, inst_idx))
                        plt.plot(inst_xs, inst_ys, '.', markersize=7, color=colors[type_inst], label=particle_names[type_inst])
                    plt.legend()
                    plt.show()

        # if (args.verbose):
        #     entry.DataBranch.check_instance_parity()

        # for plane in range(3):
        #     for instance in range(entry.DataBranch.num_instances_plane(plane)):
        #         instances_coords = entry.DataBranch.instance(plane, instance);
        #         instances_binary = entry.DataBranch.instance_binary(plane, instance);

        #         pixels_count = 0
        #         for elem in instances_binary:
        #             if elem == 1:
        #                 pixels_count += 1

        #         if numpy.shape(instances_coords)[0] != pixels_count:
        #             print plane, instance
        #             print numpy.shape(instances_coords), pixels_count
        

        # Train on each plane
        for plane in range(3):
            if args.verbose:
                print "Plane: ", plane

            optimizer.zero_grad()

            offsets, seeds = model.forward_features(torch.from_numpy(coord_dict[plane]), torch.from_numpy(feat_dict[plane]), 1)

            num_instances = entry.DataBranch.num_instances_plane(plane)

            # get positions of instance pixels per instance
            instances = entry.DataBranch.get_instance_binaries(plane)
            instances = torch.Tensor(instances)

            # Get each combined labeled class map, for each class, arranged in matrix
            # Get the indices of each type
            class_maps = []
            types = []
            for key in particle_list:
                class_maps.append(entry.DataBranch.get_class_map(plane, key, 1))
                types.append(list(entry.DataBranch.type_indices(plane, key, 1)))
            class_maps = torch.Tensor(class_maps)

            loss = spatialembed_loss(offsets, seeds, instances, class_maps, num_instances, types, verbose=args.verbose)
            loss.backward()
            optimizer.step()

        events += 1
        if events == 5:
            exit()


print 'End'