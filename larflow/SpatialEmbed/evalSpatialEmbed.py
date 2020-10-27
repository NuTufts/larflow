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

import matplotlib.pyplot as plt
from SpatialEmbed import SpatialEmbed, spatialembed_loss, post_process
from particle_list import *
from SpatialEmbedVisualization import *

NUM_TYPES = len(particle_list)

parser = argparse.ArgumentParser("Evaluate Model")
parser.add_argument('-m', "--model", type=str, required=True, help='Pytorch model file')
parser.add_argument("-v", "--verbose", action='store_true', help="Verbose")
args = parser.parse_args()

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

state = torch.load(args.model)

train_files = state['train']
test_files = state['test']
validate_files = state['validate']


# plt.plot(range(0, len(state['loss_tracking'] * 10), 10), state['loss_tracking'])
# plt.show()


model = SpatialEmbed(features_per_layer=NUM_TYPES)
model.load_state_dict(state['state_dict'])
model.eval()

if len(test_files) == 0:
    test_files = train_files

events = 0
for filenum, filename in enumerate(test_files):
    print "File ", filenum+1, "/", len(test_files)
    print "\t", filename

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


        # x, y, dummy = zip(*coord_plane0_t)
        # plt.plot(x, y, '.', markersize=5, color='black')
        # for inst_idx in xrange(entry.DataBranch.num_instances_plane(0)):
        #     inst_xs, inst_ys = zip(*entry.DataBranch.instance(0, inst_idx))
        #     plt.plot(inst_xs, inst_ys, '.', markersize=7, color='red')
        # plt.show()


        # Train on each plane
        for iterate in range(1):
            for plane in range(1):
                if args.verbose:
                    print "Plane: ", plane

                coord_plane = torch.from_numpy(coord_dict[plane]).float().to(device)
                feat_plane =  torch.from_numpy(feat_dict[plane]).float().to(device)

                offsets, seeds = model.forward_features(coord_plane, feat_plane, 1)

                results = numpy.array(post_process(coord_plane.to('cpu'), offsets.detach().cpu(), seeds.detach().cpu()))

                # Truth
                truth_types = []  
                for particle_name in particle_list:
                    truth_types.append(list(entry.DataBranch.type_indices(plane, particle_name, 1)))


                if args.verbose:
                    print "Truth:   "
                    for i, type_list in enumerate(truth_types):
                        print "\t", particle_list[i], ": "
                        print "\t    ",
                        for truth_idx in type_list:
                            print len(entry.DataBranch.instance(plane, int(truth_idx))), ", ",
                        print

                    print "-----------------------------------"

                    print "Network:    "
                    # Results of the network
                    for i, particle_name in enumerate(particle_list):
                        print "\t", particle_name, ": "
                        print "\t", "    ",
                        for elem in results[i]:
                            print len(elem), ", ",
                        print

                    print "-----------------------------------"

                

        events += 1
        # if events == 1:
        #     exit()
