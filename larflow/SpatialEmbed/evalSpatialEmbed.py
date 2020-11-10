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
from evalMetric import Evaluation

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


# I=0, U=0, return 1, label "fake"
# I=0, U>0, return 0, label "real"
# I>0, U=0, impossible
# I>0, U>0, return I/U, label "real"
def IoU(outputs, labels):
    epsilon = 1e-6
    intersection = numpy.logical_and(outputs, labels)
    union = numpy.logical_or(outputs, labels)
    isum, usum = intersection.sum(), union.sum()

    if (isum == 0) and (usum == 0): return (1, "I0U0")
    elif (isum == 0) and (usum > 0): return (0, "I0U1")
    else: return (((isum + epsilon)/(float(usum) + epsilon)), "I1U1")


def all_instances_from_seedmap(seeds):
    folded_binary = (seeds > 0.5).any(axis=1)
    return folded_binary

def evaluation_metric(coord_t, entry, results, offsets, seeds, plane, entryname, verbose=False):
    # Three different evaluation metrics:
    #     1) Binary "yes or no" this pixel is in any instance
    #     2) Evaluation for each type, purely based on what the offsets and seeds give
    #     3) Evaluation for each type, where overlaps between types are eliminated

    dtype={'names':['f{}'.format(i) for i in range(2)],  # for fast union of arrays holding tuples
           'formats':2*[numpy.float32]}                  # https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays

    learned_class_maps = []
    for class_type in results:  # union together all instances of each type, and put that into one array learned_class_maps
        folded_class = []
        if len(class_type) > 0:
            folded_class = numpy.array(class_type[0])
            for i in range(1, len(class_type)):
                folded_class = numpy.union1d(folded_class.view(dtype), numpy.array(class_type[i]).view(dtype))
            folded_class = folded_class.view(numpy.float32).reshape(-1, 2)
        learned_class_maps.append(folded_class)

    # Make the binary maps
    learned_binary_maps = []
    for class_map in learned_class_maps: # convert the learned_class_map sparse arrays to a binary map
                                         # of the same size as coord_t. aka un-sparsify
        binary_set = set([(elem[0], elem[1]) for elem in class_map])
        binary_map = [((p1, p2) in binary_set) for (p1, p2, p3) in coord_t]
        learned_binary_maps.append(binary_map)
    learned_binary_maps = numpy.array(learned_binary_maps)

    # make the segregated binary maps
    learned_binary_maps_collective = numpy.zeros(numpy.shape(seeds))
    folded_learned_binary_maps = numpy.array(learned_binary_maps).any(axis=0)
    for i in range(numpy.shape(seeds)[0]):  # make a binary map like above, except with no overlaps in a pixel
        if folded_learned_binary_maps[i]:   # reach into the seeds and pick out the highest probability to determine class
            learned_binary_maps_collective[i][numpy.argmax(numpy.array(seeds[i]) * learned_binary_maps[:,i])] = 1  # class the duplicates belong to. multiply by 

    learned_binary_maps_collective = numpy.transpose(learned_binary_maps_collective)  # to get each type as a row
    
    seeds = seeds.t() # to get each type as a row

    # 1
    folded_instance_binary_truth = numpy.array(entry.DataBranch.get_instance_binaries(plane)).any(axis=0)
    folded_seeds = numpy.array(seeds > 0.5).any(axis=0)
    of_any_instance_seeds = IoU(folded_seeds, folded_instance_binary_truth)

    if verbose: print "of_any_instance_seeds: ", of_any_instance_seeds

    # 2
    of_any_instance_class = IoU(folded_learned_binary_maps, folded_instance_binary_truth)
    if verbose: print "of_any_instance_class: ", of_any_instance_class

    # 3 & 4
    types_individual_values = []
    types_collective_values = []
    for i, particle_name in enumerate(particle_list):
        class_map = entry.DataBranch.get_class_map(plane, particle_name, 1)
        types_individual_values.append(IoU(learned_binary_maps[i], class_map))
        types_collective_values.append(IoU(learned_binary_maps_collective[i], class_map))

    if verbose: print "types_individual: ", ', '.join([str(elem) for elem in types_individual_values])
    if verbose: print "types_collective: ", ', '.join([str(elem) for elem in types_collective_values])

    # ============================ debugging
    # temp_coord_t = numpy.array(coord_t)
    # x, y, dummy = zip(*numpy.array(coord_t))

    # plt.scatter(x, y, marker='.', c=list(learned_binary_maps[0]), cmap=plt.cm.bwr)
    # plt.title("Type 1 greedy")
    # plt.savefig("type1greedy.png")
    # plt.clf()

    # plt.scatter(x, y, marker='.', c=list(learned_binary_maps[1]), cmap=plt.cm.bwr)
    # plt.title("Type 2 greedy")
    # plt.savefig("type2greedy.png")
    # plt.clf()
    
    # plt.scatter(x, y, marker='.', c=list(learned_binary_maps[2]), cmap=plt.cm.bwr)
    # plt.title("Type 3 greedy")
    # plt.savefig("type3greedy.png")
    # plt.clf()
    
    # plt.scatter(x, y, marker='.', c=list(learned_binary_maps[3]), cmap=plt.cm.bwr)
    # plt.title("Type 4 greedy")
    # plt.savefig("type4greedy.png")
    # plt.clf()

    # plt.scatter(x, y, marker='.', c=list(learned_binary_maps[4]), cmap=plt.cm.bwr)
    # plt.title("Type 5 greedy")
    # plt.savefig("type5greedy.png")
    # plt.clf()

    # plt.scatter(x, y, marker='.', c=list(learned_binary_maps[5]), cmap=plt.cm.bwr)
    # plt.title("Type 6 greedy")
    # plt.savefig("type6greedy.png")
    # plt.clf()

    # exit()
    # ============================ debugging 

    return Evaluation(of_any_instance_seeds=of_any_instance_seeds,
                      of_any_instance_class=of_any_instance_class,
                      types_individual=types_individual_values,
                      types_collective=types_collective_values,
                      num_truth_pixels=int(folded_instance_binary_truth.sum()),
                      name=entryname)


state = torch.load(args.model)

train_files = state['train']
test_files = state['test']
validate_files = state['validate']

model = SpatialEmbed(features_per_layer=NUM_TYPES)
model.load_state_dict(state['state_dict'])
model.eval()

if len(test_files) == 0:
    test_files = train_files

evaluation_metrics = []

events = 0
for filenum, filename in enumerate(test_files):
    print "File ", filenum+1, "/", len(test_files)
    print "\t", filename

    f = rt.TFile.Open(filename)
    tree = f.Get('trainingdata')

    for entrynum, entry in enumerate(tree):
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


        for plane in range(3):
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


            # if args.verbose:
            #     print "Truth:   "
            #     for i, type_list in enumerate(truth_types):
            #         print "\t", particle_list[i], ": "
            #         print "\t    ",
            #         for truth_idx in type_list:
            #             print len(entry.DataBranch.instance(plane, int(truth_idx))), ", ",
            #         print

            #     print "-----------------------------------"

            #     print "Network:    "
            #     # Results of the network
            #     for i, particle_name in enumerate(particle_list):
            #         print "\t", particle_name, ": "
            #         print "\t", "    ",
            #         for elem in results[i]:
            #             print len(elem), ", ",
            #         print

            #     print "-----------------------------------"

            evaluation_metrics.append(evaluation_metric(coord_dict[plane], 
                                                        entry, results, 
                                                        offsets.detach(), 
                                                        seeds.detach(), 
                                                        plane, 
                                                        "{}_{}e_{}p".format(filename, entrynum, plane),
                                                        verbose=True))
            
        events += 1
        if events == 3:
            break

        if (events % 1) == 0:
            print("saving")
            save_dict = {"scores": evaluation_metrics}
            save_name = "{}_test_evaluations.pickle".format(os.path.splitext(args.model)[0])
            pickle.dump(save_dict, open(save_name, 'wb'))
                


print evaluation_metrics

save_dict = {"scores": evaluation_metrics}
save_name = "{}_test_evaluations.pickle".format(os.path.splitext(args.model)[0])
pickle.dump(save_dict, open(save_name, 'wb'))
