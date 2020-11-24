import sys, os
import pickle
import argparse
import torch
# import matplotlib.pyplot as plt
import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow
import numpy
from evalMetric import Evaluation
from SpatialEmbed import SpatialEmbed, spatialembed_loss, post_process
from particle_list import *
# from SpatialEmbedVisualization import *

def parse_file_entry_name(result_name):
    path, extension = os.path.splitext(result_name)
    extension, entry, plane = extension.split("_")
    entry, plane = int(entry[:-1]), int(plane[:-1])
    path = path + extension 
    return path, entry, plane

# def save_loss_plot(model):
#     print "== Plotting loss =="
#     state = torch.load(model)
#     loss_track = state['loss_tracking']
#     x_axis = [i*15 for i, elem in enumerate(loss_track)]
    
#     # plt.ylim((0, 120000))
#     plt.xlim((0, 500))
#     plt.plot(x_axis, loss_track)
#     plt.title("Loss Plot 1iter")
#     plt.xlabel('Events*Planes')
#     plt.ylabel('Loss')

#     filename = 'plots/' + os.path.basename(model)[:-3] + "_loss.png" 
#     plt.savefig(filename)

def save_min_med_max_pkl(model, prefixes, filenames, entrynums, planes, save_location=""):
    print "== Making data pickles =="
    NUM_TYPES = len(particle_list)
    state = torch.load(model)
    model = SpatialEmbed(features_per_layer=NUM_TYPES)
    model.load_state_dict(state['state_dict'])
    model.eval()

    save_dict = {}

    for i, filename in enumerate(filenames):
        entrynum = entrynums[i]
        plane = planes[i]

        f = rt.TFile.Open(filename)
        tree = f.Get('trainingdata')

        for j, entry in enumerate(tree):
            if j != entrynum: continue

            coord_plane0_t = entry.DataBranch.coord_t_pyarray(0)
            coord_plane1_t = entry.DataBranch.coord_t_pyarray(1)
            coord_plane2_t = entry.DataBranch.coord_t_pyarray(2)

            feat_plane0_t = entry.DataBranch.feat_t_pyarray(0)
            feat_plane1_t = entry.DataBranch.feat_t_pyarray(1)
            feat_plane2_t = entry.DataBranch.feat_t_pyarray(2)

            coord_dict = {0: coord_plane0_t, 1: coord_plane1_t, 2: coord_plane2_t}
            feat_dict = {0: feat_plane0_t, 1: feat_plane1_t, 2: feat_plane2_t}
            
            coord_plane = torch.from_numpy(coord_dict[plane]).float()
            feat_plane =  torch.from_numpy(feat_dict[plane]).float()

            offsets, seeds = model.forward_features(coord_plane, feat_plane, 1)

            save_dict[prefixes[i] + "_offsets"] = offsets
            save_dict[prefixes[i] + "_seeds"] = seeds
            save_dict[prefixes[i] + "_coord"] = coord_dict[plane]

    save_name = save_location + "min_max_med_data.pkl"
    pickle.dump(save_dict, open(save_name, 'wb'))

def save_single_entry_pkl(model, filename, entrynum, plane, save_location=""):
    print "== Making data pickles =="
    NUM_TYPES = len(particle_list)
    state = torch.load(model)
    model = SpatialEmbed(features_per_layer=NUM_TYPES)
    model.load_state_dict(state['state_dict'])
    model.eval()

    save_dict = {}

    f = rt.TFile.Open(filename)
    tree = f.Get('trainingdata')

    for j, entry in enumerate(tree):
        if j != entrynum: continue

        coord_plane0_t = entry.DataBranch.coord_t_pyarray(0)
        coord_plane1_t = entry.DataBranch.coord_t_pyarray(1)
        coord_plane2_t = entry.DataBranch.coord_t_pyarray(2)

        feat_plane0_t = entry.DataBranch.feat_t_pyarray(0)
        feat_plane1_t = entry.DataBranch.feat_t_pyarray(1)
        feat_plane2_t = entry.DataBranch.feat_t_pyarray(2)

        coord_dict = {0: coord_plane0_t, 1: coord_plane1_t, 2: coord_plane2_t}
        feat_dict = {0: feat_plane0_t, 1: feat_plane1_t, 2: feat_plane2_t}
        
        coord_plane = torch.from_numpy(coord_dict[plane]).float()
        feat_plane =  torch.from_numpy(feat_dict[plane]).float()

        offsets, seeds = model.forward_features(coord_plane, feat_plane, 1)

        save_dict["offsets"] = offsets
        save_dict["seeds"] = seeds
        save_dict["coord"] = coord_dict[plane]

    save_name = save_location + "single_entry.pkl"
    pickle.dump(save_dict, open(save_name, 'wb'))


# def save_class_plots(minmax_pickle, prefixes, filenames, entrynums, planes):
#     visualization_setup(particle_names)

#     minmaxes = pickle.load(open(minmax_pickle,'rb'))
#     min_offsets, min_seeds, min_coord = minmaxes['min_offsets'], minmaxes['min_seeds'], minmaxes['min_coord']
#     med_offsets, med_seeds, med_coord = minmaxes['med_offsets'], minmaxes['med_seeds'], minmaxes['med_coord']
#     max_offsets, max_seeds, max_coord = minmaxes['max_offsets'], minmaxes['max_seeds'], minmaxes['max_coord']

#     offsets = [minmaxes['min_offsets'], minmaxes['med_offsets'], minmaxes['max_offsets']]
#     seeds = [minmaxes['min_seeds'], minmaxes['med_seeds'], minmaxes['max_seeds']]
#     coords = [minmaxes['min_coord'], minmaxes['med_coord'], minmaxes['max_coord']]

#     for i, filename in enumerate(filenames):
#         entrynum = entrynums[i]
#         plane = planes[i]

#         f = rt.TFile.Open(filename)
#         tree = f.Get('trainingdata')

#         for j, entry in enumerate(tree):
#             if j == entrynum:
#                 save_plots_for_entry(numpy.array(coords[i]),
#                                      offsets[i].detach(),
#                                      seeds[i].detach(),
#                                      entry, plane,
#                                      prefixes[i])


# def save_plots_for_entry(coord_t, offsets, seeds, entry, plane, prefix):
#     print "== Making Plots =="

#     # Plot Truth
#     x, y, dummy = zip(*coord_t)
#     plt.scatter(y, x, marker='.', c='black') # y, x because col, row
#     truth_binaries = entry.DataBranch.get_instance_binaries(plane)
#     for inst_idx, instance in enumerate(truth_binaries):
#         if sum(instance) == 0: continue
#         typeof_inst = abs(entry.DataBranch.typeof_instance(plane, inst_idx))
#         print typeof_inst
#         new_coord = [elem for i, elem in enumerate(coord_t) if instance[i]]
#         x, y, dummy = zip(*new_coord)
#         plt.scatter(y, x, marker='.', c=colors[typeof_inst], label=particle_names[typeof_inst])

#     plt.title(prefix + " Truth")
#     plt.legend()
#     plt.savefig(prefix + '_truth.png')
#     plt.clf()

#     # Plot Results
#     results = post_process(coord_t, offsets, seeds)

#     dtype={'names':['f{}'.format(i) for i in range(2)],  # for fast union of arrays holding tuples
#             'formats':2*[numpy.float32]}                  # https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays

#     learned_class_maps = []
#     for class_type in results:  # union together all instances of each type, and put that into one array learned_class_maps
#         folded_class = []
#         if len(class_type) > 0:
#             folded_class = numpy.array(class_type[0])
#             for i in range(1, len(class_type)):
#                 folded_class = numpy.union1d(folded_class.view(dtype), numpy.array(class_type[i]).view(dtype))
#             folded_class = folded_class.view(numpy.float32).reshape(-1, 2)
#         learned_class_maps.append(folded_class)

#     # Make the binary maps
#     learned_binary_maps = []
#     for class_map in learned_class_maps: # convert the learned_class_map sparse arrays to a binary map
#                                          # of the same size as coord_t. aka un-sparsify
#         binary_set = set([(elem[0], elem[1]) for elem in class_map])
#         binary_map = [((p1, p2) in binary_set) for (p1, p2, p3) in coord_t]
#         learned_binary_maps.append(binary_map)
#     learned_binary_maps = numpy.array(learned_binary_maps)

#     # make the segregated binary maps
#     learned_binary_maps_collective = numpy.zeros(numpy.shape(seeds))
#     folded_learned_binary_maps = numpy.array(learned_binary_maps).any(axis=0)
#     for i in range(numpy.shape(seeds)[0]):  # make a binary map like above, except with no overlaps in a pixel
#         if folded_learned_binary_maps[i]:   # reach into the seeds and pick out the highest probability to determine class
#             learned_binary_maps_collective[i][numpy.argmax(numpy.array(seeds[i]) * learned_binary_maps[:,i])] = 1  # class the duplicates belong to. multiply by 

#     learned_binary_maps_collective = numpy.transpose(learned_binary_maps_collective)  # to get each type as a row

#     x, y, dummy = zip(*coord_t)
#     plt.scatter(y, x, marker='.', c='black')
#     for typename_idx, typename in enumerate(particle_list):
#         if sum(learned_binary_maps_collective[typename_idx]) == 0: continue
#         new_coord = [elem for i, elem in enumerate(coord_t) if learned_binary_maps_collective[i]]
#         x, y, dummy = zip(*new_coord)
#         plt.scatter(y, x, marker='.', c=colors[typename], label=particle_names[typename])

#     plt.title(prefix + " Learned")
#     plt.legend()
#     plt.savefig(prefix + '_learned.png')
#     plt.clf()

#     seeds = seeds.t()
#     seeds = seeds > 0.5
#     x, y, dummy = zip(*coord_t)
#     plt.scatter(y, x, marker='.', c='black')
#     for typename_idx, typename in enumerate(particle_list):
#         if sum(seeds[typename_idx]) == 0: continue
#         new_coord = [elem for i, elem in enumerate(coord_t) if seeds[typename_idx][i]]
#         x, y, dummy = zip(*new_coord)
#         plt.scatter(y, x, marker='.', c=colors[typename], label=particle_names[typename])

#     plt.title(prefix + " Learned Seeds")
#     plt.legend()
#     plt.savefig(prefix + '_learned_seeds.png')
#     plt.clf()


def main(evals_filename, model, verbose=False):

    ##  COMMENT IF RUNNING ON MAYER
    scores = pickle.load(open(evals_filename, 'rb'))
    scores = scores['scores']
    scores.sort()

    min_file, min_entry, min_plane = parse_file_entry_name(scores[0].location_name)
    med_file, med_entry, med_plane = parse_file_entry_name(scores[len(scores)/2].location_name)
    max_file, max_entry, max_plane = parse_file_entry_name(scores[len(scores)-1].location_name)

    # Convert to larbys file structures
    min_file, med_file, max_file = os.path.basename(min_file), os.path.basename(med_file), os.path.basename(max_file)
    min_file, med_file, max_file = ["train_files/" + filename for filename in [min_file, med_file, max_file]]

    save_min_med_max_pkl(model, ['min', 'med', 'max'], 
                        [min_file, med_file, max_file],
                        [min_entry, med_entry, max_entry],
                        [min_plane, med_plane, max_plane],
                        save_location="/cluster/tufts/wongjiradlab/jhwang11/ubdl/larflow/larflow/SpatialEmbed/trained_models/")
    # If single entry
    # file_entry = 0
    # single_entry_file, single_entry_entry, single_entry_plane = parse_file_entry_name(scores[file_entry].location_name)
    # single_entry_file = "train_files/" + os.path.basename(single_entry_file)
    # save_single_entry_pkl(model, single_entry_file, single_entry_entry, single_entry_plane, 
    #                       save_location='/cluster/tufts/wongjiradlab/jhwang11/ubdl/larflow/larflow/SpatialEmbed/trained_models/')

    ##  COMMENT IF RUNNING ON CLUSTER
    # min_med_max_pkl_filename = 'trained_models/min_max_med_data.pkl'
    # save_class_plots(min_med_max_pkl_filename, 
    #                  ['min', 'med', 'max'],
    #                  [min_file, med_file, max_file],
    #                  [min_entry, med_entry, max_entry],
    #                  [min_plane, med_plane, max_plane])


if __name__ == '__main__':
    parser = argparse.ArgumentParser("View Results of Trained Model")
    parser.add_argument('-m', "--model", type=str, required=True, help='Pytorch model')   
    parser.add_argument('-r', "--results", type=str, required=True, help='Pickle of results from evalSpatialEmbed.py')
    parser.add_argument("-v", "--verbose", action='store_true', help="Verbose")
    args = parser.parse_args()

    main(args.results, args.model, args.verbose)