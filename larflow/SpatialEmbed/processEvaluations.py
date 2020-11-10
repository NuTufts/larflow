import sys, os
import pickle
import argparse
import torch
import matplotlib.pyplot as plt
import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow
from evalMetric import Evaluation
from SpatialEmbed import SpatialEmbed, spatialembed_loss, post_process
from particle_list import *
from SpatialEmbedVisualization import *

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
    model = SpatialEmbed()
    model.load_state_dict(state['state_dict'])
    model.eval()

    save_dict = {}

    for i, filename in enumerate(filenames):
        entrynum = entrynums[i]
        planes = planes[i]

        f = rt.TFile.Open(filename)
        tree = f.Get('trainingdata')

        for i, entry in enumerate(tree):
            if i != entrynum: continue

            coord_plane0_t = entry.DataBranch.coord_t_pyarray(0)
            coord_plane1_t = entry.DataBranch.coord_t_pyarray(1)
            coord_plane2_t = entry.DataBranch.coord_t_pyarray(2)

            feat_plane0_t = entry.DataBranch.feat_t_pyarray(0)
            feat_plane1_t = entry.DataBranch.feat_t_pyarray(1)
            feat_plane2_t = entry.DataBranch.feat_t_pyarray(2)

            coord_dict = {0: coord_plane0_t, 1: coord_plane1_t, 2: coord_plane2_t}
            feat_dict = {0: feat_plane0_t, 1: feat_plane1_t, 2: feat_plane2_t}
            
            coord_plane = torch.from_numpy(coord_dict[plane]).float().to(device)
            feat_plane =  torch.from_numpy(feat_dict[plane]).float().to(device)

            offsets, seeds = model.forward_features(coord_plane, feat_plane, 1)

            results = numpy.array(post_process(coord_plane.to('cpu'), offsets.detach().cpu(), seeds.detach().cpu()))

            save_dict[prefixes[i] + "_offsets"] = offsets
            save_dict[prefixes[i] + "_seeds"] = seeds
            save_dict[prefixes[i] + "_results"] = results

    save_name = save_location + "min_max_med_data.pkl"
    pickle.dump(save_dict, open(save_name, 'wb'))


# def save_class_plots(model, filename, entrynum, plane):
#     print "== Making Plots =="

#     NUM_TYPES = len(particle_list)
#     state = torch.load(model)
#     model = SpatialEmbed()
#     model.load_state_dict(state['state_dict'])
#     model.eval()

#     f = rt.TFile.Open(filename)
#     tree = f.Get('trainingdata')

#     for i, entry in enumerate(tree):
#         if i != entrynum: continue

#         coord_t = entry.DataBranch.coord_t_pyarray(plane)
#         feat_t = entry.DataBranch.feat_t_pyarray(plane)

#         offsets, seeds = model.forward_features(coord_plane, feat_plane, 1)
#         print len(offsets)
#         exit()


def main(evals_filename, model, verbose=False):

    scores = pickle.load(open(evals_filename, 'rb'))
    scores = scores['scores']
    scores.sort()

    min_file, min_entry, min_plane = parse_file_entry_name(scores[0].location_name)
    med_file, med_entry, med_plane = parse_file_entry_name(scores[len(scores)/2].location_name)
    max_file, max_entry, max_plane = parse_file_entry_name(scores[len(scores)-1].location_name)

    # Convert to larbys file structures
    # min_file, med_file, max_file = os.path.basename(min_file), os.path.basename(med_file), os.path.basename(max_file)
    # min_file, med_file, max_file = ["train_files/" + filename for filename in [min_file, med_file, max_file]]
    print min_file, med_file, max_file

    save_min_med_max_pkl(model, ['min', 'med', 'max'], 
                        [min_file, med_file, max_file],
                        [min_entry, med_entry, max_entry],
                        [min_plane, med_plane, max_plane],
                        save_location="/cluster/tufts/wongjiradlab/jhwang11/ubdl/larflow/larflow/SpatialEmbed/trained_models/"):


if __name__ == '__main__':
    parser = argparse.ArgumentParser("View Results of Trained Model")
    parser.add_argument('-m', "--model", type=str, required=True, help='Pytorch model')   
    parser.add_argument('-r', "--results", type=str, required=True, help='Pickle of results from evalSpatialEmbed.py')
    parser.add_argument("-v", "--verbose", action='store_true', help="Verbose")
    args = parser.parse_args()

    main(args.results, args.model, args.verbose)