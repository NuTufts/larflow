# Visualize Voxels

from __future__ import print_function
import os,sys,argparse,json
from array import *
from ctypes import c_int
from math import log

parser = argparse.ArgumentParser("Visualize Voxel Data")
parser.add_argument("input_file",type=str,help="input root file [required]")
parser.add_argument("entry",type=int,help="entry # [required]")
#parser.add_argument("-in", "--input-file",required=True,type=str,help="input root file [required]")
#parser.add_argument("-o","--output",required=True,type=str,help="output file name [required]")
# maybe add a line for num entries
#parser.add_argument("-n",   "--nentries",type=int,default=-1,help="Number of entries to run [default: -1 (all)]")
args = parser.parse_args()

import numpy as np
import ROOT as rt
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow
larcv.SetPyUtil()

rt.gStyle.SetOptStat(0)

#ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
#ioll.add_in_filename(  args.input_larlite )
#ioll.open()

import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import lardly

color_by_options = ["charge"]
colorscale = "Viridis"
option_dict = []
for opt in color_by_options:
    option_dict.append( {"label":opt,"value":opt} )

# load tree
tfile = rt.TFile(args.input_file,'open')
preppedTree  = tfile.Get('preppedTree')
print("Got tree")

nentries = preppedTree.GetEntries()
print("NENTRIES: ",nentries)

ientry = args.entry
print("Entry requested is: ",ientry)

input_files = rt.std.vector("std::string")()
input_files.push_back(args.input_file)

dataloader = larflow.lightmodel.DataLoader(input_files)
#dataloader.load_entry(ientry)

#for ientry in range(nentries):
#    dataloader.load_entry( ientry )
#    dataloader.make_arrays()

#arr = []
#data_dict = dataloader.make_arrays()
#print("data_dict['flash_info']: ", data_dict["flash_info"])
#print("shape:", data_dict["flash_info"].shape)
#print(data_dict.items())
#print("data_dict['charge_array']: ", data_dict["charge_array"])
#print("shape:", data_dict["charge_array"].shape)

#print("data_dict['coord_array']: ", data_dict["coord_array"])
#print("shape:", data_dict["coord_array"].shape)

#arr = data_dict["coord_array"].reshape((10,4))

#print("arr: ",arr)
#print(arr.shape)

#arr = dataloader.make_arrays()

#print(arr.shape)


# PLOTTING STUFF
from larlite import larutil
dv = larutil.LArProperties.GetME().DriftVelocity()
detdata = lardly.DetectorOutline()

def make_figures(ientry,minprob=0.0):

#    print("making figures for ientry={} plot-by={}".format(ientry,plotby))
    print("making figures for ientry={}".format(ientry))
    global preppedTree

    dataloader.load_entry(ientry)
    data_dict = dataloader.make_arrays()
    
    print("voxel entries: ",data_dict["coord_array"].shape)

    traces_v = []
    
    color = data_dict["charge_array"]
    # 3D trace
    voxtrace = {
        "type":"scatter3d",
        "x":data_dict["coord_array"][:,0],
        "y":data_dict["coord_array"][:,1],
        "z":data_dict["coord_array"][:,2],
        "mode":"markers",
        "name":"voxels",
        "marker":{"color":color,
                  "size":10,
                  "opacity":1}}
    traces_v.append(voxtrace)

    voxtrace["marker"]["colorscale"]="Viridis"

    return traces_v
    
def test():
    pass

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server

axis_template = {
    "showbackground": True,
    #"backgroundcolor": "#141414", # black
    #"gridcolor": "rgba(255, 255, 255)",
    #"zerolinecolor": "rgba(255, 255, 255)",
    "backgroundcolor": "rgba(100, 100, 100,0.5)",
    "gridcolor": "rgb(50, 50, 50)",
    "zerolinecolor": "rgb(0, 0, 0)",
}


plot_layout = {
    "title": "",
    "height":800,
    "margin": {"t": 0, "b": 0, "l": 0, "r": 0},
    "font": {"size": 12, "color": "black"},
    "showlegend": False,
    "plot_bgcolor": "#ffffff",
    "paper_bgcolor": "#ffffff",
    "scene": {
        "xaxis": axis_template,
        "yaxis": axis_template,
        "zaxis": axis_template,
        "aspectratio": {"x": 1, "y": 1, "z": 3},
        "camera": {"eye": {"x": 1, "y": 1, "z": 1},
                   "up":dict(x=0, y=1, z=0)},
        "annotations": [],
    },
}

eventinput = dcc.Input(
    id="input_event",
    type="number",
    placeholder="Input Event")

minprob_input = dcc.Input(
    id="min_prob",
    type="text",
    placeholder="0.0")

plotopt = dcc.Dropdown(
    options=option_dict,
    value='charge',
    id='plotbyopt',
    )


app.layout = html.Div( [
    html.Div( [ eventinput,
                plotopt,
		html.Button("Plot",id="plot")
    ] ),
    html.Hr(),
    html.Div( [
	dcc.Graph(
            id="det3d",
            figure={
                "data": [],
                "layout": plot_layout,
            },
            config={"editable": True, "scrollZoom": False},
        )],
              className="graph__container"),
    html.Div(id="out")
] )

@app.callback(
    [Output("det3d","figure"),
     Output("out","children")],
    [Input("plot","n_clicks")],
    [State("input_event","value"),
     State("plotbyopt","value"),
     State("det3d","figure")],
    )
def cb_render(*vals):
    if vals[1] is None:
        print("Input event is none")
        raise PreventUpdate
    if vals[1]>=nentries or vals[1]<0:
        print("Input event is out of range")
        raise PreventUpdate
    if vals[2] is None:
        print("Plot-by option is None")
        raise PreventUpdate

    cluster_traces_v = make_figures(int(vals[1]),minprob=0.0)
#    cluster_traces_v = make_figures(int(vals[1]),plotby=vals[2],minprob=0.0)
    vals[-1]["data"] = cluster_traces_v
    return vals[-1],"event requested: {} {}".format(vals[1],vals[2])

if __name__ == "__main__":
    app.run_server(debug=True)
