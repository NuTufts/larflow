from __future__ import print_function
import os,sys,argparse,json
from ctypes import c_int
from math import log

parser = argparse.ArgumentParser("Visuzalize Voxel Data")
parser.add_argument("input_file",type=str,help="file produced by 'prep_spatialembed.py'")
#parser.add_argument("-c","--cluster",action='store_true',default=False,help="color by cluster instance")
args = parser.parse_args()

import numpy as np
import ROOT as rt
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow
larcv.SetPyUtil()

import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


import lardly


color_by_options = ["cluster","embed"]
colorscale = "Viridis"
option_dict = []
for opt in color_by_options:
    option_dict.append( {"label":opt,"value":opt} )

# LOAD TREES
infile = rt.TFile(args.input_file)
io = infile.Get("se3dcluster")
nentries = io.GetEntries()
print("NENTRIES: ",nentries)

voxelizer = larflow.voxelizer.VoxelizeTriplets()
voxelloader = larflow.spatialembed.SpatialEmbed3DNetProducts()
voxelloader.setTreeBranches( io )
    
from larlite import larutil
dv = larutil.LArProperties.GetME().DriftVelocity()
detdata = lardly.DetectorOutline()

def make_figures(entry,plotby="cluster",minprob=0.0):

    print("making figures for entry={} plot-by={}".format(entry,plotby))
    global io
    nbytes = io.GetEntry( int(entry) )
    print("num entry bytes: ",nbytes)

    voxel_dim = voxelizer.get_dim_len()
    nvoxels   = voxelizer.get_nvoxels()
    origin    = voxelizer.get_origin()
    data      = voxelloader.getEntryDataAsNDarray(entry)
    embed     = None
    if plotby=="embed":
        embed = voxelloader.getEntryEmbedPosAsNDarray(entry)
        print("embed: ",embed)
    print("voxel entries: ",data.shape)

    if plotby in ["cluster","embed"]:
        # color by instance index
        color = np.zeros( (data.shape[0],3) )
        ninstances = data[:,3].max()
        print("Number of instances: ",ninstances)
        print("color shape: ",color.shape)
        for iid in range(ninstances):
            icluster=iid+1
            idmask = data[:,3]==icluster
            print("idmask: ",idmask.shape)
            randcolor = np.random.rand(3)*255
            print("instance[",iid,"] color: ",randcolor)
            color[idmask,:] = randcolor
    else:
        raise ValueError("unrecognized plot option:",plotby)
            
    # conversion of voxel indices to coordinate space
    fcoord_t = np.zeros( (data.shape[0],3) )
    for i in range(3):
        conversion = voxel_dim.at(i)/nvoxels.at(i)
        print("dim[",i,"] conversion")
        if plotby=="cluster":
            fcoord_t[:,i] = data[:,i]*conversion + origin[i]
        elif plotby=="embed":
            fcoord_t[:,i] = embed[:,i]*voxel_dim.at(i) + origin[i]
        print(fcoord_t[:,i].mean())

    print ("fcoord: ",fcoord_t.shape," color:",color.shape)
    # 3D trace
    voxtrace = {
        "type":"scatter3d",
        "x":fcoord_t[:,0],
        "y":fcoord_t[:,1],
        "z":fcoord_t[:,2],
        "mode":"markers",
        "name":"voxels",
        "marker":{"color":color,
                  "size":1,
                  "opacity":0.5}}
    if plotby not in ["cluster","embed"]:
        voxtrace["marker"]["colorscale"]="Viridis"
    if plotby in ["embed"]:
        voxtrace["marker"]["size"] = 3
        voxtrace["marker"]["opacity"] = 1.0        

    traces_v = []
    if plotby in ["cluster","embed"]:
        traces_v += detdata.getlines()
    traces_v.append( voxtrace )
    
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
    #"plot_bgcolor": "#141414",
    #"paper_bgcolor": "#141414",
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
    value='cluster',
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
    
    cluster_traces_v = make_figures(int(vals[1]),plotby=vals[2],minprob=0.0)
    #print(cluster_traces_v)
    vals[-1]["data"] = cluster_traces_v
    return vals[-1],"event requested: {} {}".format(vals[1],vals[2])

if __name__ == "__main__":
    app.run_server(debug=True)
