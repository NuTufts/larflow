from __future__ import print_function
import os,sys,argparse,json
from ctypes import c_int
from math import log

parser = argparse.ArgumentParser("Visuzalize Voxel Data")
parser.add_argument("input_file",type=str,help="file produced by 'prep_spatialembed.py'")
parser.add_argument("-mc","--input-mcinfo",type=str,default=None,required=False,help="input larlite mcinfo")
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


color_by_options = ["q_yplane","cluster"]
colorscale = "Viridis"
option_dict = []
for opt in color_by_options:
    option_dict.append( {"label":opt,"value":opt} )

# LOAD TREES
infile = rt.TFile(args.input_file)
io = infile.Get("s3dembed")
nentries = io.GetEntries()
print("NENTRIES: ",nentries)

voxelloader = larflow.spatialembed.Prep3DSpatialEmbed()
voxelloader.loadTreeBranches( io )

if args.input_mcinfo:
    HAS_MC = True
    print("LARLITE MCINFO PROVIDED")
    ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
    ioll.add_in_filename( args.input_mcinfo )
    ioll.open()
else:
    HAS_MC = False
    
    
from larlite import larutil
dv = larutil.LArProperties.GetME().DriftVelocity()
detdata = lardly.DetectorOutline()

def make_figures(entry,plotby="cluster",minprob=0.0):

    print("making figures for entry={} plot-by={}".format(entry,plotby))
    global io
    global ioll

    voxel_dim = voxelloader.getVoxelizer().get_dim_len()
    nvoxels   = voxelloader.getVoxelizer().get_nvoxels()
    voxorigin = voxelloader.getVoxelizer().get_origin()
    data_dict = voxelloader.getTreeEntryDataAsArray(entry)
    print("voxel entries: ",data_dict["coord_t"].shape)


    if plotby=="q_yplane":
        # color by charge of collection plane
        color = data_dict["feat_t"][:,2]
    elif plotby=="q_vplane":
        # color by charge of v plakne
        color = data_dict["feat_t"][:,1]
    elif plotby=="q_uplane":
        # color by charge of u plane
        color = data_dict["feat_t"][:,0]
    elif plotby=="cluster":
        # color by instance index
        color = np.zeros( (data_dict["coord_t"].shape[0],3) )
        ninstances = data_dict["instance_t"].max()
        print("Number of instances: ",ninstances)
        print("color shape: ",color.shape)
        for iid in range(1,ninstances+1):
            print("instance_t shape: ",data_dict["instance_t"].shape)
            idmask = data_dict["instance_t"]==iid
            print("idmask: ",idmask.shape)
            randcolor = np.random.rand(3)*255
            print("instance[",iid,"] color: ",randcolor)
            color[idmask,:] = randcolor
    else:
        raise ValueError("unrecognized plot option:",plotby)
            
    # conversion of voxel indices to coordinate space
    fcoord_t = np.zeros( data_dict["coord_t"].shape )
    for i in range(3):
        conversion = voxel_dim.at(i)/nvoxels.at(i)
        print("dim[",i,"] conversion")
        fcoord_t[:,i] = data_dict["coord_t"][:,i]*conversion + voxorigin[i]
    
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
    if plotby!="cluster":
        voxtrace["marker"]["colorscale"]="Viridis"

    traces_v = detdata.getlines()    
    traces_v.append( voxtrace )

    if HAS_MC:
        ioll.go_to( entry )
        mcpg = ublarcvapp.mctools.MCPixelPGraph()
        mcpg.buildgraphonly( ioll )
        mcpg.printGraph(mcpg.findTrackID(0),False)
    
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
