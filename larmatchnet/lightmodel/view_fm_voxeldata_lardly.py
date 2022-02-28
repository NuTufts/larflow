from __future__ import print_function
import os,sys

import chart_studio as cs
import chart_studio.plotly as py
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import numpy as np
import torch
from larcv import larcv
from larlite import larlite
from larflow import larflow
sys.path.append("../")
import lardly
#from larvoxel.larvoxelclass_dataset import larvoxelClassDataset
#from larvoxel_dataset import larvoxelDataset
from lightmodel.lm_dataset import LMDataset

from larlite import larutil
dv = larutil.LArProperties.GetME().DriftVelocity()
detdata = lardly.DetectorOutline()

color_by_options = ["charge"]
colorscale = "Viridis"
option_dict = []
for opt in color_by_options:
    option_dict.append( {"label":opt,"value":opt} )


# DATA LOADER
batch_size = 1
dataset = LMDataset( filelist=["test.root"], is_voxeldata=True )
nentries = len(dataset)

print("NENTRIES: ",nentries)

ientry = 1

#for i in enumerate(dataset):
#    print(i)

loader = torch.utils.data.DataLoader( dataset, batch_size=batch_size, collate_fn=LMDataset.collate_fn )

niter = 1

# Get entry data
#for iientry in range(niter):
#    batch = next(iter(loader))
#nvoxels = batch[0]["coord"].shape[0]
# We need to retrieved the 3d positions
#pos3d = batch[0]["coord"].astype(np.float64)*0.3
#pos3d[:,1] -= 117.0
#print(pos3d.shape)

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server

def make_figures(ientry,loader,minprob=0.0):

#    print("making figures for ientry={} plot-by={}".format(ientry,plotby))
    print("making figures for ientry={}".format(ientry))
    global larvoxeltrainingdata

    batch = next(iter(loader))

    print("voxel entries: ",batch["voxcoord"].shape)

    traces_v = []

    color = batch["voxfeat"][:,0]
    # 3D trace
    voxtrace = {
        "type":"scatter3d",
        "x":batch["voxcoord"][:,1],
        "y":batch["voxcoord"][:,2],
        "z":batch["voxcoord"][:,3],
        "mode":"markers",
        "name":"voxels",
        "marker":{"color":color,
                  "size":10,
                  "opacity":1}}
    traces_v.append(voxtrace)

    voxtrace["marker"]["colorscale"]="Viridis"

    return traces_v

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

    cluster_traces_v = make_figures(ientry,loader,minprob=0.0)
    #cluster_traces_v = make_figures(int(vals[1]),loader,minprob=0.0)
#    cluster_traces_v = make_figures(int(vals[1]),plotby=vals[2],minprob=0.0)
    vals[-1]["data"] = cluster_traces_v
    return vals[-1],"event requested: {} {}".format(vals[1],vals[2])

if __name__ == "__main__":
    app.run_server(debug=True)
