from __future__ import print_function
import os,sys,argparse,json
from ctypes import c_int
from math import log

parser = argparse.ArgumentParser("Keypoint and Triplet truth")
parser.add_argument("input_file",type=str,help="file produced by 'run_voxelizetriplets.py'")
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


color_by_options = ["larmatch","ssn-bg","ssn-track","ssn-shower","ssn-class","keypoint-nu","keypoint-track","keypoint-shower","flow-field"]
colorscale = "Viridis"
option_dict = []
for opt in color_by_options:
    option_dict.append( {"label":opt,"value":opt} )

# LOAD TREES
infile = rt.TFile(args.input_file)
io = infile.Get("voxelizer")
nentries = io.GetEntries()
print("NENTRIES: ",nentries)
    
from larlite import larutil
dv = larutil.LArProperties.GetME().DriftVelocity()

def make_figures(entry,plotby="larmatch",minprob=0.0):

    print("making figures for entry={} plot-by={}".format(entry,plotby))
    global io

    nbytes = io.GetEntry(entry)
    
    detdata = lardly.DetectorOutline()

    traces_v = detdata.getlines()

    data = io.data_v[0].make_voxeldata_dict()

    origin_x = io.data_v[0].get_origin()[0]
    origin_y = io.data_v[0].get_origin()[1]
    origin_z = io.data_v[0].get_origin()[2]

    data["voxcoord"] = data["voxcoord"].astype(np.float)
    data["voxcoord"][:,0] += origin_x/0.3
    data["voxcoord"][:,1] += origin_y/0.3
    data["voxcoord"][:,2] += origin_z/0.3
    data["voxcoord"] *= 0.3
    
    # 3D trace
    voxtrace = {
        "type":"scatter3d",
        "x":data["voxcoord"][:,0],
        "y":data["voxcoord"][:,1],
        "z":data["voxcoord"][:,2],
        "mode":"markers",
        "name":"voxels",
        "marker":{"color":data["voxlabel"],
                  "size":1,
                  "opacity":0.5,
                  "colorscale":"Viridis"}}
    
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
    value='truthmatch',
    id='plotbyopt',
    )
        

app.layout = html.Div( [
    html.Div( [ eventinput,
                minprob_input,
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
     State("min_prob","value"),
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
    if vals[3] is None:
        print("Plot-by option is None")
        raise PreventUpdate
    try:
        minprob = float(vals[2])
    except:
        print("min prob cannot be turned into float")
        raise PreventUpdate
    if minprob<0:
        minprob = 0.0
    if minprob>1.0:
        minprob = 1.0
    

    cluster_traces_v = make_figures(int(vals[1]),plotby=vals[3],minprob=minprob)
    #print(cluster_traces_v)
    vals[-1]["data"] = cluster_traces_v
    return vals[-1],"event requested: {} {} {}".format(vals[1],vals[2],vals[3])

if __name__ == "__main__":
    app.run_server(debug=True)
