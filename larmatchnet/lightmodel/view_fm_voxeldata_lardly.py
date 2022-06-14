from __future__ import print_function
import os,sys,argparse

parser = argparse.ArgumentParser("flashmatched voxel data lardly viewer")
#parser.add_argument("-ll","--larlite",required=True,type=str,help="larlite file with dltagger_allreco tracks")
parser.add_argument("-ll","--larlite",type=str,default="",help="larlite filtered mc file containing mctrack/shower and opflash info") # optional mctrack
#parser.add_argument("-fl","--flash",type=str,default="",help="larlite filtered mc file containing mctrack/shower and opflash info") # optional mctrack
#parser.add_argument("-e","--entry",required=True,type=int,help="Entry to load")

args = parser.parse_args(sys.argv[1:])

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

input_larlite = args.larlite
io_ll = larlite.storage_manager(larlite.storage_manager.kREAD)
io_ll.add_in_filename( input_larlite )
io_ll.open()


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
dataset = LMDataset( filelist=["test_COMBINED.root"], is_voxeldata=True, random_access=False )
nentries = len(dataset)

print("NENTRIES: ",nentries)

#ientry = 0

#io_ll.go_to(ientry)
#ev_mctrack = io_ll.get_data(larlite.data.kMCTrack, "mcreco")
#mctrack = ev_mctrack.at(0)

#traces3d = []

# MCTRACK
#if args.larlite!="":
#    print("VISUALIZE MCTRACKS")
#    mctrack_v = lardly.data.visualize_larlite_mctrack( mctrack )
#    print("mcytrack_v:",mctrack_v)
#    traces3d.append( mctrack_v )

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


def make_figures(ientry,loader,minprob=0.0):

#    print("making figures for ientry={} plot-by={}".format(ientry,plotby))
    print("making figures for ientry={}".format(ientry))
    global larvoxeltrainingdata


    #batch = next(iter(loader))
    batch = list(iter(loader))[ientry]

    # MCTRACK
    if args.larlite!="":

        global io_ll

        io_ll.go_to(ientry)

        #ev_mctrack = io_ll.get_data(larlite.data.kMCTrack, "mcreco")
        ev_mcshower = io_ll.get_data(larlite.data.kMCShower, "mcreco")
        #ev_opflash_cosmic = io_ll.get_data(larlite.data.kOpFlash, "simpleFlashCosmic")
        ev_opflash_beam = io_ll.get_data(larlite.data.kOpFlash, "simpleFlashBeam")

        #mctrack = ev_mctrack.at(0)
        mcshower = ev_mcshower.at(0)

        print("shower INFO X: ", mcshower.Start().X())
        #print("mcshower.at(0).X() ", mcshower.at(0).X())
        #print("First mcstep Y:", mctrack.at(0).Y() )
        #print("First mcstep Z:", mctrack.at(0).Z() )

        print("Drift velocity is: ",dv)

        traces3d = []
        flash = { "flash":[] }

        traces_v = []

        print("VISUALIZE MCTRACKS")
        #mctrack_v = lardly.data.visualize_larlite_mctrack( mctrack, do_sce_correction=False )
        #mcshower_v = lardly.data.visualize_larlite_event_mcshower( ev_mcshower )
        mcshower_lardly = lardly.data.visualize3d_larlite_mcshower( mcshower ) # returns an array of objects
        for x in mcshower_lardly:
            if x is not None:
                traces_v.append(x)
        #opflash_v = lardly.data.visualize_larlite_opflash_3d( ev_opflash_cosmic.at(0) )
        opflash_v = lardly.data.visualize_larlite_opflash_3d( ev_opflash_beam.at(0) )
        #print("mcytrack_v:",mctrack_v)
        #print("opflash_v:",opflash_v)
        #traces3d.append( mctrack_v )
        #traces3d.append( mcshower_v )

    print("voxel entries: ",batch["voxcoord"].shape)

    traces_v = []

    color = batch["voxfeat"][:,0]

    # 3D trace
    voxtrace = {
        "type":"scatter3d",
        "x":batch["voxcoord"][:,1]*0.3+(2399)*0.5*dv, #subtract 3200 from 2399 for shift
        "y":batch["voxcoord"][:,2]*0.3-120.0,
        "z":batch["voxcoord"][:,3]*0.3,
        "mode":"markers",
        "name":"voxels",
        "marker":{"color":color,
                  "size":10,
                  "opacity":1}}
    traces_v.append(voxtrace)

    if args.larlite!="":
        #traces_v.append(mctrack_v)
        #traces_v += mcshower_v
        traces_v += opflash_v
    #traces_v.append( detdata.getlines() )
    traces_v += detdata.getlines()

    voxtrace["marker"]["colorscale"]="Viridis"

    return traces_v

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
                #"data": detdata.getlines()+traces3d,
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

    #cluster_traces_v = make_figures(ientry,loader,minprob=0.0)
    cluster_traces_v = make_figures(int(vals[1]),loader,minprob=0.0)
#    cluster_traces_v = make_figures(int(vals[1]),plotby=vals[2],minprob=0.0)
    vals[-1]["data"] = cluster_traces_v
    return vals[-1],"event requested: {} {}".format(vals[1],vals[2])

if __name__ == "__main__":
    app.run_server(debug=True)
