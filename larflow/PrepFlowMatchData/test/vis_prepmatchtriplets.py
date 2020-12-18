from __future__ import print_function
import os,sys,argparse,json

parser = argparse.ArgumentParser(description='Run Prep LArFlow Match Data')
parser.add_argument("-i","--input-larmatch",required=True,type=str,help="Input larmatch file")
parser.add_argument("-ll","--input-larlite",required=False,type=str,default=False,help="Input larlite file")
args = parser.parse_args(sys.argv[1:])

import numpy as np
import ROOT as rt
from larlite import larlite
from larcv import larcv
from larflow import larflow
larcv.SetPyUtil()

import lardly

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

    
color_by_options = ["ssnet","charge","prob","dead","cluster","shower","noise"]
colorscale = "Viridis"

inputfile = "out.root"
larlite_input = "mcc9_v13_bnbnue_corsika/mcinfo-Run000001-SubRun000001.root"

detdata = lardly.DetectorOutline()
crtdet  = lardly.CRTOutline()

# LARLITE
if args.input_larlite:
    ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
    ioll.add_in_filename( args.input_larlite )
    ioll.open()
else:
    ioll = None

# LARMATCH DATA
tfile = rt.TFile(args.input_larmatch,"read")
tfile.ls()
tree = tfile.Get("larmatchtriplet")
nentries = tree.GetEntries()
print("NENTRIES: ",nentries)

def make_figures(entry):
    from larcv import larcv
    larcv.load_pyutil()

    from larflow import larflow
    larcv.SetPyUtil()    
    print("making figures for entry={}".format(entry))
    global tree
    tree.GetEntry(entry)

    npts = tree.triplet_v.front()._pos_v.size()
    pos_v = np.zeros( (npts, 4) )
    for i in xrange( npts ):
        for j in xrange(3):
            pos_v[i,j] = tree.triplet_v.front()._pos_v[i][j]
        pos_v[i,3] = tree.triplet_v.front()._truth_v[i]
    print("number of triplet positions: ",tree.triplet_v.front()._pos_v.size())

            
    trace = {
        "type":"scatter3d",
        "x":pos_v[:,0],
        "y":pos_v[:,1],
        "z":pos_v[:,2],
        "mode":"markers",
        "name":"larmatch",
        "marker":{"color":pos_v[:,3],"size":1,"opacity":0.8,"colorscale":"Viridis"},
    }
    cluster_traces_v = [ trace ]


    # MC info to compare
    if ioll:
        global ioll
        ioll.go_to(entry)        
        ev_mctrack = ioll.get_data(larlite.data.kMCTrack, "mcreco" )
        print("number of mctracks: ",ev_mctrack.size())
        cluster_traces_v += lardly.data.visualize_larlite_event_mctrack( ev_mctrack )

    return detdata.getlines()+cluster_traces_v

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
        

app.layout = html.Div( [
    html.Div( [ eventinput,
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
     State("det3d","figure")],
    )
def cb_render(*vals):
    if vals[1] is None:
        print("Input event is none")
        raise PreventUpdate
    if vals[1]>=nentries or vals[1]<0:
        print("Input event is out of range")
        raise PreventUpdate

    cluster_traces_v = make_figures(int(vals[1]))
    #print(cluster_traces_v)
    vals[-1]["data"] = cluster_traces_v
    return vals[-1],"event requested: {}".format(vals[1])

if __name__ == "__main__":
    app.run_server(debug=True)
