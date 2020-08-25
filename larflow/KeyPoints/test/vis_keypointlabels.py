from __future__ import print_function
import os,sys,argparse,json
from ctypes import c_int
from math import log

parser = argparse.ArgumentParser("Keypoint and Triplet truth")
parser.add_argument("input_file",type=str,help="file produced by 'run_keypointdata.py'")
parser.add_argument("--no-keypoints",action='store_false',default=True,help="if flag given, turns off keypoints")
args = parser.parse_args()

import numpy as np
import ROOT as rt
from larlite import larlite
from larcv import larcv
from larflow import larflow
larcv.SetPyUtil()

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import lardly
    
color_by_options = ["truthmatch",
                    "isclosematch",
                    "dist2keypoint_nu",
                    "dist2keypoint_trk",
                    "dist2keypoint_shr",                    
                    "ssnetlabels",
                    "ssnetweights"]

colorscale = "Viridis"
option_dict = []
for opt in color_by_options:
    option_dict.append( {"label":opt,"value":opt} )

# LOAD TREES
tfile = rt.TFile(args.input_file,'open')
ev_triplet  = tfile.Get('larmatchtriplet')
ev_keypoint = tfile.Get('keypointlabels')
ev_ssnet    = tfile.Get("ssnetlabels")

nentries = ev_keypoint.GetEntries()
print("NENTRIES: ",nentries)

def make_figures(entry,plotby="truthmatch"):

    print("making figures for entry={} plot-by={}".format(entry,plotby))
    global ev_triplet
    global ev_keypoint

    ev_triplet.GetEntry(entry)
    ev_keypoint.GetEntry(entry)
    ev_ssnet.GetEntry(entry)

    # Get proposed points numpy
    triplet = ev_triplet.triplet_v.at(0)

    # number of triplets
    ntriplets = triplet._triplet_v.size()
    print("number of triplets: ",ntriplets)
    nsamples = 50000
    sig = 10.0
    index = np.arange(ntriplets)
    np.random.shuffle(index)
    pos3d = np.zeros( (nsamples,4) )

    detdata = lardly.DetectorOutline()
    
    traces_v = []

    for i in xrange(nsamples):
        for v in xrange(3):
            pos3d[i,v] = triplet._pos_v[  index[i] ][v]
    

    if plotby=="truthmatch":
        for i in xrange(nsamples):
            pos3d[i,3] = triplet._truth_v[index[i]]
    elif plotby=="isclosematch":
        for i in xrange(nsamples):
            pos3d[i,3] = ev_keypoint.kplabel[ index[i] ][0]
    elif plotby=="ssnetlabels":
        for i in xrange(nsamples):
            pos3d[i,3] = ev_ssnet.trackshower_label_v[ index[i] ]
    elif plotby=="ssnetweights":
        for i in xrange(nsamples):
            pos3d[i,3] = log(1.0+ev_ssnet.trackshower_weight_v[ index[i] ])
    elif plotby=="dist2keypoint_nu":
        for i in xrange(nsamples):
            if ev_keypoint.kplabel_nuvertex[index[i]][0]==0:
                pos3d[i,3] = 0.0
            else:
                dist = 0.0
                for v in xrange(3):
                    dist += ev_keypoint.kplabel_nuvertex[index[i]][1+v]*ev_keypoint.kplabel_nuvertex[index[i]][1+v]
                pos3d[i,3] = np.exp( -0.5*dist/(sig*sig) )
    elif plotby=="dist2keypoint_trk":
        for i in xrange(nsamples):
            if ev_keypoint.kplabel_trackends[index[i]][0]==0:
                pos3d[i,3] = 0.0
            else:
                dist = 0.0                
                for v in xrange(3):
                    dist += ev_keypoint.kplabel_trackends[index[i]][1+v]*ev_keypoint.kplabel_trackends[index[i]][1+v]
                pos3d[i,3] = np.exp( -0.5*dist/(sig*sig) )                
    elif plotby=="dist2keypoint_shr":
        for i in xrange(nsamples):
            if ev_keypoint.kplabel_showerstart[index[i]][0]==0:
                pos3d[i,3] = 0.0
            else:
                dist = 0.0            
                for v in xrange(3):
                    dist += ev_keypoint.kplabel_showerstart[index[i]][1+v]*ev_keypoint.kplabel_showerstart[index[i]][1+v]
                pos3d[i,3] = np.exp( -0.5*dist/(sig*sig) )                

                
    clusterplot = {
        "type":"scatter3d",
        "x":pos3d[:,0],
        "y":pos3d[:,1],
        "z":pos3d[:,2],
        "mode":"markers",
        "name":"larmatchtriplets",
        "marker":{"color":pos3d[:,3],"size":1,"colorscale":colorscale}
    }

    if plotby=="truthmatch":
        clusterplot["marker"]["colorscale"] = "Bluered"
    
    traces_v.append( clusterplot )

    # make scatter of keypoints
    if args.no_keypoints:
        for kptype,kpcolor in [("trackends","rgb(0,255,0)"),
                               ("showerstart","rgb(0,0,255)"),
                               ("nuvertex","rgb(255,0,0)")]:
            exec("brname=ev_keypoint.kppos_%s"%(kptype))
            print(brname)
            kppos = np.zeros( (brname.size(),3) )
            for i in xrange( kppos.shape[0] ):
                for v in xrange(3):
                    kppos[i,v] = brname[i][v]
            kpplot = {
                "type":"scatter3d",
                "x":kppos[:,0],
                "y":kppos[:,1],
                "z":kppos[:,2],
                "mode":"markers",
                "name":"keypoints",
                "marker":{"color":kpcolor,"size":5},
            }
            traces_v.append(kpplot)
            
    # add microboone TPC outline
    traces_v += detdata.getlines(color=(0,0,0))
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

plotopt = dcc.Dropdown(
    options=option_dict,
    value='truthmatch',
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

    cluster_traces_v = make_figures(int(vals[1]),plotby=vals[2])
    #print(cluster_traces_v)
    vals[-1]["data"] = cluster_traces_v
    return vals[-1],"event requested: {} {}".format(vals[1],vals[2])

if __name__ == "__main__":
    app.run_server(debug=True)
