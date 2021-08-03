from __future__ import print_function
import os,sys,argparse,json

parser = argparse.ArgumentParser(description='Run Prep LArFlow Match Data')
parser.add_argument("-i","--input-larmatch",required=True,type=str,help="Input larmatch file")
parser.add_argument("-ll","--input-larlite",required=False,type=str,default=False,help="Input larlite file")
parser.add_argument("-t","--truth-only",default=False,action="store_true",help="Visualize true hits only")
parser.add_argument("-s","--num-sample",default=30000,type=int,help="Max number to plot")
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

    
color_by_options = ["larmatch","instance","class"]
colorscale = "Viridis"
option_dict = []
for opt in color_by_options:
    option_dict.append( {"label":opt,"value":opt} )

detdata = lardly.DetectorOutline()
crtdet  = lardly.CRTOutline()

particle_id_color = {0:(0,0,0),      # no label
                     1:(255,125,50), # Cosmic
                     2:(0,0,0),      # BNB
                     3:(255,0,0),    # electron
                     4:(0,255,0),    # gamma
                     5:(0,125,125),  # pi0
                     6:(155,0,155),  # Muon
                     7:(255,255,0),  # Kaon
                     8:(255,165,0),  # pion                     
                     9:(0,0,255)}    # proton

particle_id_name = {0:"nolabel",  # no label
                    1:"delta",    # no label
                    2:"nolabel",  # no label
                    3:"electron", # electron
                    4:"gamma",    # gamma
                    5:"pi0",  # pi0
                    6:"muon", # Muon
                    7:"kaon", # Kaon
                    8:"pion", # proton
                    9:"proton"}   # pion

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

def make_figures(entry,plotby="larmatch"):
    from larcv import larcv
    larcv.load_pyutil()

    from larflow import larflow
    larcv.SetPyUtil()    
    print("making figures for entry={} plotby={}".format(entry,plotby))
    global tree
    global ioll
    tree.GetEntry(entry)    

    nsample = args.num_sample 
    
    cluster_traces_v = []
    # coords
    if not args.truth_only:
        npts = tree.triplet_v.front()._pos_v.size()
        index = np.arange(npts)
        np.random.shuffle(index)
              
        pos_v = np.zeros( (nsample, 3) )
        color_v = np.zeros( nsample )        
        for i in xrange( nsample ):
            for j in xrange(3):
                pos_v[i,j] = tree.triplet_v.front()._pos_v[ index[i] ][j]
            color_v[i] = tree.triplet_v.front()._truth_v[ index[i] ]
        print("number of triplet positions: ",tree.triplet_v.front()._pos_v.size()," num plotted=",nsample)
        trace = {
            "type":"scatter3d",
            "x":pos_v[:,0],
            "y":pos_v[:,1],
            "z":pos_v[:,2],
            "mode":"markers",
            "name":"larmatch",
            "marker":{"color":color_v,"size":1,"opacity":0.8,"colorscale":"Viridis"},
        }
        cluster_traces_v.append(trace)
    else:
        data = tree.triplet_v.front().make_truthonly_triplet_ndarray()
        pos_v = data["spacepoint_t"]
        print("number of truth-only triplet positions: ",pos_v.shape)
    
    
    if args.truth_only:
        print("make truth-only traces")
        if plotby=="larmatch":
            color_v = np.ones( pos_v.shape[0] )
            trace = {
                "type":"scatter3d",
                "x":pos_v[:,0],
                "y":pos_v[:,1],
                "z":pos_v[:,2],
                "mode":"markers",
                "name":"larmatch",
                "marker":{"color":color_v,"size":1,"opacity":0.8,"colorscale":"Viridis"},
            }
            cluster_traces_v.append( trace )
        elif plotby=="class":
            print("class values: ",np.unique(data["segment_t"]))
            for pid in range(0,9+1):
                idmask = data["segment_t"]==pid
                print("class_t[",pid,"] num=",idmask.sum())                
                if idmask.sum()>0:                
                    pidcoord_t = pos_v[idmask,:]
                    color = particle_id_color[pid]
                    strcolor = "rgb(%d,%d,%d)"%(color[0],color[1],color[2])
                    voxtrace = {
                        "type":"scatter3d",
                        "x":pidcoord_t[:,0],
                        "y":pidcoord_t[:,1],
                        "z":pidcoord_t[:,2],
                        "mode":"markers",
                        "name":"%s"%(particle_id_name[pid]),
                        "marker":{"color":strcolor,
                                  "size":1,
                                  "opacity":0.5}}
                    cluster_traces_v.append(voxtrace)
        elif plotby=="instance":
            instances = np.unique(data["instance_t"])
            print("instance values: ",instances)
            for nid,iid in enumerate(instances):
                idmask = data["instance_t"]==iid                
                #print(" #{} instance[{}]".format(nid,iid)," num=",idmask.sum())
                if idmask.sum()>0:
                    pidcoord_t = pos_v[idmask,:]
                    color = np.random.rand(3)*255
                    strcolor = "rgb(%d,%d,%d)"%(color[0],color[1],color[2])
                    voxtrace = {
                        "type":"scatter3d",
                        "x":pidcoord_t[:,0],
                        "y":pidcoord_t[:,1],
                        "z":pidcoord_t[:,2],
                        "mode":"markers",
                        "name":"%d"%(iid),
                        "marker":{"color":strcolor,
                                  "size":1,
                                  "opacity":0.5}}
                    cluster_traces_v.append(voxtrace)
                    

            


    # MC info to compare
    if ioll:
        print("draw mc track and shower truth")
        global ioll
        ioll.go_to(entry)

        mctrack_v = lardly.data.visualize_larlite_event_mctrack( ioll.get_data(larlite.data.kMCTrack, "mcreco"), origin=1)
        cluster_traces_v += mctrack_v

        mcshower_v = lardly.data.visualize_larlite_event_mcshower( ioll.get_data(larlite.data.kMCShower, "mcreco"), return_dirplot=True )
        #print("mcshower_v: ",len(mcshower_v))        
        for ishr in range( len(mcshower_v)/3 ):
            cluster_traces_v.append( mcshower_v[3*ishr+2] )


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
        
plotopt = dcc.Dropdown(
    options=option_dict,
    value=color_by_options[0],
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
     State("det3d","figure"),
     State("plotbyopt","value")],
    )
def cb_render(*vals):
    if vals[1] is None:
        print("Input event is none")
        raise PreventUpdate
    if vals[1]>=nentries or vals[1]<0:
        print("Input event is out of range")
        raise PreventUpdate

    cluster_traces_v = make_figures(int(vals[1]),plotby=str(vals[3]))
    #print(cluster_traces_v)
    vals[-2]["data"] = cluster_traces_v
    return vals[-2],"event requested: {}".format(vals[1])

if __name__ == "__main__":
    app.run_server(debug=True)
