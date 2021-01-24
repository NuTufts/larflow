from __future__ import print_function
import os,sys,argparse,json

parser = argparse.ArgumentParser("Plot Keypoint output")
parser.add_argument("-ll","--input-larlite",required=True,type=str,help="mask-rcnn larlite output file")
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

color_by_options = ["larmatch","keypoint"]
colorscale = "Viridis"
option_dict = []
for opt in color_by_options:
    option_dict.append( {"label":opt,"value":opt} )

# OPEN LARLITE FILE
io = larlite.storage_manager( larlite.storage_manager.kREAD )
io.add_in_filename( args.input_larlite )
io.open()

nentries = io.get_entries()
CURRENT_EVENT = None

print("NENTRIES: ",nentries)

def make_figures(entry,vtxid,plotby="larmatch",treename="larmatch",minprob=0.0):
    from larcv import larcv
    larcv.load_pyutil()
    detdata = lardly.DetectorOutline()
    
    from larflow import larflow
    larcv.SetPyUtil()    
    print("making figures for entry={} plot-by={}".format(entry,plotby))
    global io
    io.go_to(entry)

    traces_v = []


    #  PLOT TRACK PCA-CLUSTERS: FULL/COSMIC
    clusters = [("unused","notmrcnn","rgb(10,10,10,0.1)",0.1,True),
                ("mrcnn","mrcnn","rgb(10,10,150)",0.3,True)]
    
    for (name,producer,rgbcolor,opa,plotme) in clusters:
        if not plotme:
            continue
        
        ev_trackcluster = io.get_data(larlite.data.kLArFlowCluster, producer )
        for icluster in range(ev_trackcluster.size()):
            lfcluster = ev_trackcluster.at( icluster )
            cluster_trace = lardly.data.visualize_larlite_larflowhits( lfcluster, name="%s[%d]"%(name,icluster) )
            
            if name in ["unused"]:
                # fixed color
                zcol = rgbcolor
            else:
                # random color
                randcol = np.random.randint(0,high=255,size=3)
                zcol = "rgb(%d,%d,%d)"%(randcol[0],randcol[1],randcol[2])
            
            cluster_trace["marker"]["color"] = zcol
            cluster_trace["marker"]["opacity"] = opa
            cluster_trace["marker"]["width"] = 1.0
            traces_v.append(cluster_trace)            
               
    
    # add detector outline
    traces_v += detdata.getlines(color=(10,10,10))
    
    return traces_v

def test():
    pass
    
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server

# 3D PLOT WINDOW
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

# INPUT FORM: EVENT NUM
eventinput = dcc.Input(
    id="input_event",
    type="number",
    placeholder="Input Event")

# INPUT FORM: VERTEX
plottrack = dcc.Dropdown(
    options=[
        {'label':'notloaded','value':'Event not loaded'},
    ],
    value='notloaded',
    id='plotvertexid',
)
        
# INPUT FORM: Score option (not used right now)
plotopt = dcc.Dropdown(
    options=option_dict,        
    value='larmatch',
    id='plotbyopt',
    )
        

# PAGE  LAYOUT
app.layout = html.Div( [
    html.Div( [ eventinput,
                plottrack,
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
     Output("plotvertexid","options"),
     Output("plotvertexid","value"),
     Output("out","children")],
    [Input("plot","n_clicks")],
    [State("input_event","value"),
     State("plotvertexid","value"),
     State("plotbyopt","value"),
     State("det3d","figure")],
    )
def cb_render(*vals):
    """
    runs when plot button is clicked
    """
    global EVENT_DATA
    global UNMATCHED_CLUSTERS
    global io
    global CURRENT_EVENT    
    if vals[1] is None:
        print("Input event is none")
        raise PreventUpdate
    if vals[1]>=nentries or vals[1]<0:
        print("Input event is out of range")
        raise PreventUpdate
    if vals[3] is None:
        print("Plot-by option is None")
        raise PreventUpdate

    vertexid = vals[2]
    entry    = int(vals[1])
    if entry!=CURRENT_EVENT:
        # first time we access an entry, we default to the "all" view of the vertices
        CURRENT_EVENT = entry
        vertexid = "all"
    cluster_traces_v = make_figures(int(vals[1]),vertexid,plotby=vals[2])
    vtxoptions = []
    vtxoptions.append( {'label':"all",'value':"all"} )
    
    # update the figure's traces
    vals[-1]["data"] = cluster_traces_v
    return vals[-1],vtxoptions,vertexid,"event requested: {}; vertexid: {}; plot-option: {}".format(vals[1],vals[2],vals[3])

if __name__ == "__main__":
    app.run_server(debug=True)
