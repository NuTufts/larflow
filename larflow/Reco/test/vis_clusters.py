from __future__ import print_function
import os,sys,argparse,json

parser = argparse.ArgumentParser("Plot Reco Clusters for Inspection")
parser.add_argument("-ll","--input-larlite",required=True,type=str,help="kpsrecomanager larlite output file")
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

def make_figures(entry,clustername):
    """ 
    if clustername is None return all clusters. 
    else if string, return specific cluster
    """
    
    from larcv import larcv
    larcv.load_pyutil()
    detdata = lardly.DetectorOutline()
    
    from larflow import larflow
    larcv.SetPyUtil()    
    print("making figures for entry={} cluster={}".format(entry,clustername))
    global io
    global kpsanatree
    io.go_to(entry)

    traces_v = []
    cluster_list = []

    plot_producer = None
    plot_index    = None
    if clustername != "all":
        plot_producer = clustername.split(":")[0]
        plot_index    = int(clustername.split(":")[1])
    
    #  PLOT TRACK PCA-CLUSTERS: FULL/COSMIC
    clusters = [("cosmic","trackprojsplit_full","rgb(150,150,150)",0.15,False),
                ("wctrack","trackprojsplit_wcfilter","rgb(125,200,125)",1.0,True),
                ("wcshower","showergoodhit","rgb(200,125,125)",0.5,False)]
    for (name,producer,rgbcolor,opa,drawme) in clusters:

        if not drawme:
            continue
        
        ev_trackcluster = io.get_data(larlite.data.kLArFlowCluster, producer )
        ev_pcacluster   = io.get_data(larlite.data.kPCAxis,         producer )
        
        for icluster in range(ev_trackcluster.size()):

            
            lfcluster = ev_trackcluster.at( icluster )
            cluster_trace = lardly.data.visualize_larlite_larflowhits( lfcluster, name="%s[%d]"%(name,icluster) )

            clabel = "%s:%d (%d hits)"%(producer,icluster,lfcluster.size())
            cvalue = "%s:%d"%(producer,icluster)            
            cluster_list.append( {"label":clabel,"value":cvalue} )
            

            if clustername!="all":
                cluster_trace["marker"]["color"] = "rgb(50,50,50)"
            else:
                r3 = np.random.randint(255,size=3)
                rand_color = "rgb(%d,%d,%d)"%( r3[0], r3[1], r3[2] )
                cluster_trace["marker"]["color"] = rand_color
                
            cluster_trace["marker"]["opacity"] = opa
            cluster_trace["marker"]["width"] = 5.0


            pcaxis = ev_pcacluster.at( icluster )
            pcatrace = lardly.data.visualize_pcaxis( pcaxis )
            pcatrace["name"] = "%s-pca[%d]"%(name,icluster)
            pcatrace["line"]["color"] = "rgb(0,0,0)"
            pcatrace["line"]["width"] = 1
            pcatrace["line"]["opacity"] = 1.0

            if plot_producer is not None and plot_producer==producer and plot_index==icluster:
                cluster_trace["marker"]["color"] = rgbcolor                

            traces_v.append(cluster_trace)
            traces_v.append( pcatrace )    
    
    # add detector outline
    traces_v += detdata.getlines(color=(10,10,10))
    print("Number of clusters in event: ",len(cluster_list))
    
    return traces_v,cluster_list

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

# INPUT FORM: CLUSTER LIST
plotcluster = dcc.Dropdown(
    options=[
        {'label':'all','value':'all'},
    ],
    value='all',
    id='plotcluster',
)
        
# PAGE  LAYOUT
app.layout = html.Div( [
    html.Div( [ eventinput,
                plotcluster,
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
     Output("plotcluster","options"),
     Output("plotcluster","value"),
     Output("out","children")],
    [Input("plot","n_clicks")],
    [State("input_event","value"),
     State("plotcluster","value"),
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

    clustername = vals[2]
    entry       = int(vals[1])
    if entry!=CURRENT_EVENT:
        # first time we access an entry, we default to the "all" view of the vertices
        CURRENT_EVENT = entry
        clustername = "all"
    cluster_traces_v,cluster_options = make_figures(int(vals[1]),clustername)
    cluster_options.append( {'label':"all",'value':"all"} )
    
    # update the figure's traces
    vals[-1]["data"] = cluster_traces_v
    return vals[-1],cluster_options,clustername,"event requested: {}; cluster: {}".format(vals[1],vals[2])

if __name__ == "__main__":
    app.run_server(debug=True)
