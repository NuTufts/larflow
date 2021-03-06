from __future__ import print_function
import os,sys,argparse,json

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

detdata = lardly.DetectorOutline().getlines()
crtdata = lardly.CRTOutline().getlines()
    
color_by_options = ["ssnet","charge","prob","dead","cluster"]
colorscale = "Viridis"

treename = "pcacluster"
merged_inputfile = "merged_dlreco_extbnb_run3_821c2dfc-96b0-4725-a187-6b628cbbbea3.root"
track_inputfile = "larflow_reco_extbnb_run3.root"
crt_inputfile   = "crtmatch_reco_extbnb_run3.root"

io = larlite.storage_manager( larlite.storage_manager.kREAD )
io.add_in_filename( merged_inputfile )
io.add_in_filename( track_inputfile )
io.add_in_filename( crt_inputfile )
io.open()

nentries = io.get_entries()
print("NENTRIES: ",nentries)

def make_crtmatch_line( ioll ):

    line_traces_v = []
    
    ev_crtline = ioll.get_data( larlite.data.kPCAxis, "crtmatch" )

    for i in xrange(ev_crtline.size()):
        crtline = ev_crtline.at(i)

        pca_pts = np.zeros( (3,3) )
        for ipt in range(3):
            pca_pts[0,ipt] = crtline.getEigenVectors()[3][ipt]
            pca_pts[1,ipt] = crtline.getAvePosition()[ipt]
            pca_pts[2,ipt] = crtline.getEigenVectors()[4][ipt]
            
        pca_plot = {
            "type":"scatter3d",
            "x":pca_pts[:,0],
            "y":pca_pts[:,1],
            "z":pca_pts[:,2],
            "mode":"lines",
            "name":"pca[%d]"%(i),
            "line":{"color":"rgb(0,0,0)","size":4}
        }

        line_traces_v.append( pca_plot  )

    return line_traces_v

def make_crt_hits( io_ll ):
    evopflash_beam   = io_ll.get_data(larlite.data.kOpFlash,"simpleFlashBeam")
    evopflash_cosmic = io_ll.get_data(larlite.data.kOpFlash,"simpleFlashCosmic")
    
    print("Visualize CRT")
    ev_crthits = io_ll.get_data(larlite.data.kCRTHit,"crthitcorr")
    crthit_v = [ lardly.data.visualize_larlite_event_crthit( ev_crthits, "crthitcorr", window=[-3200,3200]) ]
    #filtered_crthit_v = lardly.ubdl.filter_crthits_wopreco( evopflash_beam, evopflash_cosmic, ev_crthits )
    #if True:
    #    vis_filtered_crthit_v = [ lardly.data.visualize_larlite_crthit( x ) for x in filtered_crthit_v ]
    #    return vis_filtered_crthit_v

    return crthit_v

def make_figures(entry,plotby="ssnet"):
    from larcv import larcv
    larcv.load_pyutil()

    from larflow import larflow
    larcv.SetPyUtil()    
    print("making figures for entry={} plot-by={}".format(entry,plotby))
    global io
    io.go_to(entry)

    evclusters = io.get_data( larlite.data.kLArFlowCluster, treename )
    evpcaxis   = io.get_data( larlite.data.kPCAxis, treename )
    nclusters = evclusters.size()

    cluster_traces_v = []

    for icluster in xrange(nclusters):

        cluster = evclusters.at(icluster)
        nhits = cluster.size()

        if plotby in ["ssnet","cluster"]:
            pts = larflow.reco.PyLArFlow.as_ndarray_larflowcluster_wssnet( cluster )
        elif plotby=="charge":
            pts = larflow.reco.PyLArFlow.as_ndarray_larflowcluster_wcharge( cluster )
        elif plotby=="prob":
            pts = larflow.reco.PyLArFlow.as_ndarray_larflowcluster_wprob( cluster )
        elif plotby=="dead":
            pts = larflow.reco.PyLArFlow.as_ndarray_larflowcluster_wdeadch( cluster )
        
        if plotby in ["ssnet","charge","prob","dead"]:
            colors = pts[:,3]
        elif plotby in ["cluster"]:
            r3 = np.random.randint(255,size=3)
            colors = "rgb(%d,%d,%d)"%( r3[0], r3[1], r3[2] )
        clusterplot = {
            "type":"scatter3d",
            "x":pts[:,0],
            "y":pts[:,1],
            "z":pts[:,2],
            "mode":"markers",
            "name":"[%d]"%(icluster),
            "marker":{"color":colors,"size":1,"colorscale":colorscale}
        }
        cluster_traces_v.append( clusterplot )

        # PCA-axis
        llpca = evpcaxis.at(icluster)

        pca_pts = np.zeros( (3,3) )
        for i in range(3):
            pca_pts[0,i] = llpca.getEigenVectors()[3][i]
            pca_pts[1,i] = llpca.getAvePosition()[i]
            pca_pts[2,i] = llpca.getEigenVectors()[4][i]
            
        pca_plot = {
            "type":"scatter3d",
            "x":pca_pts[:,0],
            "y":pca_pts[:,1],
            "z":pca_pts[:,2],
            "mode":"lines",
            "name":"pca[%d]"%(icluster),
            "line":{"color":"rgb(255,255,255)","size":4}
        }
        cluster_traces_v.append( pca_plot )

    pcalines = make_crtmatch_line( io )
    crthits  = make_crt_hits( io )

    return detdata+crtdata+cluster_traces_v+pcalines+crthits

        

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
        "aspectratio": {"x": 1, "y": 1, "z": 1.2},
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
    options=[
        {'label':'ssnet','value':'ssnet'},
        {'label':'charge','value':'charge'},
        {'label':'prob','value':'prob'},
        {'label':'cluster','value':'cluster'},
        {'label':'on dead channel','value':'dead'}],
    value='ssnet',
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
