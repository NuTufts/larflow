from __future__ import print_function
import os,sys,argparse,json

parser = argparse.ArgumentParser("Plot Keypoint output")
parser.add_argument("-ll","--input-larlite",required=True,type=str,help="kpsrecomanager larlite output file")
parser.add_argument("-ana","--input-kpsana",required=True,type=str,help="kpsrecomanager ana output file")
parser.add_argument("-mc","--input-mcinfo",type=str,default=None,help="dl merged or larlite mcinfo with truth info")
parser.add_argument("--draw-flash",action='store_true',default=False,help="If true, draw in-time flash PMT data [default: false]")
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
if args.input_mcinfo is not None:
    io.add_in_filename( args.input_mcinfo )
    HAS_MC = True
else:
    HAS_MC = False
io.open()

# OPEN VERTEX RECO FILE
anafile = rt.TFile( args.input_kpsana )
kpsanatree = anafile.Get("KPSRecoManagerTree")
nentries = kpsanatree.GetEntries()
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
    global kpsanatree
    io.go_to(entry)
    nbytes = kpsanatree.GetEntry(entry)
    if nbytes==0:
        return []
    
    traces_v = []

    # GET THE VERTEX DATA
    plotall = True
    if vtxid == "all" or vtxid=="notloaded":
        plotall = True
    else:
        vtxid = int(vtxid)
        plotall = False

    if args.draw_flash:
        ev_flash = io.get_data(larlite.data.kOpFlash,"simpleFlashBeam")
        nflashes = 0
        for iflash in range(ev_flash.size()):
            flash = ev_flash.at(iflash)
            if flash.Time()>2.94 and flash.Time()<4.86:            
                flash_trace_v = lardly.data.visualize_larlite_opflash_3d( flash )
                traces_v += flash_trace_v
                nflashes += 1
                break
        if nflashes==0:
            traces_v += lardly.data.visualize_empty_opflash()        

    vtxinfo = []
    for ivtx in range( kpsanatree.numerged_v.size() ):
        vtxinfo.append( {"label":"%d (%.2f)"%(ivtx,kpsanatree.numerged_v.at(ivtx).score ), "value":ivtx} )
        if not plotall and ivtx!=vtxid:
            # skip if asked for specific vertex info
            continue

        vertexcand = kpsanatree.numerged_v.at(ivtx)
        # Get the keypoint data
        kpvertex = io.get_data( larlite.data.kLArFlow3DHit, vertexcand.keypoint_producer ).at( vertexcand.keypoint_index )
        
        # make vertex traces
        kptrace = {
            "type":"scatter3d",
	    "x": [kpvertex[0]],
            "y": [kpvertex[1]],
            "z": [kpvertex[2]],
            "mode":"markers",
	    "name":"KP%d"%(ivtx),
            "marker":{"color":[vertexcand.score],"size":5,"opacity":0.5},
        }
        traces_v.append( kptrace )
        
        
        # we want to plot the clusters associated with this
        # if in all mode, we plot pca-axis only (else too messy)
        # we plot hits by plot-by option?
        cluster_list = []
        for icluster in range(vertexcand.cluster_v.size()):
            clustinfo = vertexcand.cluster_v.at(icluster)
            lfcluster = io.get_data( larlite.data.kLArFlowCluster, clustinfo.producer ).at( clustinfo.index )

            cluster_trace = lardly.data.visualize_larlite_larflowhits( lfcluster, name="v[%d]c[%d]"%(ivtx,icluster) )
            if clustinfo.type==larflow.reco.NuVertexCandidate.kTrack:
                cluster_trace["marker"]["color"] = "rgb(0,255,0)"
            else:
                cluster_trace["marker"]["color"] = "rgb(255,0,0)"
            cluster_trace["marker"]["opacity"] = 0.3
            traces_v.append(cluster_trace)            
            cluster_list.append(lfcluster)

            pcaxis = io.get_data( larlite.data.kPCAxis, clustinfo.producer ).at( clustinfo.index )
            pcatrace = lardly.data.visualize_pcaxis( pcaxis )
            pcatrace["name"] = "v[%d]c[%d]"%(ivtx,icluster)
            pcatrace["line"]["color"] = "rgb(0,0,0)"
            pcatrace["line"]["width"] = 3
            pcatrace["line"]["opacity"] = 1.0            
            traces_v.append( pcatrace )

    #  PLOT TRACK PCA-CLUSTERS: FULL/COSMIC
    clusters = [("cosmic","trackprojsplit_full","rgb(150,150,150)",0.15),
                ("wctrack","trackprojsplit_wcfilter","rgb(125,200,125)",0.05),
                ("wcshower","showergoodhit","rgb(200,125,125)",0.05)]
    for (name,producer,rgbcolor,opa) in clusters:
        ev_trackcluster = io.get_data(larlite.data.kLArFlowCluster, producer )
        ev_pcacluster   = io.get_data(larlite.data.kPCAxis,         producer )
        print("plot pca clusters: ",producer,ev_trackcluster)
        for icluster in range(ev_trackcluster.size()):
            lfcluster = ev_trackcluster.at( icluster )
            cluster_trace = lardly.data.visualize_larlite_larflowhits( lfcluster, name="%s[%d]"%(name,icluster) )
            cluster_trace["marker"]["color"] = rgbcolor
            cluster_trace["marker"]["opacity"] = opa
            traces_v.append(cluster_trace)            

            pcaxis = ev_pcacluster.at( icluster )
            pcatrace = lardly.data.visualize_pcaxis( pcaxis )
            pcatrace["name"] = "%s-pca[%d]"%(name,icluster)
            pcatrace["line"]["color"] = "rgb(0,0,0)"
            pcatrace["line"]["width"] = 1
            pcatrace["line"]["opacity"] = 1.0            
            traces_v.append( pcatrace )
            
    # TRACK RECO
    for name,track_producer in [("CTRK","cosmictrack"),("NUTRK","nutrack")]:
        if name in ["NUTRK"]:
            continue
        ev_track = io.get_data(larlite.data.kTrack,track_producer)
        for itrack in xrange(ev_track.size()):
            trktrace = lardly.data.visualize_larlite_track( ev_track[itrack] )
            trktrace["name"] = "%s[%d]"%(name,itrack)
            trktrace["line"]["color"] = "rgb(50,0,100)"
            trktrace["line"]["width"] = 5
            trktrace["line"]["opacity"] = 1.0
            traces_v.append( trktrace )
    

    if HAS_MC:
        mctrack_v = lardly.data.visualize_larlite_event_mctrack( io.get_data(larlite.data.kMCTrack, "mcreco"), origin=1)
        traces_v += mctrack_v

        mcshower_v = lardly.data.visualize_larlite_event_mcshower( io.get_data(larlite.data.kMCShower, "mcreco"), return_dirplot=True )
        traces_v += mcshower_v
        
    
    # add detector outline
    traces_v += detdata.getlines(color=(10,10,10))
    
    return traces_v,vtxinfo

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
    cluster_traces_v,vtxoptions = make_figures(int(vals[1]),vertexid,plotby=vals[2])
    vtxoptions.append( {'label':"all",'value':"all"} )
    
    # update the figure's traces
    vals[-1]["data"] = cluster_traces_v
    return vals[-1],vtxoptions,vertexid,"event requested: {}; vertexid: {}; plot-option: {}".format(vals[1],vals[2],vals[3])

if __name__ == "__main__":
    app.run_server(debug=True)
