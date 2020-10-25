from __future__ import print_function
import os,sys,argparse,json

parser = argparse.ArgumentParser("Plot Keypoint output")
parser.add_argument("-ll","--input-larlite",required=True,type=str,help="kpsrecomanager larlite output file")
#parser.add_argument("-ana","--input-kpsana",required=True,type=str,help="kpsrecomanager ana output file")
parser.add_argument("-mc","--input-mcinfo",type=str,default=None,help="dl merged or larlite mcinfo with truth info")
#parser.add_argument("--draw-crtdata",type=str,default=None,help="dl merged or larlite with crt info")
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
#anafile = rt.TFile( args.input_kpsana )
#kpsanatree = anafile.Get("KPSRecoManagerTree")
#nentries = kpsanatree.GetEntries()
nentries = io.get_entries()
CURRENT_EVENT = None

print("NENTRIES: ",nentries)

def make_figures(entry,vtxid,plotby="larmatch",treename="larmatch",minprob=0.0):
    from larcv import larcv
    larcv.load_pyutil()
    detdata = lardly.DetectorOutline()
    crtdata = lardly.CRTOutline().getlines()    
    
    from larflow import larflow
    larcv.SetPyUtil()    
    print("making figures for entry={} plot-by={}".format(entry,plotby))
    global io
    #global kpsanatree
    io.go_to(entry)
    #nbytes = kpsanatree.GetEntry(entry)
    #if nbytes==0:
    #    return []
    
    traces_v = []

    # GET THE VERTEX DATA
    plotall = True
    if vtxid == "all" or vtxid=="notloaded":
        plotall = True
    else:
        plotall = False

    # CRT HITS
    if True:
        evopflash_beam   = io.get_data(larlite.data.kOpFlash,"simpleFlashBeam")
        evopflash_cosmic = io.get_data(larlite.data.kOpFlash,"simpleFlashCosmic")
        
        evhits = io.get_data(larlite.data.kCRTHit,"crthitcorr")
        crthit_v = [ lardly.data.visualize_larlite_event_crthit( evhits, "crthitcorr", notimeshift=False) ]
        filtered_crthit_v = lardly.ubdl.filter_crthits_wopreco( evopflash_beam, evopflash_cosmic, evhits, verbose=True )
        vis_filtered_crthit_v = [ lardly.data.visualize_larlite_crthit( x, notimeshift=False ) for x in filtered_crthit_v ]
        traces_v += vis_filtered_crthit_v

        # CRT TRACKS
        #evtracks   = io.get_data(larlite.data.kCRTTrack,"crttrack")
        #crttrack_v = lardly.data.visualize_larlite_event_crttrack( evtracks, "crttrack", notimeshift=True)
        #traces_v += crttrack_v
        

    if False:
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

    #  PLOT TRACK PCA-CLUSTERS: FULL/COSMIC
    if plotall:
        clusters = [("cosmic","cosmictrackclusters","rgb(10,10,150)",0.1,True)]
        for (name,producer,rgbcolor,opa,plotme) in clusters:
            if not plotme:
                continue
        
            ev_trackcluster = io.get_data(larlite.data.kLArFlowCluster, producer )
            ev_pcacluster   = io.get_data(larlite.data.kPCAxis,         producer )
            print("plot pca clusters: ",producer,ev_trackcluster)
            for icluster in range(ev_trackcluster.size()):
                lfcluster = ev_trackcluster.at( icluster )
                cluster_trace = lardly.data.visualize_larlite_larflowhits( lfcluster, name="%s[%d]"%(name,icluster) )
                cluster_trace["marker"]["color"] = rgbcolor
                cluster_trace["marker"]["opacity"] = opa
                cluster_trace["marker"]["width"] = 1.0
                traces_v.append(cluster_trace)            
                
                pcaxis = ev_pcacluster.at( icluster )
                pcatrace = lardly.data.visualize_pcaxis( pcaxis )
                pcatrace["name"] = "%s-pca[%d]"%(name,icluster)
                pcatrace["line"]["color"] = "rgb(0,0,0)"
                pcatrace["line"]["width"] = 1
                pcatrace["line"]["opacity"] = 1.0            
                traces_v.append( pcatrace )

    # TRACK RECO
    tracks = [("CMT","cosmictrack","rgb(0,100,50)",False,True),
              ("SIM","simplecosmictrack","rgb(0,100,50)",False,True),
              ("CON","containedcosmic","rgb(100,0,50)",False,False)]
    
    if plotall:
        for name,track_producer,zrgb,plotme,plotcluster in tracks:
            if not plotme:
                continue
            ev_track = io.get_data(larlite.data.kTrack,track_producer)
            for itrack in xrange(ev_track.size()):
                trktrace = lardly.data.visualize_larlite_track( ev_track[itrack] )
                trktrace["name"] = "LN-%s[%d]"%(name,itrack)
                trktrace["line"]["color"] = zrgb
                trktrace["line"]["width"] = 5
                trktrace["line"]["opacity"] = 1.0
                traces_v.append( trktrace )
            if not plotcluster:
                continue
            ev_trackcluster = io.get_data( larlite.data.kLArFlowCluster,track_producer)
            for itrack in xrange(ev_trackcluster.size()):
                trktrace = lardly.data.visualize_larlite_larflowhits( ev_trackcluster[itrack] )
                trktrace["name"] = "PT-%s[%d]"%(name,itrack)
                trktrace["marker"]["color"] = zrgb
                trktrace["marker"]["width"] = 5
                trktrace["marker"]["opacity"] = 1.0
                traces_v.append( trktrace )
        

    vtxinfo = []
    for treename in ["matchcrthit"]:
        ev_crttrack = io.get_data(larlite.data.kCRTTrack, treename )
        ev_opflash  = io.get_data(larlite.data.kOpFlash,  treename )
        ev_track    = io.get_data(larlite.data.kTrack,    treename )
        ev_trackcluster = io.get_data(larlite.data.kLArFlowCluster, treename)

        ntracks = ev_crttrack.size()

        print(" tree set: ",treename)
        print("   ncrttrack=",ev_crttrack.size(),")")
        print("   nopflash=",ev_opflash.size(),")")
        print("   ntrack=",ev_track.size(),")")
        print("   ncluster=",ev_trackcluster.size(),")")

        for n in xrange(ntracks):
            matchidname = "%s:%d"%(treename,n)
            vtxinfo.append( {"label":matchidname,"value":matchidname} )
            
            vis_crttrack = [lardly.data.visualize_larlite_crttrack(ev_crttrack.at(n),notimeshift=True)]
            vis_opflash  =  lardly.data.visualize_larlite_opflash_3d( ev_opflash.at(n) )
            vis_track  = [ lardly.data.visualize_larlite_track( ev_track.at(n) ) ]
            vis_cluster = [ lardly.data.visualize_larlite_larflowhits( ev_trackcluster.at(n) ) ]

            if not plotall and vtxid!=matchidname:
                continue
            
            for vis_lf in vis_track:
                vis_lf["line"]["color"]="rgb(0,0,255)"
            vis_track[0]["name"] = "%s[%d]"%(treename,n)
            vis_cluster[0]["marker"]["color"]="rgb(255,0,0)"
            vis_cluster[0]["name"]="%s[%d]"%(treename,n)
            traces_v += vis_crttrack
            traces_v += vis_opflash
            traces_v += vis_track
            traces_v += vis_cluster
            

    if HAS_MC:
        mctrack_v = lardly.data.visualize_larlite_event_mctrack( io.get_data(larlite.data.kMCTrack, "mcreco"), origin=1)
        traces_v += mctrack_v

        mcshower_v = lardly.data.visualize_larlite_event_mcshower( io.get_data(larlite.data.kMCShower, "mcreco"), return_dirplot=True )
        traces_v += mcshower_v
        
    
    # add detector outline
    traces_v += detdata.getlines(color=(10,10,10))
    traces_v += crtdata

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
        "aspectratio": {"x": 1, "y": 1, "z": 1},
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
