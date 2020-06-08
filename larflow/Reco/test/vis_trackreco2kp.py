from __future__ import print_function
import os,sys,argparse,json

parser = argparse.ArgumentParser("Plot Keypoint output")
#parser.add_argument("-dl","--input-larflow",required=True,type=str,help="larflow input")
parser.add_argument("-tr","--input-trackreco",required=True,type=str,help="trackreco2kp larlite output file")
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

io = larlite.storage_manager( larlite.storage_manager.kREAD )
#io.add_in_filename( args.input_larflow )
io.add_in_filename( args.input_trackreco )
io.open()

nentries = io.get_entries()

print("NENTRIES: ",nentries)

def make_figures(entry,plotby="larmatch",treename="larmatch",minprob=0.3):
    from larcv import larcv
    larcv.load_pyutil()
    detdata = lardly.DetectorOutline()
    
    from larflow import larflow
    larcv.SetPyUtil()    
    print("making figures for entry={} plot-by={}".format(entry,plotby))
    global io
    io.go_to(entry)

    traces_v = []
    
    # TRACK FROM TrackReco2KP
    if False:
        ev_tracks  = io.get_data( larlite.data.kTrack, "track2kp" )
        ev_cluster = io.get_data( larlite.data.kLArFlowCluster, "track2kp" )

        for itrack in range(ev_tracks.size()):
            track = ev_tracks.at(itrack)
            color = (np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256))
            track_trace = lardly.data.visualize_larlite_track( track, track_id=itrack, color=color )
            track_points = lardly.data.visualize_larlite_larflowhits( ev_cluster.at(itrack), name="track%d"%(itrack) )
            traces_v.append(track_trace)
            traces_v.append(track_points)

        # unused points
        ev_unused = io.get_data( larlite.data.kLArFlow3DHit, "track2kpunused" )
        unused_trace = lardly.data.visualize_larlite_larflowhits( ev_unused, name="unused" )
        unused_trace["marker"]["color"] = "rgb(125,125,125)"
        unused_trace["marker"]["opacity"] = 0.1    
        traces_v.append(unused_trace)
            

    # keypoints
    if True:
        ev_kp = io.get_data( larlite.data.kLArFlow3DHit, "keypoint_bigcluster" )
        for ikp in range(ev_kp.size()):
            kptrace = {
                "type":"scatter3d",
	        "x": [ev_kp[ikp][0]],
                "y": [ev_kp[ikp][1]],
                "z": [ev_kp[ikp][2]],
                "mode":"markers",
	        "name":"KP%d"%(ikp),
                "marker":{"color":"rgb(255,0,0)","size":5,"opacity":0.5},
            }
            traces_v.append(kptrace)

        ev_kp2 = io.get_data( larlite.data.kLArFlow3DHit, "keypoint_smallcluster" )
        for ikp2 in range(ev_kp2.size()):
            kptrace = {
                "type":"scatter3d",
	        "x": [ev_kp2[ikp2][0]],
                "y": [ev_kp2[ikp2][1]],
                "z": [ev_kp2[ikp2][2]],
                "mode":"markers",
	        "name":"KP%d"%(ikp2),
                "marker":{"color":"rgb(0,0,255)","size":5,"opacity":0.5},
            }
            traces_v.append(kptrace)

    # HITS BY SSNET: SHOWER
    if False:
        ev_showerhit = io.get_data( larlite.data.kLArFlow3DHit, "maxshowerhit" )
        shower_hit_trace = lardly.data.visualize_larlite_larflowhits( ev_showerhit, name="showerhit", score_threshold=0.5 )
        shower_hit_trace["marker"]["color"] = "rgb(255,125,255)"
        shower_hit_trace["marker"]["opacity"] = 0.5
        traces_v.append( shower_hit_trace )

    # HITS BY SSNET: TRACK        
    if False:
        ev_trackhit  = io.get_data( larlite.data.kLArFlow3DHit, "maxtrackhit" )        
        track_hit_trace = lardly.data.visualize_larlite_larflowhits( ev_trackhit, name="trackhit", score_threshold=0.1 )
        track_hit_trace["marker"]["color"] = "rgb(0,0,255)"
        track_hit_trace["marker"]["opacity"] = 0.5
        traces_v.append( track_hit_trace )

    # SHOWER-KP RECO
    if True:
        ev_showerkp        = io.get_data( larlite.data.kLArFlowCluster, "showerkp" )
        ev_showerkp_pca    = io.get_data( larlite.data.kPCAxis, "showerkp" )
        ev_kpshower        = io.get_data( larlite.data.kLArFlow3DHit, "showerkp" ) 
        ev_showerkp_unused = io.get_data( larlite.data.kLArFlow3DHit, "showerkpunused" )
        ev_shower_goodhit_pca  = io.get_data( larlite.data.kPCAxis, "showergoodhit" )        
        print("Number of shower clusters: ",ev_showerkp.size())
        for ishr in range(ev_showerkp.size()):
            showercluster = ev_showerkp.at(ishr)
            showertrace = lardly.data.visualize_larlite_larflowhits( showercluster,"shower[%d]"%(ishr) )
            r3 = np.random.randint(255,size=3)
            colors = "rgb(%d,%d,%d)"%( r3[0], r3[1], r3[2] )
            showertrace["marker"]["color"] = colors
            showertrace["marker"]["opacity"] = 0.2
            traces_v.append( showertrace )        
        
        shower_pcatrace_v = lardly.data.visualize_event_pcaxis( ev_showerkp_pca )
        for tr in shower_pcatrace_v:
            tr["line"]["color"] = "rgb(255,0,0)"
            tr["line"]["width"] = 3
            tr["line"]["opacity"] = 1.0            
        traces_v += shower_pcatrace_v

        goodhit_pcatrace_v = lardly.data.visualize_event_pcaxis( ev_shower_goodhit_pca )
        for tr in goodhit_pcatrace_v:
            tr["line"]["color"] = "rgb(255,255,255)"
            tr["line"]["width"] = 3
            tr["line"]["opacity"] = 1.0            
        traces_v += goodhit_pcatrace_v
        
        for ikp in range(ev_kpshower.size()):
            kptrace = {
                "type":"scatter3d",
	        "x": [ev_kpshower[ikp][0]],
                "y": [ev_kpshower[ikp][1]],
                "z": [ev_kpshower[ikp][2]],
                "mode":"markers",
	        "name":"SHR:KP%d"%(ikp),
                "marker":{"color":"rgb(0,0,255)","size":5,"opacity":0.5},
            }
            traces_v.append( kptrace )
        
        
        unusedtrace = lardly.data.visualize_larlite_larflowhits( ev_showerkp_unused,"shrnotused" )
        unusedtrace["marker"]["color"] = "rgb(125,125,125)"
        unusedtrace["marker"]["opacity"] = 0.1
        #traces_v.append( unusedtrace )

    # TRACK/SHOWER PCA CLUSTER OUTPUT
    if True:
        # TRACK
        treename="trackprojsplit"
        evclusters = io.get_data( larlite.data.kLArFlowCluster, treename )
        evpcaxis   = io.get_data( larlite.data.kPCAxis, treename )
        nclusters = evclusters.size()
    
        print("[tree %s] num clusters=%d; num pcaxis=%d"%(treename,nclusters,evpcaxis.size()))

        for icluster in xrange(nclusters):
            
            cluster = evclusters.at(icluster)
            nhits = cluster.size()
            print("  [%d] track projection cluster, nhits=%d"%(icluster,nhits))
            clusterplot = lardly.data.visualize_larlite_larflowhits( cluster )
            clusterplot["name"] = "[%d]%s"%(icluster,treename)
            traces_v.append( clusterplot )

            # PCA-axis
            llpca = evpcaxis.at( icluster )

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
                "name":"[%d]pca-%s"%(icluster,treename),
                "line":{"color":"rgb(0,0,255)","size":2}
            }
            traces_v.append( pca_plot )

    if False:
        # SHOWER PCA
        ev_pcacluster = io.get_data( larlite.data.kLArFlowCluster, "lfshower" )
        evpcaxis   = io.get_data( larlite.data.kPCAxis, "lfshower" )
        nclusters = ev_pcacluster.size()
    
        print("[tree %s] num shower clusters=%d; num pcaxis=%d"%("lfshower",nclusters,evpcaxis.size()))

        for icluster in xrange(nclusters):
            
            cluster = ev_pcacluster.at(icluster)
            nhits = cluster.size()
            
            pts = larflow.reco.PyLArFlow.as_ndarray_larflowcluster_wssnet( cluster )
            r3 = np.random.randint(255,size=3)
            colors = "rgb(%d,%d,%d)"%( r3[0], r3[1], r3[2] )
                
            clusterplot = {
                "type":"scatter3d",
                "x":pts[:,0],
                "y":pts[:,1],
                "z":pts[:,2],
                "mode":"markers",
                "name":"%s[%d]"%(treename,icluster),
                "marker":{"color":colors,"size":1,"colorscale":colorscale}
            }
            traces_v.append( clusterplot )

            # PCA-axis
            llpca = evpcaxis.at( icluster )

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
                "name":"%s-pca[%d]"%(treename,icluster),
                "line":{"color":"rgb(255,255,255)","size":2}
            }
            traces_v.append( pca_plot )

    if False:
        ev_tracker = io.get_data(larlite.data.kTrack, "pcatracker" )
        for itrack in range(ev_tracker.size()):
            track_trace = lardly.data.visualize_larlite_track( ev_tracker.at(itrack), itrack, color="rgb(0,0,0)" )
            traces_v.append(track_trace)
            

    # add detector outline
    traces_v += detdata.getlines()
    
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
    value='larmatch',
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
