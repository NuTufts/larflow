from __future__ import print_function
import os,sys,argparse,json

parser = argparse.ArgumentParser("Plot Keypoint output")
parser.add_argument("-ana","--input-kpsana",required=True,type=str,help="kpsrecomanager ana output file")
parser.add_argument("-ll","--input-larlite",type=str,default=None,help="kpsrecomanager larlite output file")
parser.add_argument("-mc","--input-mcinfo",type=str,default=None,help="dl merged or larlite mcinfo with truth info")
parser.add_argument("--draw-flash",action='store_true',default=False,help="If true, draw in-time flash PMT data [default: false]")
parser.add_argument("--draw-perfect",action='store_true',default=False,help="If flag provided, will plot perfect reco instead")
args = parser.parse_args()

import numpy as np
import ROOT as rt
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow
#larcv.SetPyUtil()

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
if args.input_larlite is not None:
    io = larlite.storage_manager( larlite.storage_manager.kREAD )
    io.add_in_filename( args.input_larlite )
    HAS_LARLITE = True
    print("HAS_LARLITE")
else:
    HAS_LARLITE = False
    
if args.input_mcinfo is not None:
    io.add_in_filename( args.input_mcinfo )
    HAS_MC = True
else:
    HAS_MC = False

if HAS_LARLITE:
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
    global kpsanatree
    
    if HAS_LARLITE:
        global io
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

    if args.draw_flash and HAS_LARLITE:
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
    clusters = [("cosmic","trackprojsplit_full","rgb(10,10,150)",0.1,False),
                ("wctrack","trackprojsplit_wcfilter","rgb(125,200,125)",0.1,False),
                ("hip","hip","rgb(0,0,255)",0.1,False),                
                ("wcshower","showergoodhit","rgb(200,125,125)",0.1,False)]
    for (name,producer,rgbcolor,opa,plotme) in clusters:
        if not plotme or not HAS_LARLITE:
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
            pcatrace["line"]["opacity"] = opa
            traces_v.append( pcatrace )
            
    # plot vertices
    vertex_v = kpsanatree.nufitted_v
    vtxinfo = []
    for ivtx in range( kpsanatree.nufitted_v.size() ):
        nuvtx = vertex_v.at(ivtx)
        ntracks  = nuvtx.track_v.size()
        nshowers = nuvtx.shower_v.size()

        kplabel = "KP%d"%(ivtx)
        if nuvtx.keypoint_type==0:
            kplabel += "-NU"
        elif nuvtx.keypoint_type==1:
            kplabel += "-T"
        elif nuvtx.keypoint_type==2:
            kplabel += "-S"
        elif nuvtx.keypoint_type==3:
            kplabel += "-V"
        elif nuvtx.keypoint_type==4:
            kplabel += "-CMU"
            
        vtxinfo.append( {"label":"[%d] %s (%.2f) ntracks=%d nshowers=%d"%(ivtx,kplabel,vertex_v.at(ivtx).score,ntracks,nshowers), "value":ivtx} )
        if not plotall and ivtx!=vtxid:
            # skip if asked for specific vertex info
            continue

        print( "VERTEX[%d] HAS %d TRACKS and %d SHOWERS"%(ivtx,nuvtx.track_v.size(),nuvtx.shower_v.size()) )

        #vertexcand = kpsanatree.nuvetoed_v.at(ivtx)
        vertexcand_fit = kpsanatree.nufitted_v.at(ivtx)        
        # Get the keypoint data

        
        # make vertex traces
        kptrace = {
            "type":"scatter3d",
	    #"x": [vertexcand.pos[0],vertexcand_fit.pos[0]],
            #"y": [vertexcand.pos[1],vertexcand_fit.pos[1]],
            #"z": [vertexcand.pos[2],vertexcand_fit.pos[2]],
	    "x": [vertexcand_fit.pos[0]],
            "y": [vertexcand_fit.pos[1]],
            "z": [vertexcand_fit.pos[2]],
            "mode":"markers",
	    "name":kplabel,
            "marker":{"color":[0.0,1.0],"size":5,"opacity":0.9,"colorscale":"Viridis"},
        }
        traces_v.append( kptrace )
        
        
        # we want to plot the clusters associated with this
        # if in all mode, we plot pca-axis only (else too messy)
        # we plot hits by plot-by option?
        if False and HAS_LARLITE:
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

        # PLOT TRACK FOR VERTEX
        for itrack in range(nuvtx.track_v.size()):

            # track-cluster
            lfcluster = nuvtx.track_hitcluster_v[itrack]
            cluster_trace = lardly.data.visualize_larlite_larflowhits( lfcluster, name="v[%d]c[%d]"%(ivtx,itrack) )
            cluster_trace["marker"]["color"] = "rgb(0,255,0)"
            cluster_trace["marker"]["opacity"] = 0.8
            cluster_trace["marker"]["size"] = 2.0
            traces_v.append(cluster_trace)            
            
            # track-line
            track = nuvtx.track_v.at(itrack)
            trktrace = lardly.data.visualize_larlite_track( track )
            trktrace["name"] = "V[%d]-T[%d]"%(ivtx,itrack)
            trktrace["line"]["color"] = "rgb(0,255,255)"
            trktrace["line"]["width"] = 5
            trktrace["line"]["opacity"] = 1.0
            traces_v.append( trktrace )

            kemu = nuvtx.track_kemu_v.at(itrack)
            kep  = nuvtx.track_keproton_v.at(itrack)
            llmu = nuvtx.track_muid_v.at(itrack)
            llp  = nuvtx.track_protonid_v.at(itrack)
            llr  = nuvtx.track_mu_vs_proton_llratio_v.at(itrack)
            print("  TRACK[%d] KE(mu)=%.2f KE(p)=%.2f -log(L)_mu=%.2f -log(L)_p=%.2f LLratio=%.2f"%(itrack,kemu,kep,llmu,llp,llr))

        # PLOT SHOWER FOR VERTEX
        for ishower in range(nuvtx.shower_v.size()):

            shower = nuvtx.shower_v.at(ishower)
            shower_trunk = nuvtx.shower_trunk_v.at(ishower)
            shower_pca   = nuvtx.shower_pcaxis_v.at(ishower)

            #print(" trunk length: ",(shower_trunk.LocationAtPoint(1)-shower_trunk.LocationAtPoint(0)).Mag())
            #print(" trunk start: ",(shower_trunk.LocationAtPoint(0)[0],shower_trunk.LocationAtPoint(0)[1],shower_trunk.LocationAtPoint(0)[2]))
            #print(" trunk end: ",(shower_trunk.LocationAtPoint(1)[0],shower_trunk.LocationAtPoint(1)[1],shower_trunk.LocationAtPoint(1)[2]))            
            
            cluster_trace = lardly.data.visualize_larlite_larflowhits( shower, name="V[%s]-S[%d] N[%d]"%(ivtx,ishower,shower.size()) )
            trunk_trace   = lardly.data.visualize_larlite_track( shower_trunk )
            
            rgbcolor = np.random.randint(255,size=3)
            cluster_trace["marker"]["color"] = "rgb(%d,%d,0)"%(rgbcolor[0],rgbcolor[1])
            cluster_trace["marker"]["opacity"] = 0.8
            cluster_trace["marker"]["size"] = 2.0
            traces_v.append(cluster_trace)
            
            trunk_trace["line"]["color"] = "rgb(200,0,200)"
            trunk_trace["line"]["width"] = 5
            trunk_trace["line"]["opacity"] = 1.0
            trunk_trace["name"] = "TRNK[%s,%d]"%(ivtx,ishower)
            traces_v.append(trunk_trace)

            pcatrace = lardly.data.visualize_pcaxis( shower_pca )
            pcatrace["name"] = "SHR[%d]"%(ishower)
            pcatrace["line"]["color"] = "rgb(255,0,0)"
            pcatrace["line"]["width"] = 5
            pcatrace["line"]["opacity"] = 1.0
            traces_v.append( pcatrace )

            shower_mom_v = nuvtx.shower_plane_mom_vv.at(ishower)
            sh_p0 = shower_mom_v[0].E()
            sh_p1 = shower_mom_v[1].E()
            sh_p2 = shower_mom_v[2].E()            

            shower_dqdx_v = nuvtx.shower_plane_dqdx_vv.at(ishower)
            dqdx_p0 = shower_dqdx_v[0]
            dqdx_p1 = shower_dqdx_v[1]
            dqdx_p2 = shower_dqdx_v[2]
            
            print("  SHOWER[%d] KE=(%.2f,%.2f,%.2f) dq/dx=(%.2f,%.2f,%.2f)"%(ishower,sh_p0,sh_p1,sh_p2,dqdx_p0,dqdx_p1,dqdx_p2))

            
    # TRACK RECO
    for name,track_producer,zrgb,plotme in [("BTRK","boundarycosmicnoshift","rgb(50,0,100)",True),
                                            ("CTRK","containedcosmic","rgb(100,0,50)",True),
                                            ("NUTRK","nutrack_fitted","rgb(0,100,50)",False)]:
        if False and name in ["NUTRK"]:
            continue
        if not plotme:
            continue
        if not HAS_LARLITE:
            print("no larlite, skipping the plotting of ",name)
            continue

        print(name,track_producer,zrgb,plotme)
        ev_track = io.get_data(larlite.data.kTrack,track_producer)
        try:
            nreco_tracks = ev_track.size()
        except:
            continue
        
        for itrack in range(nreco_tracks):
            trktrace = lardly.data.visualize_larlite_track( ev_track[itrack] )
            trktrace["name"] = "%s[%d]"%(name,itrack)
            trktrace["line"]["color"] = zrgb
            trktrace["line"]["width"] = 5
            trktrace["line"]["opacity"] = 0.3
            traces_v.append( trktrace )
    

    if HAS_MC and HAS_LARLITE:

        #mcpg = ublarcvapp.mctools.MCPixelPGraph()
        #mcpg.buildgraphonly( io )
        #mcpg.printGraph(0,False)

        larbysmc = ublarcvapp.mctools.LArbysMC()
        larbysmc.process( io )
        larbysmc.printInteractionInfo()

        mcinfoplots = lardly.data.visualize_nu_interaction(io, do_sce_correction=True )
        traces_v += mcinfoplots
        
        #mctrack_v = lardly.data.visualize_larlite_event_mctrack( io.get_data(larlite.data.kMCTrack, "mcreco"), origin=1)
        #traces_v += mctrack_v

        #mcshower_v = lardly.data.visualize_larlite_event_mcshower( io.get_data(larlite.data.kMCShower, "mcreco"), return_dirplot=True )
        #traces_v.append( mcshower_v[2] )

    # Check for perfect reco
    num_nu_perfect = 0        
    if args.draw_perfect:
        try:
            num_nu_perfect = kpsanatree.nu_perfect_v.size()
        except:
            num_nu_perfect = 0
            print("no perfect vertex info")
            pass
        
    if args.draw_perfect and num_nu_perfect>0:
        # perfect nu vtx
        print("Perfect Vertex Plotted")
        nuperfect = kpsanatree.nu_perfect_v.at(0)
        for itrack in range(nuperfect.track_v.size()):
            per_cluster = nuperfect.track_hitcluster_v.at(itrack)
            per_track   = nuperfect.track_v.at(itrack)
            print("  true-track[%d] nhits=%d"%(itrack,per_cluster.size()))
            cluster_trace = lardly.data.visualize_larlite_larflowhits( per_cluster, name="tTRK[%d]"%(itrack) )
            cluster_trace["marker"]["color"] = 'rgb(0,0,1)'
            cluster_trace["marker"]["opacity"] = 0.2
            cluster_trace["marker"]["width"] = 1.0
            traces_v.append( cluster_trace )
            
            trktrace = lardly.data.visualize_larlite_track( per_track )
            trktrace["name"] = "tT[%d]"%(itrack)
            trktrace["line"]["color"] = "rgb(0,0,0)"
            trktrace["line"]["width"] = 1
            trktrace["line"]["opacity"] = 1.0
            traces_v.append( trktrace )
                
        for ishower in range(nuperfect.shower_v.size()):
            per_cluster = nuperfect.shower_v.at(ishower)
            per_track   = nuperfect.shower_trunk_v.at(ishower)
            print("  true-shower[%d] nhits=%d"%(ishower,per_cluster.size()))
            cluster_trace = lardly.data.visualize_larlite_larflowhits( per_cluster, name="tSHR[%d]"%(ishower) )
            cluster_trace["marker"]["color"] = 'rgb(0,0,1)'
            cluster_trace["marker"]["opacity"] = 0.2
            cluster_trace["marker"]["width"] = 1.0
            traces_v.append( cluster_trace )
            
            trktrace = lardly.data.visualize_larlite_track( per_track )
            trktrace["name"] = "tS[%d]"%(ishower)
            trktrace["line"]["color"] = "rgb(0,0,0)"
            trktrace["line"]["width"] = 1
            trktrace["line"]["opacity"] = 1.0
            traces_v.append( trktrace )
                
            
        
    
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
