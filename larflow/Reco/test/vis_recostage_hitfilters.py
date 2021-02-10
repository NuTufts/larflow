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

filtered_hit_producers = ["ssnetsplit_full_showerhit",
                          "ssnetsplit_full_trackhit",
                          "taggerrejecthit",
                          "full_maxtrackhit",
                          "maxshowerhit",
                          "maxtrackhit_wcfilter",
                          "projsplitnoise",
                          "hip",
                          "showerkp",
                          "ssnetsplit_full_showerhit",
                          "ssnetsplit_wcfilter_showerhit",
                          "ssnetsplit_wcfilter_trackhit",
                          "taggerfilterhit"]

color_by_options = ["larmatch","keypoint"]
colorscale = "RdBu"
option_dict = []
for opt in color_by_options:
    option_dict.append( {"label":opt,"value":opt} )
hit_option_dict = []
for opt in filtered_hit_producers:
    hit_option_dict.append( {"label":opt,"value":opt} )
    

# OPEN LARLITE FILE
io = larlite.storage_manager( larlite.storage_manager.kREAD )
io.add_in_filename( args.input_larlite )
io.set_verbosity(1)
if args.input_mcinfo is not None and args.input_mcinfo!=args.input_larlite:
    print("adding in mc info")
    io.add_in_filename( args.input_mcinfo )
    HAS_MC = True
else:
    HAS_MC = False
for hitproducer in filtered_hit_producers:
    io.set_data_to_read( larlite.data.kLArFlow3DHit, hitproducer )
io.set_data_to_read( larlite.data.kMCTrack,  "mcreco" )
io.set_data_to_read( larlite.data.kMCShower, "mcreco" )
io.set_data_to_read( larlite.data.kMCTruth,  "generator" )
io.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashBeam" )
io.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashCosmic" )
io.open()

# OPEN VERTEX RECO FILE
anafile = rt.TFile( args.input_kpsana )
kpsanatree = anafile.Get("KPSRecoManagerTree")
nentries = io.get_entries()
CURRENT_EVENT = None

print("NENTRIES: ",nentries)

def make_figures(entry,hitproducer,plotby="larmatch",minprob=0.0):
    from larcv import larcv
    larcv.load_pyutil()
    detdata = lardly.DetectorOutline()
    
    from larflow import larflow
    larcv.SetPyUtil()    
    print("making figures for entry={} producer={} plot-by={}".format(entry,hitproducer,plotby))
    global io
    global kpsanatree
    io.go_to(entry)
    nbytes = kpsanatree.GetEntry(entry)
    if nbytes==0:
        return []
    
    traces_v = []

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


    print("hitproducer: ",hitproducer)
    sys.stdout.flush()
    ev_hit = io.get_data(larlite.data.kLArFlow3DHit, str(hitproducer) )
    print("ev_hit: ",ev_hit)
    sys.stdout.flush()    
    print("number of hits in '%s': %d"%(hitproducer,ev_hit.size()))
    sys.stdout.flush()    
    hit_trace = lardly.data.visualize_larlite_larflowhits( ev_hit )
    hit_trace["marker"]["colorscale"] = "RdBu"
    traces_v.append(hit_trace)
    
    if HAS_MC:
        mctrack_v = lardly.data.visualize_larlite_event_mctrack( io.get_data(larlite.data.kMCTrack, "mcreco"), origin=1)
        traces_v += mctrack_v

        mcshower_v = lardly.data.visualize_larlite_event_mcshower( io.get_data(larlite.data.kMCShower, "mcreco"), return_dirplot=True )
        traces_v += mcshower_v
        
    
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

# INPUT FORM: FILTERED HIT CONTAINERS
plottrack = dcc.Dropdown(
    options=hit_option_dict,
    value="maxshowerhit",
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

    hitproducer = vals[2]
    entry    = int(vals[1])
    cluster_traces_v = make_figures(int(vals[1]),hitproducer)
    
    # update the figure's traces
    vals[-1]["data"] = cluster_traces_v
    return vals[-1],"event requested: {}; hit producer: {}; plot-option: {}".format(vals[1],vals[2],vals[3])

if __name__ == "__main__":
    app.run_server(debug=True)
