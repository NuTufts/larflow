from __future__ import print_function
import os,sys,argparse,json

parser = argparse.ArgumentParser("Plot Keypoint output")
parser.add_argument("-va","--input-vtxact",required=True,type=str,help="larflow input")
parser.add_argument("-lm","--input-larmatch",default=None,required=False,type=str,help="larflow input")
parser.add_argument("-mc","--input-mcinfo",default=None,required=False,type=str,help="larflow input")
args = parser.parse_args()

import numpy as np
import ROOT as rt
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow
larcv.SetPyUtil()

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import lardly

color_by_options = ["larmatch","shower","keypoint"]
colorscale = "Viridis"
option_dict = []
for opt in color_by_options:
    option_dict.append( {"label":opt,"value":opt} )

# colors for the keypoints
keypoint_colors = { -1:"rgb(50,50,50)",
                    0:"rgb(255,0,0)",
                    1:"rgb(0,255,0)",
                    2:"rgb(0,0,255)"}

io = larlite.storage_manager( larlite.storage_manager.kREAD )
io.add_in_filename( args.input_vtxact )
if args.input_larmatch is not None and args.input_larmatch!=args.input_vtxact:
    io.add_in_filename( args.input_larmatch )
if args.input_mcinfo is not None and args.input_mcinfo not in [args.input_vtxact,args.input_larmatch]:
    io.add_in_filename( args.input_mcinfo )
io.open()

nentries = io.get_entries()

print("NENTRIES: ",nentries)

def make_figures(entry,plotby="larmatch",treename="larmatch",keypoint_tree="keypoint", minprob=0.1):
    from larcv import larcv
    larcv.load_pyutil()
    detdata = lardly.DetectorOutline()
    
    from larflow import larflow
    larcv.SetPyUtil()    
    print("making figures for entry={} plot-by={}".format(entry,plotby))
    global io
    io.go_to(entry)

    lfname = treename
    #lfname = "taggerfilterhit" # output of WC filter
    #lfname = "ssnetsplit_wcfilter_trackhit" # SSNet split
    #lfname = "maxtrackhit_wcfilter"
        
    ev_lfhits = io.get_data( larlite.data.kLArFlow3DHit, lfname )
    npoints = ev_lfhits.size()

    ev_vacand    = io.get_data( larlite.data.kLArFlow3DHit, "vacand" )
    ev_keypoints = io.get_data( larlite.data.kLArFlow3DHit, "keypoint" )    
    
    traces_v = []

    # TRUE VTX
    if args.input_mcinfo is not None:
        mcdata = ublarcvapp.mctools.LArbysMC()
        mcdata.process( io )
        mcdata.printInteractionInfo()
        truevtx = np.zeros((1,3))
        truevtx[0,0] = mcdata._vtx_detx
        truevtx[0,1] = mcdata._vtx_sce_y
        truevtx[0,2] = mcdata._vtx_sce_z
        print("true vtx:",truevtx)
        vtxtrace = {"type":"scatter3d",
                    "x":truevtx[:,0],
                    "y":truevtx[:,1],
                    "z":truevtx[:,2],
                    "mode":"markers",
                    "name":"NuVtx",
                    "marker":{"color":"rgb(0,255,255)",
                              "size":"5",
                              "opacity":0.5}
        }
        traces_v.append( vtxtrace )
    

    print("Plotting Hits: produername=",lfname)
    plot_shower_score = False
    if plotby=="shower":
        plot_shower_score = True
    lfhit_v = [ lardly.data.visualize_larlite_larflowhits( ev_lfhits, lfname, score_threshold=minprob, plot_renormed_shower_score=plot_shower_score) ]
    lfhit_v[0]["marker"]["colorscale"] = "RdBu"
    traces_v += lfhit_v

    va_trace = lardly.data.visualize_larlite_larflowhits( ev_vacand, "VA" )
    va_trace["marker"]["size"] = 3.0
    va_trace["marker"]["opacity"] = 1.0    
    va_trace["marker"]["color"] = "rgb(255,0,255)"
    traces_v.append( va_trace )

    # VA cluster pca
    for iv in range(ev_vacand.size()):
        va = ev_vacand.at(iv)
        vadir = np.zeros( (2,3) )
        for i in range(3):
            vadir[0,i] = va[i]
            vadir[1,i] = va[i] + 10.0*va[19+i]
        vadir_trace = {"type":"scatter3d",
                       "x":vadir[:,0],
                       "y":vadir[:,1],
                       "z":vadir[:,2],
                       "mode":"lines",
                       "name":"VA[%d]"%(iv),
                       "line":{"color":"rgb(0,0,0)","width":3}
        }
        traces_v.append(vadir_trace)
        
    kp_trace = lardly.data.visualize_larlite_larflowhits( ev_keypoints, "KP" )
    kp_trace["marker"]["size"] = 5.0
    kp_trace["marker"]["opacity"] = 0.5    
    kp_trace["marker"]["color"] = "rgb(0,255,0)"
    #traces_v.append( kp_trace )

        
               
    
    # end of loop over treenames
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
