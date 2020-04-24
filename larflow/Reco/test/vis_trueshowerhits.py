from __future__ import print_function
import os,sys,argparse,json

parser = argparse.ArgumentParser("Plot Keypoint output")
parser.add_argument("input_hits",type=str,help="larflow input")
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

color_by_options = ["larmatch"]
colorscale = "Viridis"
option_dict = []
for opt in color_by_options:
    option_dict.append( {"label":opt,"value":opt} )

io = larlite.storage_manager( larlite.storage_manager.kREAD )
io.add_in_filename( args.input_hits )
io.open()

nentries = io.get_entries()
print("NENTRIES: ",nentries)

def make_figures(entry,plotby="larmatch",treename="trueshowerhits",minprob=-1.0):
    from larcv import larcv
    larcv.load_pyutil()
    detdata = lardly.DetectorOutline()
    
    from larflow import larflow
    larcv.SetPyUtil()    
    print("making figures for entry={} plot-by={}".format(entry,plotby))
    global io
    io.go_to(entry)
    ev_lfhits = io.get_data( larlite.data.kLArFlow3DHit, treename )
    npoints = ev_lfhits.size()
    
    traces_v = []        

    if plotby=="larmatch":
        lfhit_v = [ lardly.data.visualize_larlite_larflowhits( ev_lfhits, "trueshowerhits", score_threshold=minprob) ]
        traces_v += lfhit_v
    elif plotby in ["keypoint"]:
        hitindex=13
        xyz = np.zeros( (npoints,4 ) )
        ptsused = 0
        for ipt in xrange(npoints):
            hit = ev_lfhits.at(ipt)

            if hit.track_score<minprob:
                continue

            xyz[ptsused,0] = hit[0]
            xyz[ptsused,1] = hit[1]
            xyz[ptsused,2] = hit[2]
            if plotby=="ssn-class":
                idx = np.argmax( np.array( (hit[10],hit[11],hit[12]) ) )
                xyz[ptsused,3] = float(idx)/2.0
            else:
                xyz[ptsused,3] = hit[hitindex]
            #print(xyz[ptsused,3])
            ptsused += 1

        print("make hit data[",plotby,"] npts=",npoints," abovethreshold(plotted)=",ptsused)
        larflowhits = {
            "type":"scatter3d",
            "x": xyz[:ptsused,0],
            "y": xyz[:ptsused,1],
            "z": xyz[:ptsused,2],
            "mode":"markers",
            "name":plotby,
            "marker":{"color":xyz[:ptsused,3],"size":1,"opacity":0.8,"colorscale":'Viridis'},
        }
        traces_v.append( larflowhits )
        

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
