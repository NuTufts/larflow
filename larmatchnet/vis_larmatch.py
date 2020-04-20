#!/usr/bin/env python
from __future__ import print_function
import os,sys,argparse

"""
Visualize LArMatch output (coming from the deploy_larmatch.py script)

Requires lardly be setup:
  (1) install plot.ly and dash (see lardly readme)
  (2) clone lardly repository somewhere (only do once of course)
  (3) add the repository folder to your python path: export PYTHONPATH=/your/path/to/lardly/repo/folder:$PYTHONPATH
Note that step (3) has to be done each time you start a new shell.

"""

parser = argparse.ArgumentParser("test_3d lardly viewer")
parser.add_argument("-ll","--larlite",required=True,type=str,help="larlite file with dltagger_allreco tracks")
#parser.add_argument("-e","--entry",required=True,type=int,help="Entry to load")
parser.add_argument("-p","--minprob",type=float,default=0.0,help="score threshold on hits")

args = parser.parse_args(sys.argv[1:])

import os
import json
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from larlite import larlite
from larcv import larcv
import lardly


input_larlite = args.larlite

# LARLITE
io_ll = larlite.storage_manager(larlite.storage_manager.kREAD)
io_ll.add_in_filename( input_larlite )
io_ll.open()

NENTRIES = io_ll.get_entries()
CURRENT_ENTRY = None
detdata = lardly.DetectorOutline()

def make_figures(entry):

    global io_ll
    global args
    
    io_ll.go_to(entry)

    detdata = lardly.DetectorOutline()
    
    # OPFLASH
    ev_lfhits = io_ll.get_data(larlite.data.kLArFlow3DHit,"larmatch")
    print("num larflow hits: ",ev_lfhits.size())
    lfhits_v =  [ lardly.data.visualize_larlite_larflowhits( ev_lfhits, "larmatch", score_threshold=args.minprob) ]
    lfhits_v += detdata.getlines()
    
    return lfhits_v

# WIDGET FOR INPUT
eventinput = dcc.Input(
    id="input_event",
    type="number",
    placeholder="Input Event")

# APP
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# SERVER
server = app.server

# AXIS TEMPLATE
axis_template = {
    "showbackground": True,
    "backgroundcolor": "#141414",
    "gridcolor": "rgb(255, 255, 255)",
    "zerolinecolor": "rgb(255, 255, 255)",
}

# 3D PLOTTING OPTIONS
plot_layout = {
    "title": "",
    "height":800,
    "margin": {"t": 0, "b": 0, "l": 0, "r": 0},
    "font": {"size": 12, "color": "white"},
    "showlegend": False,
    "plot_bgcolor": "#141414",
    "paper_bgcolor": "#141414",
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

# LAYOUT OF VIEWER
app.layout = html.Div( [
    # Add input widget
    html.Div( [ eventinput, html.Button("Plot",id="plot") ] ),
    # Dividing line
    html.Hr(),
    # 3D graph
    html.Div( [
        dcc.Graph(
            id="det3d",
            figure={
                "data": detdata.getlines(),
                "layout": plot_layout,
            },
            config={"editable": True, "scrollZoom": False},
        )],
              className="graph__container"),
    # A place for output
    html.Div(id="out")
] )

# ACTION OF BUTTON
@app.callback(
    [Output("det3d","figure"),
     Output("out","children")],
    [Input("plot","n_clicks")],
    [State("input_event","value"),
     State("det3d","figure")],
    )
def cb_render(*vals):

    global CURRENT_ENTRY
    
    if vals[1] is None:
        print("Input event is none")
        raise PreventUpdate
    if vals[1]>=NENTRIES or vals[1]<0:
        print("Input event is out of range")
        raise PreventUpdate

    entry = int(vals[1])
    if CURRENT_ENTRY is None or CURRENT_ENTRY!=entry:
        # get the data
        lfhit_traces_v = make_figures(int(vals[1]))
        # set the data
        vals[-1]["data"] = detdata.getlines()+lfhit_traces_v
        CURRENT_ENTRY = entry
        return vals[-1],"Event[{}] plotted".format(vals[1])
    else:
        # no need to update
        raise PreventUpdate


if __name__ == "__main__":
    app.run_server(debug=True)
