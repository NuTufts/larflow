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
parser.add_argument("-e","--entry",required=True,type=int,help="Entry to load")
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
ientry        = args.entry

# LARLITE
io_ll = larlite.storage_manager(larlite.storage_manager.kREAD)
io_ll.add_in_filename( input_larlite )
io_ll.open()
io_ll.go_to(ientry)

# OPFLASH
ev_lfhits = io_ll.get_data(larlite.data.kLArFlow3DHit,"larmatch")
print("num larflow hits: ",ev_lfhits.size())
lfhits_v =  [ lardly.data.visualize_larlite_larflowhits( ev_lfhits, "larmatch", score_threshold=args.minprob) ]

detdata = lardly.DetectorOutline()

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server

axis_template = {
    "showbackground": True,
    "backgroundcolor": "#141414",
    "gridcolor": "rgb(255, 255, 255)",
    "zerolinecolor": "rgb(255, 255, 255)",
}

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


app.layout = html.Div( [
    html.Div( [
        dcc.Graph(
            id="det3d",
            figure={
                "data": detdata.getlines()+lfhits_v,
                "layout": plot_layout,
            },
            config={"editable": True, "scrollZoom": False},
        )],
              className="graph__container"),
    ] )

if __name__ == "__main__":
    app.run_server(debug=True)
