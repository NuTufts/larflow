from __future__ import print_function
import os,sys,argparse,json

parser = argparse.ArgumentParser("Plot Keypoint output")
parser.add_argument("num_samples",type=int,help="Number of points to generate")
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

from larcv import larcv
larcv.load_pyutil()
detdata = lardly.DetectorOutline()

from ROOT import std
from larflow import larflow
larcv.SetPyUtil()    

scb = larflow.scb.SCBoundary()
x = np.random.random_sample( (args.num_samples,) )*400.0 - 50.0;
y = np.random.random_sample( (args.num_samples,) )*2.0*116.0 - 116.0
z = np.random.random_sample( (args.num_samples,) )*1037.0

xboundary = np.zeros( (args.num_samples,) )
for i in range(args.num_samples):
    xboundary[i] = scb.XatBoundary( x[i], y[i], z[i] )

traces_v = []

hits = {
    "type":"scatter3d",
    "x":xboundary,
    "y":y,
    "z":z,
    "mode":"markers",
    "name":"scb",
    "marker":{"color":"rgb(255,0,0)","size":3, "opacity":0.1}}
traces_v.append( hits )
    
# add detector outline
traces_v += detdata.getlines(color=(10,10,10))
        
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

        
# PAGE  LAYOUT
app.layout = html.Div( [
    html.Div( [
        dcc.Graph(
            id="det3d",
            figure={
                "data": traces_v,
                "layout": plot_layout,
            },
            config={"editable": True, "scrollZoom": False},
        )],
              className="graph__container"),
    html.Div(id="out")
] )


if __name__ == "__main__":
    app.run_server(debug=True)
