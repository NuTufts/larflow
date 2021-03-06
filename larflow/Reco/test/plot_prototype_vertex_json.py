from __future__ import print_function
import os,sys,argparse,json

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

detdata = lardly.DetectorOutline().getlines()

# parse json

finput = open("out_prototype_vertex.json",'r')
j = json.load(finput)

jvertices = j["vertices"]

print("number of vertices: ",len(jvertices))

traces3d = []
for jvertex in jvertices:

    # get the two clusters
    colors = [ 'rgb(255,0,0)',
               'rgb(0,255,0)',
               'rgb(0,0,255)' ]
    
    for ic,cluster in enumerate(jvertex["clusters"]):
        hits = cluster["hits"]
        hit_np = np.zeros( (len(hits),3) )
        for ihit in xrange(len(hits)):
            hit_np[ihit,0] = hits[ihit][0]
            hit_np[ihit,1] = hits[ihit][1]
            hit_np[ihit,2] = hits[ihit][2]

        
        cluster_plot = {
            "type":"scatter3d",
            "x":hit_np[:,0],
            "y":hit_np[:,1],
            "z":hit_np[:,2],
            "mode":"markers",
            "name":"c",
            "marker":{"color":colors[ jvertex["cluster_types"][ic] ],"size":1,"opacity":0.3}}
        
        traces3d.append( cluster_plot )
        
        line_np = np.zeros( (3,3) )
        for ihit in xrange(3):
            for i in xrange(3):
                line_np[ihit,i] = cluster["pca"][ihit][i]
        
                
        pca_plot = {
            "type":"scatter3d",
            "x":line_np[:,0],
            "y":line_np[:,1],
            "z":line_np[:,2],
            "mode":"lines",
            "name":"pca",
            "line":{"color":"rgb(255,255,255)","size":2} }
        traces3d.append( pca_plot )
    
    vertex_plot = {
        "type":"scatter3d",
        "x":[jvertex["pos"][0]],
        "y":[jvertex["pos"][1]],
        "z":[jvertex["pos"][2]],
        "mode":"markers",
        "name":"vtx",
        "marker":{"color":"rgb(0,0,0)","size":4,"opacity":1.0} }

    print("vertex [type ",jvertex["type"],"] pos: ",jvertex["pos"])
    traces3d.append( vertex_plot )
        
        
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
    "backgroundcolor": "rgba(100, 100, 100,0.5)", #white
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

app.layout = html.Div( [
    html.Hr(),
    html.Div( [
        dcc.Graph(
            id="det3d",
            figure={
                "data": detdata+traces3d,
                "layout": plot_layout,
            },
            config={"editable": True, "scrollZoom": False},
        )],
              className="graph__container"),
    html.Div(id="out")
] )

                       

if __name__ == "__main__":
    app.run_server(debug=True)
