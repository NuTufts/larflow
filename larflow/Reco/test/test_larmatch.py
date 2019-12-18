from __future__ import print_function
import os,sys,argparse


import os
import json
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import numpy as np


#contour_types = ["track","split"]
#contour_types = ["track","shower"]
#contour_types = ["split","merged"]
contour_types = ["all","shower"]

cluster_dict = {}
pca_dict = {}
for top in contour_types:
    
    print("parse data for ",top)
    with open('dump_%s.json'%(top),'r') as f:
        data = json.load(f)

    cluster_v = []
    pca_v     = []

    for cidx,cluster in enumerate(data["clusters"][:-1]):
        pts = np.zeros( (len(cluster["hits"]),3) )
        for idx,hit in enumerate(cluster["hits"]):
            for i in range(3):
                pts[idx,i] = hit[i]
        color = "rgb(%d,%d,%d)"%(np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256))
        clusterplot = {
            "type":"scatter3d",
            "x":pts[:,0],
            "y":pts[:,1],
            "z":pts[:,2],
            "mode":"markers",
            "name":"%s[%d]"%(top,cidx),
            "marker":{"color":color,"size":1,"colorscale":'Viridis'}
        }
        cluster_v.append( clusterplot )
    
        pca_pts = np.zeros( (3,3) )
        for idx,pt in enumerate(cluster["pca"]):
            for i in range(3):
                pca_pts[idx,i] = pt[i]
        pca_plot = {
            "type":"scatter3d",
            "x":pca_pts[:,0],
            "y":pca_pts[:,1],
            "z":pca_pts[:,2],
            "mode":"lines",
            "name":"pca-%s[%d]"%(top,cidx),
            "line":{"color":"rgb(255,255,255)","size":2}
        }
        pca_v.append( pca_plot )

    cluster_dict[top] = cluster_v
    pca_dict[top] = pca_v

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
            id="det3d_%s"%(contour_types[0]),
            figure={
                "data": cluster_dict[contour_types[0]]+pca_dict[contour_types[0]],
                "layout": plot_layout,
            },
            config={"editable": True, "scrollZoom": False},
        )],
              className="graph__container"),
    html.Div( [
        dcc.Graph(
            id="det3d_%s"%(contour_types[1]),
            figure={
                "data": cluster_dict[contour_types[1]]+pca_dict[contour_types[1]],
                "layout": plot_layout,
            },
            config={"editable": True, "scrollZoom": False},
        )],
        className="graph__container"),
] )

if __name__ == "__main__":
    app.run_server(debug=True)
