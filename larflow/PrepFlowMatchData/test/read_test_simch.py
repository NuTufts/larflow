import os,sys
import h5py
import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import lardly
from lardly.detectoroutline import DetectorOutline

fh5 = h5py.File('out_test.h5', 'r')

opacity=0.5
marker_size=2.0
#colorby = 'edep'
colorby = 'trackid'
#pos_var='true'
pos_var='reco'

triplet_data = fh5['triplet_data']


columns = ['pos_x','pos_y','pos_z',
    'pos_x_reco','pos_y_reco','pos_z_reco',
    'edep',
    'trackid',
    'pid',
    'aid',
    'origin',
    'uwire',
    'vwire',
    'ywire',
    'tick',
    'row']

data = {}

for col in columns:
    npts = len(triplet_data[col])
    data[col] = np.array( triplet_data[col], dtype=np.float32 ).reshape((npts,1))
    print(col,": ",data[col].shape)

detdata = DetectorOutline()


customdata = np.concatenate( [data['edep'],data['pid'],data['trackid'],data['aid']],axis=1 )

hovertemplate = """
<b>x</b>: %{x:.1f}<br>
<b>y</b>: %{y:.1f}<br>
<b>z</b>: %{z:.1f}<br>
<b>edep</b>: %{customdata[0]:.3f} MeV<br>
<b>PID</b>:  %{customdata[1]:d}<br>
<b>TID</b>:  %{customdata[2]:d}<br>
<b>AID</b>:  %{customdata[3]:d}<br>
"""

if pos_var=='reco':
    pos_var_x = 'pos_x_reco'
    pos_var_y = 'pos_y_reco'
    pos_var_z = 'pos_z_reco'
else:
    pos_var_x = 'pos_x'
    pos_var_y = 'pos_y'
    pos_var_z = 'pos_z'

simch_plots = []
if colorby=='edep':
    simch_plot = {
        "type":"scatter3d",
        "x":data[pos_var_x][:,0],
        "y":data[pos_var_y][:,0],
        "z":data[pos_var_z][:,0],
        "mode":"markers",
        "name":f"edep",
        "hovertemplate":hovertemplate,
        "customdata":customdata,
        "marker":{"color":data['edep'][:,0],"opacity":opacity,"size":marker_size,'colorscale':'Viridis','cmin':0.0,'cmax':5.0}
    }
    simch_plots.append(simch_plot)
elif colorby=='trackid':
    itrackid = data['trackid'][:,0].astype(np.int64)
    unique_ids = np.unique( itrackid )
    for tid in unique_ids:
        mask = itrackid==tid
        xcolor = np.random.randint(0,255,3)
        scolor=f'rgba({xcolor[0]},{xcolor[1]},{xcolor[2]},1)'
        simch_plot = {
            "type":"scatter3d",
            "x":data[pos_var_x][mask[:],0],
            "y":data[pos_var_y][mask[:],0],
            "z":data[pos_var_z][mask[:],0],
            "mode":"markers",
            "name":f"edep",
            "hovertemplate":hovertemplate,
            "customdata":customdata[mask[:],:],
            "marker":{"color":scolor,"opacity":opacity,"size":marker_size}
        }
        simch_plots.append(simch_plot)
else:
    print("unknown color option: ",colorby)
    sys.exit(0)



 
traces = detdata.getlines()+simch_plots

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
        "aspectratio": {"x": 1, "y": 1, "z": 4},
        "camera": {"eye": {"x": 2, "y": 2, "z": 2},
                   "up":dict(x=0, y=1, z=0)},
        "annotations": [],
    },
}

app.layout = html.Div( [
    html.Div( [
        dcc.Graph(
            id="det3d",
            figure={
                "data": traces,
                "layout": plot_layout,
            },
            config={"editable": True, "scrollZoom": False},
        )],
        className="graph__container"),
    ] )

app.run_server(debug=True)



