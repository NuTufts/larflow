import os,sys
import argparse
import h5py
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

parser = argparse.ArgumentParser("Visualize HDF5 output from MCPixelLabels")
parser.add_argument("-ipt", "--input-points",required=True,type=str,help="Input HDF5 file.")
parser.add_argument("-ikp", "--input-keypoints",required=True,type=str,help="Input HDF5 file.")
parser.add_argument("-c",   "--colorby",  default='trackid', help="Color mode. Option: [instance,ancestor,edep]")
parser.add_argument("-p",   "--pos-mode", default="reco",type=str,help="Position mode. Option: [true,reco]")
parser.add_argument("-d",   "--use-data", action='store_true', default=False, help="If given, plot the data spacepoints")
args = parser.parse_args()

import lardly
from lardly.detectoroutline import DetectorOutline

fh5    = h5py.File(args.input_points, 'r')
fh5_kp = h5py.File(args.input_keypoints,'r')

opacity=0.8
marker_size=1.0

colorby_options  = ['edep','trackid']
pos_mode_options = ['true','reco']

kptype_color = {
    0:"rgba(255,0,0,1.0)",   # track start
    1:"rgba(0,0,255,1.0)",   # track end
    2:"rgba(255,0,125,1.0)", # shower start
    3:"rgba(125,0,255,1.0)", # michel start
    4:"rgba(0,125,255,1.0)", # delta start
}
kptype_name = {
    0:"TrackStart",
    1:"TrackEnd",
    2:"Shower",
    3:"Michel",
    4:"Delta"
}

if args.colorby not in colorby_options:
    print("Color Mode option given is invalid. Options: ",colorby_options)
    sys.exit(0)

if args.pos_mode not in pos_mode_options:
    print("Position Mode option invalid. Options: ",pos_mode_options)
    sys.exit(0)

#colorby = 'edep'
#colorby = 'trackid'
#colorby = 'hasmatch'
colorby = args.colorby
pos_var = args.pos_mode
#pos_var='true'
#pos_var='reco'

#NMAX_RECO_PTS=50000
NMAX_RECO_PTS=-1

if not args.use_data:
    triplet_truth = fh5['triplet_truth']
else:
    triplet_truth = fh5['triplet_data']
    pos_var='true'
mckeypoints   = fh5_kp['mckeypoints']

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
    if col not in triplet_truth:
        continue
    npts = len(triplet_truth[col])
    data[col] = np.array( triplet_truth[col], dtype=np.float32 )
    if len(data[col].shape)==1:
        data[col] = data[col].reshape((npts,1))
    print(col,": ",data[col].shape)
if args.use_data:
    data['edep'] = np.zeros( (data['pos_x'].shape[0],3),dtype=np.float32)


kpdata = {}
kpcols = ['pos','kptype','pid','trackid','imgcoord']
for col in kpcols:
    kpdata[col] = np.array( mckeypoints[col] )
    print(col,": ",kpdata[col].shape)

detdata = DetectorOutline()
customdata = np.concatenate( [data['pid'],data['trackid'],data['aid'],data['edep']],axis=1 )
print("customdata: ",customdata.shape)
hovertemplate = """
<b>x</b>: %{x:.1f}<br>
<b>y</b>: %{y:.1f}<br>
<b>z</b>: %{z:.1f}<br>
<b>PID</b>:  %{customdata[0]:d}<br>
<b>TID</b>:  %{customdata[1]:d}<br>
<b>AID</b>:  %{customdata[2]:d}<br>
<b>edep</b>: %{customdata[3]:.3f}, %{customdata[4]:.3f} , %{customdata[5]:.3f}  MeV<br>
"""

kpcustom = np.concatenate( (kpdata['pid'].reshape(-1,1), 
    kpdata['trackid'].reshape(-1,1), 
    kpdata['imgcoord']), 
    axis=1)
kphovertext = """
<b>x</b>: %{x:.1f}<br>
<b>y</b>: %{y:.1f}<br>
<b>z</b>: %{z:.1f}<br>
<b>PID</b>: %{customdata[0]:d}<br>
<b>TID</b>: %{customdata[1]:d}<br>
<b>IMG</b>: %{customdata[2]:d}, %{customdata[3]:d} , %{customdata[4]:d}<br>
"""

if pos_var=='reco':
    pos_var_x = 'pos_x_reco'
    pos_var_y = 'pos_y_reco'
    pos_var_z = 'pos_z_reco'
else:
    pos_var_x = 'pos_x'
    pos_var_y = 'pos_y'
    pos_var_z = 'pos_z'

source = data

simch_plots = []
if colorby=='edep':
    simch_plot = {
        "type":"scatter3d",
        "x":source[pos_var_x][:,0],
        "y":source[pos_var_y][:,0],
        "z":source[pos_var_z][:,0],
        "mode":"markers",
        "name":f"edep",
        "hovertemplate":hovertemplate,
        "customdata":customdata,
        "marker":{"color":source['edep'][:,0],"opacity":opacity,"size":marker_size,'colorscale':'Viridis','cmin':0.0,'cmax':5.0}
    }
    simch_plots.append(simch_plot)
elif colorby=='trackid':
    itrackid = source['trackid'][:,0].astype(np.int64)
    unique_ids = np.unique( itrackid )
    for tid in unique_ids:
        if tid<=0:
            continue
        mask = itrackid==tid
        xcolor = np.random.randint(0,255,3)
        scolor=f'rgba({xcolor[0]},{xcolor[1]},{xcolor[2]},1)'
        simch_plot = {
            "type":"scatter3d",
            "x":source[pos_var_x][mask[:],0],
            "y":source[pos_var_y][mask[:],0],
            "z":source[pos_var_z][mask[:],0],
            "mode":"markers",
            "name":f"tid[{tid}]",
            "hovertemplate":hovertemplate,
            "customdata":customdata[mask[:],:],
            "marker":{"color":scolor,"opacity":opacity,"size":marker_size}
        }
        simch_plots.append(simch_plot)
# elif colorby=='hasmatch':
#     simch_plot = {
#         "type":"scatter3d",
#         "x":source['pos_x'][:,0],
#         "y":source['pos_y'][:,0],
#         "z":source['pos_z'][:,0],    
#         "mode":"markers",
#         "name":f"recopts",
#         "marker":{"color":source['hasmatch'][:,0],"opacity":opacity,"size":marker_size,'colorscale':'Viridis'},
#         #"marker":{"color":"rgba(220,220,220,1)","opacity":opacity,"size":marker_size}
#     }
#     simch_plots.append( simch_plot )
else:
    print("unknown color option: ",colorby)
    sys.exit(0)

# MAKE KEYPOINT PLOT

kptypes = np.unique(kpdata['kptype'])
for ikptype in kptypes:
    kptype_mask = kpdata['kptype']==ikptype
    kptype_pos  = kpdata['pos'][kptype_mask[:],:]
    kptype_custom = kpcustom[kptype_mask[:],:]
    kp_plot = {
        "type":"scatter3d",
        "x":kptype_pos[:,0],
        "y":kptype_pos[:,1],
        "z":kptype_pos[:,2],    
        "mode":"markers",
        "name":kptype_name[ikptype],
        "hovertemplate":kphovertext,
        "customdata":kptype_custom,
        "marker":{"color":kptype_color[ikptype],"opacity":1.0,"size":3.0},
    }
    simch_plots.append( kp_plot )

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



