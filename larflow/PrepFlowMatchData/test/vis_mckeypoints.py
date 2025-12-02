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
parser.add_argument("-i", "--input-h5",required=True,type=str,help="Input HDF5 file.")
parser.add_argument("-e", "--entry",    required=True, type=int, help="Entry")
parser.add_argument("-c", "--colorby",  default='trackid', help="Color mode. Option: [instance,ancestor,edep]")
parser.add_argument("-p", "--pos-mode", default="reco",type=str,help="Position mode. Option: [true,reco]")
parser.add_argument("-d", "--use-data", action='store_true', default=False, help="If given, plot the data spacepoints")
args = parser.parse_args()

import lardly
from lardly.detectoroutline import DetectorOutline

fh5    = h5py.File(args.input_h5, 'r')

opacity=0.8
marker_size=2.0

colorby_options  = ['edep','trackid','hasmatch',
  'kpnu','kptrackstart','kptrackend','kpshower','kpmichel','kpdelta',
  'ssnet-label', 'ssnet-boundary', 'ssnet-weight']

pos_mode_options = ['true','reco']
kpindex = {
    'kpnu':0,
    'kptrackstart':1,
    'kptrackend':2,
    'kpshower':3,
    'kpmichel':4,
    'kpdelta':5
}

kptype_color = {
    0:"rgba(255,153,51,1.0)", # nu
    1:"rgba(255,0,0,1.0)",   # track start
    2:"rgba(0,0,255,1.0)",   # track end
    3:"rgba(255,0,125,1.0)", # shower start
    4:"rgba(125,0,255,1.0)", # michel start
    5:"rgba(0,125,255,1.0)", # delta start
}
kptype_name = {
    0:'Nu',
    1:"TrackStart",
    2:"TrackEnd",
    3:"Shower",
    4:"Michel",
    5:"Delta"
}

ssnet_class_colors = {
    0:'rgba(50,50,50,1.0)',  # background
    1:'rgba(255,0,0,1.0)',   # electron
    2:'rgba(200,125,0.0,1)', # photon
    3:'rgba(0,0,255,1)',     # muon
    4:'rgba(0,125,255,1)',   # proton
    5:'rgba(125,0,255,1)',   # pion/kaon
    6:'rgba(125,125,0,1)',   # other
}

if args.colorby not in colorby_options:
    print("Color Mode option given is invalid. Options: ",colorby_options)
    sys.exit(0)

if args.pos_mode not in pos_mode_options:
    print("Position Mode option invalid. Options: ",pos_mode_options)
    sys.exit(0)

entry_groupname = f"entry_{args.entry}"
#colorby = 'edep'
#colorby = 'trackid'
#colorby = 'hasmatch'
colorby = args.colorby
pos_var_opt = args.pos_mode

#NMAX_RECO_PTS=50000
NMAX_RECO_PTS=-1

if not args.use_data:
    triplet_truth = fh5[f"{entry_groupname}/triplet_truth"]
else:
    triplet_truth = fh5[f"{entry_groupname}/triplet_data"]
    pos_var_opt='true'
mckeypoints   = fh5[f"{entry_groupname}/mckeypoints"]

# triplet_data columns
columns = ['pos',
    'pos_reco',
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
if args.use_data:
    columns += ['hasmatch','kpscores','ssnet_label','ssnet_boundary','ssnet_weight','pixval']


data = {}
for col in columns:
    if col not in triplet_truth:
        continue
    npts = len(triplet_truth[col])
    data[col] = np.array( triplet_truth[col], dtype=np.float32 )
    if len(data[col].shape)==1:
        data[col] = data[col].reshape((npts,1))
    print(col,": ",data[col].shape)

kpdata = {}
kpcols = ['pos','kptype','pid','trackid','imgcoord']
for col in kpcols:
    kpdata[col] = np.array( mckeypoints[col] )
    print(col,": ",kpdata[col].shape)

detdata = DetectorOutline()

if not args.use_data:
    hovertemplate = """
    <b>x</b>: %{x:.1f}<br>
    <b>y</b>: %{y:.1f}<br>
    <b>z</b>: %{z:.1f}<br>
    <b>PID</b>:  %{customdata[0]:d}<br>
    <b>TID</b>:  %{customdata[1]:d}<br>
    <b>AID</b>:  %{customdata[2]:d}<br>
    <b>edep</b>: %{customdata[3]:.3f}, %{customdata[4]:.3f} , %{customdata[5]:.3f}  MeV<br>
    """
    customdata = np.concatenate( [data['pid'],data['trackid'],data['aid'],data['edep']],axis=1 )
else:
    hovertemplate = """
    <b>x</b>: %{x:.1f}<br>
    <b>y</b>: %{y:.1f}<br>
    <b>z</b>: %{z:.1f}<br>
    <b>PID</b>:  %{customdata[0]:d}<br>
    <b>TID</b>:  %{customdata[1]:d}<br>
    <b>AID</b>:  %{customdata[2]:d}<br>
    <b>edep</b>: %{customdata[3]:.3f}, %{customdata[4]:.3f} , %{customdata[5]:.3f}  MeV<br>
    <b>pixval</b>: %{customdata[6]:.3f}, %{customdata[7]:.3f} , %{customdata[8]:.3f}  MeV<br>
    <b>Wires</b>: (%{customdata[9]:d}, %{customdata[10]:d} , %{customdata[11]:d})<br>
    <b>tick</b>: %{customdata[12]:d}
    """
    customdata = np.concatenate( [data['pid'],data['trackid'],data['aid'],data['edep'],data['pixval'],data['uwire'],data['vwire'],data['ywire'],data['tick']],axis=1 )
print("customdata: ",customdata.shape)

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

if pos_var_opt=='reco':
    pos_var = 'pos_reco'
else:
    pos_var = 'pos'

source = data

simch_plots = []
if colorby=='edep':
    simch_plot = {
        "type":"scatter3d",
        "x":source[pos_var][:,0],
        "y":source[pos_var][:,1],
        "z":source[pos_var][:,2],
        "mode":"markers",
        "name":f"edep",
        "hovertemplate":hovertemplate,
        "customdata":customdata,
        "marker":{"color":source['edep'][:,2],"opacity":opacity,"size":marker_size,'colorscale':'Viridis','cmin':0.0,'cmax':5.0}
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
            "x":source[pos_var][mask[:],0],
            "y":source[pos_var][mask[:],1],
            "z":source[pos_var][mask[:],2],
            "mode":"markers",
            "name":f"tid[{tid}]",
            "hovertemplate":hovertemplate,
            "customdata":customdata[mask[:],:],
            "marker":{"color":scolor,"opacity":opacity,"size":marker_size}
        }
        simch_plots.append(simch_plot)
elif colorby=='hasmatch':
    simch_plot = {
        "type":"scatter3d",
        "x":source[pos_var][:,0],
        "y":source[pos_var][:,1],
        "z":source[pos_var][:,2],    
        "mode":"markers",
        "name":f"recopts",
        "hovertemplate":hovertemplate,
        "customdata":customdata,
        "marker":{"color":source['hasmatch'][:,0],"opacity":opacity,"size":marker_size,'colorscale':'Viridis'},
    }
    simch_plots.append( simch_plot )
elif colorby in ['kpnu','kptrackstart','kptrackend','kpshower','kpmichel','kpdelta']:
    kpidx   = kpindex[colorby]
    kpscore = source['kpscores'][:,kpidx]
    simch_plot = {
        "type":"scatter3d",
        "x":source[pos_var][:,0],
        "y":source[pos_var][:,1],
        "z":source[pos_var][:,2],    
        "mode":"markers",
        "name":f"recopts",
        "hovertemplate":hovertemplate,
        "customdata":customdata,
        "marker":{"color":kpscore,"opacity":opacity,"size":marker_size,'colorscale':'Viridis'},
    }
    simch_plots.append( simch_plot )
elif colorby == 'ssnet-label':
    for iclass in range(7):
        ssnet_labels = source['ssnet_label'][:,0]
        ssnet_mask = ssnet_labels==iclass
        xcolor = ssnet_class_colors[iclass]
        simch_plot = {
            "type":"scatter3d",
            "x":source[pos_var][ssnet_mask[:],0],
            "y":source[pos_var][ssnet_mask[:],1],
            "z":source[pos_var][ssnet_mask[:],2],
            "mode":"markers",
            "name":f"ssnet[{iclass}]",
            "hovertemplate":hovertemplate,
            "customdata":customdata[ssnet_mask[:],:],
            "marker":{"color":xcolor,"opacity":opacity,"size":marker_size}
        }
        simch_plots.append(simch_plot)
elif colorby == 'ssnet-boundary':
    hasmatch = source["hasmatch"][:,0]==1
    ssnet_boundary = source['ssnet_boundary'][hasmatch[:],0]
    nboundary_max = np.max( ssnet_boundary )
    print("NUM SSNET BOUNDARY MAX: ",nboundary_max)
    fnboundary = ssnet_boundary.astype( np.float32 )/nboundary_max
    simch_plot = {
        "type":"scatter3d",
        "x":source[pos_var][hasmatch[:],0],
        "y":source[pos_var][hasmatch[:],1],
        "z":source[pos_var][hasmatch[:],2],
        "mode":"markers",
        "name":f"nboundary",
        "hovertemplate":hovertemplate,
        "customdata":customdata,
        "marker":{"color":fnboundary,"opacity":opacity,"size":marker_size,'colorscale':'Viridis'}
    }
    simch_plots.append(simch_plot)
elif colorby == 'ssnet-weight':
    hasmatch = source["hasmatch"][:,0]==1
    ssnet_weight = source['ssnet_weight'][hasmatch[:],0]
    simch_plot = {
        "type":"scatter3d",
        "x":source[pos_var][hasmatch[:],0],
        "y":source[pos_var][hasmatch[:],1],
        "z":source[pos_var][hasmatch[:],2],
        "mode":"markers",
        "name":f"ssweight",
        "hovertemplate":hovertemplate,
        "customdata":customdata,
        "marker":{"color":ssnet_weight,"opacity":opacity,"size":marker_size,'colorscale':'Viridis'}
    }
    simch_plots.append(simch_plot)
else:
    print("unknown color option: ",colorby)
    sys.exit(0)

# MAKE KEYPOINT PLOT

kptypes = np.unique(kpdata['kptype'])
for ikptype in kptypes:
    kptype_mask = kpdata['kptype']==ikptype
    kptype_pos  = kpdata['pos'][kptype_mask[:],:]
    kptype_custom = kpcustom[kptype_mask[:],:]

    msize = 5.0
    if ikptype==0:
        msize = 8.0

    kp_plot = {
        "type":"scatter3d",
        "x":kptype_pos[:,0],
        "y":kptype_pos[:,1],
        "z":kptype_pos[:,2],    
        "mode":"markers",
        "name":kptype_name[ikptype],
        "hovertemplate":kphovertext,
        "customdata":kptype_custom,
        "marker":{"color":kptype_color[ikptype],"opacity":1.0,"size":msize},
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



