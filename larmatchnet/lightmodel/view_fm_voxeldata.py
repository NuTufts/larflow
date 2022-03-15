from __future__ import print_function
import os,sys
import chart_studio as cs
import chart_studio.plotly as py
import plotly.graph_objects as go
import numpy as np
import torch
from larcv import larcv
from larlite import larlite
from larflow import larflow
sys.path.append("../")
import lardly
#from larvoxel.larvoxelclass_dataset import larvoxelClassDataset
#from larvoxel_dataset import larvoxelDataset
from lightmodel.lm_dataset import LMDataset

# load utility to draw TPC outline
detdata = lardly.DetectorOutline()

# DATA LOADER
batch_size = 1
dataset = LMDataset( filelist=["test.root"], is_voxeldata=True )
nentries = len(dataset)

print("NENTRIES: ",nentries)

#for i in enumerate(dataset):
#    print(i)

loader = torch.utils.data.DataLoader( dataset, batch_size=batch_size, collate_fn=LMDataset.collate_fn )

niter = 11

# Get entry data
#batch = next(iter(loader))
#nvoxels = batch[0]["coord"].shape[0]
# We need to retrieved the 3d positions
#pos3d = batch[0]["coord"].astype(np.float64)*0.3
#pos3d[:,1] -= 117.0
#print(pos3d.shape)

for iiter in range(niter):
    batch = next(iter(loader))
    
    print("====================================")
    print("batch: ", batch)
    print(batch.keys())
    for name,d in batch.items():
        if  name in ["entry","tree_entry"]:
            print("  ",name,": ",d)
        elif type(d) is np.ndarray:
            print("  ",name,": ",d.shape)
        else:
            print("  ",name,": ",type(d))
            print(" coord: ",data["coord"][:10,:])
    
    nvoxels = batch["voxcoord"].shape[0]
    pos3d = batch["voxcoord"].astype(np.float64)*0.3
    #pos3d[:,1] -= 117.0
    #print(pos3d.shape)

    #PLOT!
    labelcol = np.mean( np.clip( batch["voxfeat"], 0, 3.0 )/3.0,axis=1 )
    ave3d = np.mean(pos3d,axis=0,keepdims=True).squeeze()
    ave3d[0] += 125.0
    ave3d[2] -= 1036
    ave3d[0] /= 256
    ave3d[1] /= 117.0
    ave3d[2] /= 1036
    particle_plot = {
    "type":"scatter3d",
    "x":pos3d[:,0],
    "y":pos3d[:,1],
    "z":pos3d[:,2],
        "mode":"markers",
        "name":"particle",
        "marker":{"color":labelcol,"size":1,"colorscale":"Viridis"}
    }

    detlines = detdata.getlines(color=(10,10,10))

    # DATA
    particle_plot_data = [particle_plot] + detdata.getlines(color=(10,10,10))
    # LAYOUT
    axis_template = {
        "showbackground": True,
        "backgroundcolor": "rgba(100, 100, 100,0.5)",
        "gridcolor": "rgb(50, 50, 50)",
        "zerolinecolor": "rgb(0, 0, 0)",
    }

    layout = go.Layout(
        title='PLOT VOXELS',
        autosize=True,
        hovermode='closest',
        showlegend=False,
        scene= {
            "xaxis": axis_template,
            "yaxis": axis_template,
            "zaxis": axis_template,
            "aspectratio": {"x": 1, "y": 1, "z": 3},
            "camera": {"eye": {"x": ave3d[0]+0.1, "y": ave3d[1], "z": ave3d[2]},
                   "center":dict(x=ave3d[0], y=ave3d[1], z=ave3d[2]),
                   "up":dict(x=0, y=1, z=0)},
            "annotations": [],
        }
    )

    fig = go.Figure(data=particle_plot_data, layout=layout)
    fig.show()
