import six
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import ROOT as rt
from larcv import larcv
from larlite import larlite
from ROOT import larutil

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from matplotlib import colors

# INPUT
inputfile = "../output_pixmatch_larlite.root"
io = larlite.storage_manager(larlite.storage_manager.kREAD)
io.add_in_filename( inputfile )
io.open()
# get larflow hits
io.go_to(0)
ev_larflow = io.get_data(larlite.data.kLArFlow3DHit, "flowhits" )
nhits = ev_larflow.size()
#nhits = 200
print "Number of larflow hits: ",nhits

# our data
pos_np = np.zeros( (nhits,3) )
id_np  = np.zeros( (nhits) )
for ihit in xrange(nhits):
    hit = ev_larflow.at(ihit)
    pos_np[ihit,0] = hit.at(0)
    pos_np[ihit,1] = hit.at(1)
    pos_np[ihit,2] = hit.at(2)
    id_np[ihit]    = hit.trackid

X = pos_np
y = id_np

# normalize dataset for easier parameter selection
X = StandardScaler().fit_transform(X)

# DEFINE ALGOS
default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 20}

params = default_base.copy()
dbscan = cluster.DBSCAN(eps=params['eps'])
birch = cluster.Birch(n_clusters=params['n_clusters'])
gmm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')
            
estimators = (
    ('DBSCAN', dbscan),
    ('Birch', birch),
    ('GaussianMixture', gmm)
)
estimators = [('DBSCAN', dbscan)]

# get colors
colors_ = list(six.iteritems(colors.cnames))
for i, (n,h) in enumerate(colors_):
    colors_[i] = h

fignum = 1
titles = ['dbscan', 'birch', 'gmm']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X)
    if hasattr(est, 'labels_'):
        labels = est.labels_.astype(np.int)
    else:
        labels = est.predict(X)
    colors_ = np.array(colors_)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2],
               s=10, c=colors_[labels], depthshade=False, edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1

    fig.show()
'''    
# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
                
ax.scatter(X[:, 2], X[:, 0], X[:, 2], c=colors_[y.astype(np.float)], edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Ground Truth')
ax.dist = 12
fig.show()
'''
raw_input()
