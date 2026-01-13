import os,sys

input_file = sys.argv[1]

from larlite import larlite
from ublarcvapp import ublarcvapp

mcpg = ublarcvapp.mctools.MCParticleGraph()
mcpg.cluster_nu_particles( True )

ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
ioll.add_in_filename( input_file )
ioll.open()

nentries = ioll.get_entries()

for ientry in range(nentries):
    print(f"======== ENTRY [{ientry}] ============")
    ioll.go_to(ientry)
    mcpg.clear()
    mcpg.buildgraph( ioll )
    mcpg.printGraph(0,False)
    #mcpg.printAllNodeInfo()
    if True:
        break
