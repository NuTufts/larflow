Simple LArFlowHit display based on TEve.
Further functionality (e.g. plotting larflowcluster and pcaxis, hit color by category) is underway.

Usage:
In case you do not have gdml libraries:
1) To generate a geometry root file
root -l -b -q make_simplified_uboone.C 

2) To display event
root -l 'draw_geometry.C("path-to-larflowhits-file","larflowhits-producer",event-number,true)'

In case you do have gdml:
root -l 'draw_geometry.C("path-to-larflowhits-file","larflowhits-producer",event-number,false)'