import os,sys,time

from serverfeed.server import Server


##
## Contents of test file
##

"""
*         ../../testdata/smallsample/larcv_dlcosmictag_5482426_95_smallsample082918.root  
  KEY: TTree    image2d_infillCropped_tree;1    infillCropped tree
  KEY: TTree    image2d_larflow_y2u_tree;1      larflow_y2u tree
  KEY: TTree    image2d_adc_tree;1      adc tree
  KEY: TTree    image2d_pixvisi_tree;1  pixvisi tree
  KEY: TTree    image2d_pixflow_tree;1  pixflow tree
  KEY: TTree    image2d_larflow_y2v_tree;1      larflow_y2v tree
  KEY: TTree    image2d_ssnetCropped_shower_tree;1      ssnetCropped_shower tree
  KEY: TTree    image2d_ssnetCropped_endpt_tree;1       ssnetCropped_endpt tree
  KEY: TTree    image2d_ssnetCropped_track_tree;1       ssnetCropped_track tree
"""

if __name__ == "__main__":

    broker = Server("*")
    broker.start(-1)
    
    
    

    
