import os,sys
import ROOT as rt

"""
******************************************************************************
*Tree    :nushowerbuilder_mcana_tree: MC Analysis to evaluate and tune NuShowerBuilder Algorithm *
*Entries :     2534 : Total =          181322 bytes  File  Size =      70404 *
*        :          : Tree compression factor =   2.52                       *
******************************************************************************
*Br    0 :closest_recovtx_dist : closest_recovtx_dist/F                      *
*Entries :     2534 : Total  Size=      10790 bytes  File Size  =        521 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=  19.66     *
*............................................................................*
*Br    1 :trueprong_pixsum_MeV : trueprong_pixsum_MeV/F                      *
*Entries :     2534 : Total  Size=      10790 bytes  File Size  =        780 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=  13.13     *
*............................................................................*
*Br    2 :trueprong_efficiency : trueprong_efficiency/F                      *
*Entries :     2534 : Total  Size=      10790 bytes  File Size  =        678 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=  15.11     *
*............................................................................*
*Br    3 :trueprong_dist2vtx : trueprong_dist2vtx/F                          *
*Entries :     2534 : Total  Size=      10780 bytes  File Size  =        793 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=  12.92     *
*............................................................................*
*Br    4 :recofragment_purity : recofragment_purity/F                        *
*Entries :     2534 : Total  Size=      10785 bytes  File Size  =        688 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=  14.89     *
*............................................................................*
*Br    5 :recofragment_dist2vtx : recofragment_dist2vtx/F                    *
*Entries :     2534 : Total  Size=      10795 bytes  File Size  =       9189 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.12     *
*............................................................................*
*Br    6 :recofragment_impactpar : recofragment_impactpar/F                  *
*Entries :     2534 : Total  Size=      10800 bytes  File Size  =       9432 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
*Br    7 :recofragment_cosine : recofragment_cosine/F                        *
*Entries :     2534 : Total  Size=      10785 bytes  File Size  =       6687 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.53     *
*............................................................................*
*Br    8 :recofragment_pixsum : recofragment_pixsum/F                        *
*Entries :     2534 : Total  Size=      10785 bytes  File Size  =       9227 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.11     *
*............................................................................*
*Br    9 :reco_outcome : reco_outcome/I                                      *
*Entries :     2534 : Total  Size=      10750 bytes  File Size  =       1201 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   8.52     *
*............................................................................*
*Br   10 :groundtruth_outcome : groundtruth_outcome/I                        *
*Entries :     2534 : Total  Size=      10785 bytes  File Size  =        470 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=  21.80     *
*............................................................................*
*Br   11 :trueprong_trunkdir : trueprong_trunkdir[3]/F                       *
*Entries :     2534 : Total  Size=      31058 bytes  File Size  =       1452 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=  21.02     *
*............................................................................*
*Br   12 :recofragment_trunkdir : recofragment_trunkdir[3]/F                 *
*Entries :     2534 : Total  Size=      31073 bytes  File Size  =      27887 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.09     *
*............................................................................*
"""

data_dir="/cluster/tufts/wongjiradlabnu/nutufts/data/v3dev_shower_mcanalysis/mcc9_v40_NC_Pi0_run3b/larflowreco/ana/"

tree = rt.TChain("nushowerbuilder_mcana_tree")

flist = os.popen("find %s -type f"%(data_dir))
for f in flist:
    f = f.strip()
    #print(f)
    tree.Add(f)
    
nentries = tree.GetEntries()
print("Number of entries: ",nentries)

fout = rt.TFile("out_nuvertexshower.root","recreate")

varlist = ["recofragment_cosine",
           "recofragment_dist2vtx",
           "recofragment_impactpar",
           "recofragment_pixsum"]
label = "groundtruth_outcome"

hcosine    = rt.TH1D("hcosine",   "",1000,-1.01, 1.01)
hcosine_bg = rt.TH1D("hcosine_bg","",1000,-1.01, 1.01)

hdist2vtx    = rt.TH1D("hdist2vtx",    "", 5000, 0, 1000 )
hdist2vtx_bg = rt.TH1D("hdist2vtx_bg", "", 5000, 0, 1000 )

himpactpar    = rt.TH1D("himpactpar",    "", 1000, 0, 100.0 )
himpactpar_bg = rt.TH1D("himpactpar_bg", "", 1000, 0, 100.0 )

hpixsum    = rt.TH1D("hpixsum",    "", 1000, 0, 500 )
hpixsum_bg = rt.TH1D("hpixsum_bg", "", 1000, 0, 500 )

tree.Draw("recofragment_cosine>>hcosine","groundtruth_outcome==1")
tree.Draw("recofragment_cosine>>hcosine_bg","groundtruth_outcome==0")
tree.Draw("recofragment_dist2vtx>>hdist2vtx","groundtruth_outcome==1")
tree.Draw("recofragment_dist2vtx>>hdist2vtx_bg","groundtruth_outcome==0")
tree.Draw("recofragment_impactpar>>himpactpar","groundtruth_outcome==1")
tree.Draw("recofragment_impactpar>>himpactpar_bg","groundtruth_outcome==0")
tree.Draw("recofragment_pixsum>>hpixsum","groundtruth_outcome==1")
tree.Draw("recofragment_pixsum>>hpixsum_bg","groundtruth_outcome==0")

fout.Write()
    
    
