import os,sys
import ROOT as rt
from larcv import larcv

#infile = sys.argv[1]
#producer=sys.argv[2]
infile = "/mnt/hdd1/twongj01/sparse_larflow_data/larflow_sparsify_cropped_train.root"
producer="sparsecropdual"

rt.gStyle.SetOptStat(0)

io = larcv.IOManager(larcv.IOManager.kREAD)
io.add_in_file(infile)
io.initialize()
thresh = 10.0

for ientry in xrange(io.get_n_entries()):

    if ientry>=5:
        break

    print "Entry[%d]"%(ientry),"::"
    io.read_entry(ientry)
    ev_sparse = io.get_data(larcv.kProductSparseImage,producer)

    sparseimg = ev_sparse.at(0)
    nfeatures = sparseimg.nfeatures()
    meta_v    = sparseimg.meta_v()
    npts      = sparseimg.pixellist().size()/(2+nfeatures)
    nvalues   = sparseimg.pixellist().size()
    print "  len(ev_sparse)=",ev_sparse.SparseImageArray().size(),
    print "  nfeatures=",nfeatures,
    print "  npts=",npts,
    print "  sparsefrac=%.3f"%( npts/float(832*512) )

    ncols = meta_v.front().cols()
    nrows = meta_v.front().rows()
    origin_yplane_x = meta_v.at(0).min_x()
    origin_yplane_y = meta_v.at(0).min_y()
    
    print "  yplane origin (wire,tick)=(%.2f,%.2f)"%(origin_yplane_x,origin_yplane_y)

    # we're plotting src, tar, flow
    c = rt.TCanvas("c","Cropped Flow",1500,800)
    c.Divide(3,2)

    # hists
    hsrc  = rt.TH2D("hsrc","",
                    ncols,meta_v.at(0).min_x(),meta_v.at(0).max_x(),
                    nrows,meta_v.at(0).min_y(),meta_v.at(0).max_y())
    htar1 = rt.TH2D("htar1","",
                    ncols,meta_v.at(1).min_x(),meta_v.at(1).max_x(),
                    nrows,meta_v.at(1).min_y(),meta_v.at(1).max_y())
    htar2 = rt.TH2D("htar2","",
                    ncols,meta_v.at(2).min_x(),meta_v.at(2).max_x(),
                    nrows,meta_v.at(2).min_y(),meta_v.at(2).max_y())
    hflow1 = rt.TH2D("hflow1","",
                     ncols,meta_v.at(3).min_x(),meta_v.at(3).max_x(),
                     nrows,meta_v.at(3).min_y(),meta_v.at(3).max_y())
    hflow2 = rt.TH2D("hflow2","",
                     ncols,meta_v.at(4).min_x(),meta_v.at(4).max_x(),
                     nrows,meta_v.at(4).min_y(),meta_v.at(4).max_y())
    
    #c = rt.TCanvas("cflowerrs","Flow errors",1200,600)
    #c.Divide(2,1)
    # for fl in flows:
    #     h[(fl,nfeatures['y2u'])] = h[(fl,0)].Clone("h%s_flowerrs"%(fl))


    # # checking values
    # nsource_above_thresh       = {}
    # nsource_above_thresh_wflow = {}   
    # nflow2nothing              = {}
    # nflow2good                 = {}

    nsource_above_thresh = 0
    nsource_above_thresh_wflow = 0
    nflow2nothing = 0
    nflow2good = 0
    
    # draw source and target images    
    for ipt in xrange(npts):
        start = ipt*(2+nfeatures)
        row = int(sparseimg.pixellist()[start])
        col = int(sparseimg.pixellist()[start+1])

        # ADC values. Cut on threshold.
        for ifeat,h in enumerate([hsrc,htar1,htar2]):
            featval = sparseimg.pixellist()[start+2+ifeat]
            if featval>thresh:
                h.SetBinContent( col+1, row+1, featval )

        # flow
        for ifeat,h in enumerate([hflow1,hflow2]):
            featval = sparseimg.pixellist()[start+5+ifeat]
            if featval>-4000.0:
                h.SetBinContent( col+1, row+1, featval )
            

    #     # evaluate errors, draw flow
    #     for ipt in xrange(npts[fl]):
    #         start = ipt*(2+nfeatures[fl])
    #         row = int(sparseimg[fl].pixellist()[start])
    #         col = int(sparseimg[fl].pixellist()[start+1])    
    #         src_feat_above_thresh = False
    #         for ifeat in xrange(nfeatures[fl]):
    #             featval = sparseimg[fl].pixellist()[start+2+ifeat]            
    #             if ifeat==0 and featval>=thresh:
    #                 src_feat_above_thresh = True
    #                 nsource_above_thresh[fl]  += 1
    #                 #print ifeat," ",featval," ",nsource_above_thresh
    #             elif ifeat==2:
    #                 if src_feat_above_thresh and featval>-4000:
    #                     # reach for target feature
    #                     nsource_above_thresh_wflow[fl] += 1
    #                     srcval = h[(fl,0)].GetBinContent( int(col+1), row+1 )
    #                     tarval = h[(fl,1)].GetBinContent( int(featval+col+1), row+1 )
    #                     if tarval>=thresh:
    #                         nflow2good[fl] += 1
    #                     else:
    #                         h[(fl,3)].SetBinContent( col+1, row+1, 10.0 )
    #                         h[(fl,1)].SetBinContent( int(featval+col+1), row+1, 1.0 )
    #                         nflow2nothing[fl] += 1
    #                         print "  flow2nothing[%s]: src=%d tar=%d srcadc=%.2f taradc=%.2f flow=%.2f"%(fl,col,featval+col,srcval,tarval,featval)
    #                 if featval>-4000:
    #                     h[(fl,ifeat)].SetBinContent( col+1, row+1, featval )

    #     print "   [%s] nsrcabovethresh=%d wflow=%d"%(fl,nsource_above_thresh[fl],nsource_above_thresh_wflow[fl])
    #     print "   [%s] fracgood=%.2f fracbad=%.2f"%(fl,nflow2good[fl]/float(nsource_above_thresh_wflow[fl]),
    #                                                 nflow2nothing[fl]/float(nsource_above_thresh_wflow[fl]))

    # dra canvas
    c.cd(1)
    hsrc.Draw("colz")
    c.cd(2)
    htar1.Draw("colz")
    c.cd(3)
    hflow1.Draw("colz")

    
    c.cd(4)
    hsrc.Draw("colz")
    c.cd(5)
    htar2.Draw("colz")
    c.cd(6)
    hflow2.Draw("colz")
    
    # for ifeat in xrange(4):
    #     c[ifeat].cd()
    #     c[ifeat].Draw()

    #     c[ifeat].cd(1)
    #     h[('y2u',ifeat)].Draw("COLZ")
    #     c[ifeat].cd(2)
    #     h[('y2v',ifeat)].Draw("COLZ")
        
    #     c[ifeat].Update()
    #     c[ifeat].SaveAs("dumpedimages/test_sparseimg_entry%d_img%d.pdf"%(ientry,ifeat))

    c.Draw()
    c.Update()
        
    print "[ENTER] for next entry"
    if True:
        raw_input()
        
