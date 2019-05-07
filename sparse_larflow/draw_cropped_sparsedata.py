import os,sys
import ROOT as rt
from larcv import larcv

#infile = sys.argv[1]
#producer=sys.argv[2]
infile = "/mnt/hdd1/twongj01/sparse_larflow_data/larflow_sparsify_cropped_train.root"
producer="sparsecropy2u"

rt.gStyle.SetOptStat(0)

io = larcv.IOManager(larcv.IOManager.kREAD)
io.add_in_file(infile)
io.initialize()
thresh = 5.0

for ientry in xrange(io.get_n_entries()):

    if ientry>=5:
        break

    print "Entry[%d]"%(ientry),":",
    io.read_entry(ientry)
    ev_sparse = io.get_data(larcv.kProductSparseImage,producer)
    print " len(ev_sparse)=",ev_sparse.SparseImageArray().size()," ",
    sparseimg = ev_sparse.at(0)

    nfeatures = sparseimg.nfeatures()
    meta_v    = sparseimg.meta_v()
    ncols = meta_v.front().cols()
    nrows = meta_v.front().rows()
    origin_yplane_x = meta_v.at(0).min_x()
    origin_yplane_y = meta_v.at(0).min_y()
    
    print "nfeatures=%d"%(nfeatures)," yplane origin=(%.2f,%.2f)"%(origin_yplane_x,origin_yplane_y)    

    c = {}
    h = {}
    for ifeat in xrange(nfeatures):
        yplane_maxx     = meta_v.at(ifeat).max_x()
        yplane_maxy     = meta_v.at(ifeat).max_y()
        origin_y        = meta_v.at(ifeat).min_y()
        origin_x        = meta_v.at(ifeat).min_x()
        c[ifeat] = rt.TCanvas("c%d"%(ifeat),"Image %d"%(ifeat),1200,600)
        h[ifeat] = rt.TH2D("h%d"%(ifeat),"",ncols,origin_x,yplane_maxx,nrows,origin_y,yplane_maxy)
    c[nfeatures] = rt.TCanvas("cflowerrs","Flow errors",1200,600)
    h[nfeatures] = h[0].Clone("hflowerrs")
    


    nvalues = sparseimg.pixellist().size()
    npts    = nvalues/(2+nfeatures)

    # checking values
    nsource_above_thresh = 0
    nsource_above_thresh_wflow = 0    
    nflow2nothing = 0
    nflow2good    = 0

    # draw source and target images    
    for ipt in xrange(npts):
        start = ipt*(2+nfeatures)
        row = int(sparseimg.pixellist()[start])
        col = int(sparseimg.pixellist()[start+1])
        for ifeat in xrange(2):
            featval = sparseimg.pixellist()[start+2+ifeat]
            h[ifeat].SetBinContent( col+1, row+1, featval )

    # evaluate errors, draw flow            
    for ipt in xrange(npts):
        start = ipt*(2+nfeatures)
        row = int(sparseimg.pixellist()[start])
        col = int(sparseimg.pixellist()[start+1])    
        src_feat_above_thresh = False            
        for ifeat in xrange(nfeatures):
            featval = sparseimg.pixellist()[start+2+ifeat]            
            if ifeat==0 and featval>=thresh:
                src_feat_above_thresh = True
                nsource_above_thresh += 1
                #print ifeat," ",featval," ",nsource_above_thresh
            elif ifeat==2:
                if src_feat_above_thresh and featval>-4000:
                    # reach for target feature
                    nsource_above_thresh_wflow += 1
                    srcval = h[0].GetBinContent( int(col+1), row+1 )
                    tarval = h[1].GetBinContent( int(featval+col+1), row+1 )
                    if tarval>=thresh:
                        nflow2good += 1
                    else:
                        h[3].SetBinContent( col+1, row+1, 10.0 )
                        h[1].SetBinContent( int(featval+col+1), row+1, 1.0 )
                        nflow2nothing += 1
                        print "flow2nothing: src=%.2f tar=%.2f srcadc=%.2f taradc=%.2f flow=%.2f"%(col,featval+col,srcval,tarval,featval)
                if featval>-4000:
                    h[ifeat].SetBinContent( col+1, row+1, featval )

    print "   npts=",npts
    print "   nsrcabovethresh=%d wflow=%d"%(nsource_above_thresh,nsource_above_thresh_wflow)
    print "   fracgood=%.2f fracbad=%.2f"%(nflow2good/float(nsource_above_thresh_wflow),
                                           nflow2nothing/float(nsource_above_thresh_wflow))

    for ifeat in xrange(nfeatures+1):
        c[ifeat].cd()
        c[ifeat].Draw()
        h[ifeat].Draw("COLZ")
        c[ifeat].Update()
        c[ifeat].SaveAs("dumpedimages/test_sparseimg_entry%d_img%d.pdf"%(ientry,ifeat))
    print "[ENTER] for next entry"
    if True:
        raw_input()
        
