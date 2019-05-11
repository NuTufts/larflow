import os,sys
import ROOT as rt
from larcv import larcv

#infile = sys.argv[1]
#producer=sys.argv[2]
infile = "/mnt/hdd1/twongj01/sparse_larflow_data/larflow_sparsify_cropped_train.root"
producer_y2u="sparsecropy2u"
producer_y2v="sparsecropy2v"

flows = ['y2u','y2v']

rt.gStyle.SetOptStat(0)

io = larcv.IOManager(larcv.IOManager.kREAD)
io.add_in_file(infile)
io.initialize()
thresh = 5.0

for ientry in xrange(io.get_n_entries()):

    if ientry>=5:
        break

    print "Entry[%d]"%(ientry),"::"
    io.read_entry(ientry)
    ev_sparse = { 'y2u':io.get_data(larcv.kProductSparseImage,producer_y2u),
                  'y2v':io.get_data(larcv.kProductSparseImage,producer_y2v) }
    sparseimg = {}
    nfeatures = {}
    meta_v    = {}
    npts      = {}
    nvalues   = {}
    for fl in flows:
        sparseimg[fl] = ev_sparse[fl].at(0)
        nfeatures[fl] = sparseimg[fl].nfeatures()
        meta_v[fl]    = sparseimg[fl].meta_v()
        npts[fl]      = sparseimg[fl].pixellist().size()/(2+nfeatures[fl])
        nvalues[fl]   = sparseimg[fl].pixellist().size()
        print "  len(ev_sparse[%s])="%(fl),ev_sparse[fl].SparseImageArray().size(),
        print "  nfeatures=",nfeatures[fl],
        print "  npts=",npts[fl],
        print "  sparsefrac=%.3f"%( npts[fl]/float(832*512) )

    ncols = meta_v['y2u'].front().cols()
    nrows = meta_v['y2u'].front().rows()
    origin_yplane_x = meta_v['y2u'].at(0).min_x()
    origin_yplane_y = meta_v['y2u'].at(0).min_y()
    
    print "  yplane origin=(%.2f,%.2f)"%(origin_yplane_x,origin_yplane_y)

    c = {}
    h = {}
    for ifeat in xrange(nfeatures['y2u']):
        yplane_maxx     = meta_v['y2u'].at(ifeat).max_x()
        yplane_maxy     = meta_v['y2u'].at(ifeat).max_y()
        origin_y        = meta_v['y2u'].at(ifeat).min_y()
        origin_x        = meta_v['y2u'].at(ifeat).min_x()
        c[ifeat] = rt.TCanvas("c%d"%(ifeat),"Image %d"%(ifeat),1400,600)
        c[ifeat].Divide(2,1)
        for fl in flows:
            h[(fl,ifeat)] = rt.TH2D("h%s_%d"%(fl,ifeat),"",ncols,origin_x,yplane_maxx,nrows,origin_y,yplane_maxy)
    c[nfeatures['y2u']] = rt.TCanvas("cflowerrs","Flow errors",1200,600)
    c[nfeatures['y2u']].Divide(2,1)
    for fl in flows:
        h[(fl,nfeatures['y2u'])] = h[(fl,0)].Clone("h%s_flowerrs"%(fl))


    # checking values
    nsource_above_thresh       = {}
    nsource_above_thresh_wflow = {}   
    nflow2nothing              = {}
    nflow2good                 = {}
    for fl in flows:
        nsource_above_thresh[fl] = 0
        nsource_above_thresh_wflow[fl] = 0
        nflow2nothing[fl] = 0
        nflow2good[fl] = 0

        # draw source and target images    
        for ipt in xrange(npts[fl]):
            start = ipt*(2+nfeatures[fl])
            row = int(sparseimg[fl].pixellist()[start])
            col = int(sparseimg[fl].pixellist()[start+1])
            for ifeat in xrange(2):
                featval = sparseimg[fl].pixellist()[start+2+ifeat]
                h[(fl,ifeat)].SetBinContent( col+1, row+1, featval )

        # evaluate errors, draw flow
        for ipt in xrange(npts[fl]):
            start = ipt*(2+nfeatures[fl])
            row = int(sparseimg[fl].pixellist()[start])
            col = int(sparseimg[fl].pixellist()[start+1])    
            src_feat_above_thresh = False
            for ifeat in xrange(nfeatures[fl]):
                featval = sparseimg[fl].pixellist()[start+2+ifeat]            
                if ifeat==0 and featval>=thresh:
                    src_feat_above_thresh = True
                    nsource_above_thresh[fl]  += 1
                    #print ifeat," ",featval," ",nsource_above_thresh
                elif ifeat==2:
                    if src_feat_above_thresh and featval>-4000:
                        # reach for target feature
                        nsource_above_thresh_wflow[fl] += 1
                        srcval = h[(fl,0)].GetBinContent( int(col+1), row+1 )
                        tarval = h[(fl,1)].GetBinContent( int(featval+col+1), row+1 )
                        if tarval>=thresh:
                            nflow2good[fl] += 1
                        else:
                            h[(fl,3)].SetBinContent( col+1, row+1, 10.0 )
                            h[(fl,1)].SetBinContent( int(featval+col+1), row+1, 1.0 )
                            nflow2nothing[fl] += 1
                            print "  flow2nothing[%s]: src=%d tar=%d srcadc=%.2f taradc=%.2f flow=%.2f"%(fl,col,featval+col,srcval,tarval,featval)
                    if featval>-4000:
                        h[(fl,ifeat)].SetBinContent( col+1, row+1, featval )

        print "   [%s] nsrcabovethresh=%d wflow=%d"%(fl,nsource_above_thresh[fl],nsource_above_thresh_wflow[fl])
        print "   [%s] fracgood=%.2f fracbad=%.2f"%(fl,nflow2good[fl]/float(nsource_above_thresh_wflow[fl]),
                                                    nflow2nothing[fl]/float(nsource_above_thresh_wflow[fl]))

    for ifeat in xrange(4):
        c[ifeat].cd()
        c[ifeat].Draw()

        c[ifeat].cd(1)
        h[('y2u',ifeat)].Draw("COLZ")
        c[ifeat].cd(2)
        h[('y2v',ifeat)].Draw("COLZ")
        
        c[ifeat].Update()
        c[ifeat].SaveAs("dumpedimages/test_sparseimg_entry%d_img%d.pdf"%(ientry,ifeat))
        
    print "[ENTER] for next entry"
    if True:
        raw_input()
        
