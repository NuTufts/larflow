import os,sys
import ROOT as rt

def make_particle_node_tgraph( node, adc_v ):
    #print "primary pid[",node.pid,"]"
    if node.pid in [-11,11,2212,13,-13,211,-211,321,-321,22]:
        #print "making tgraph for pid=",node.pid
        e_v = []
        for p in xrange(3):
            if node.pix_vv[p].size()==0:
                e_v.append(None)
                continue
            print "node pixels plane[",p,"]: ",node.pix_vv[p].size()/2
            meta = adc_v[p].meta()
            g = rt.TGraph( node.pix_vv[p].size()/2 )
            for j in xrange( node.pix_vv[p].size()/2 ):
                g.SetPoint(j, node.pix_vv[p][2*j+1], node.pix_vv[p][2*j] ) # wire, tick
            g.SetMarkerStyle(20)
            g.SetMarkerSize(0.5)                
            if node.pid in [11,-11,22]:
                if node.origin==1:
                    g.SetMarkerColor(rt.kRed)
            elif node.pid in [13,-13]:
                if node.origin==2:
                    g.SetMarkerColor(rt.kGreen)
                elif node.origin==1:
                    g.SetMarkerColor(rt.kMagenta)
            elif node.pid in [2212]:
                if node.origin==1:                    
                    g.SetMarkerColor(rt.kBlue)
            elif node.pid in [211,-211,321,-321]:
                if node.origin==1:                    
                    g.SetMarkerColor(rt.kOrange)
            e_v.append(g)
        return e_v
    else:
        return [None,None,None]



