import os,sys
import ROOT
from larlite import larlite

fname = sys.argv[1]

io = larlite.storage_manager( larlite.storage_manager.kREAD )
io.add_in_filename( fname )
io.open()

nentries = io.get_entries()

for i in xrange(5):

    io.go_to(i)
    evbeamflash   = io.get_data( larlite.data.kOpFlash, "simpleFlashBeam" )
    evcosmicflash = io.get_data( larlite.data.kOpFlash, "simpleFlashCosmic" )

    print "====================================================="
    print "====================================================="
    print "Entry: ",i
    print "  nbeam=",evbeamflash.size()
    print "  ncosmic=",evcosmicflash.size()

    for ib in xrange(evbeamflash.size()):
        flash = evbeamflash.at(ib)
        print "  beam[%d] petot=%.2f"%(ib,flash.TotalPE())
    for ic in xrange(evcosmicflash.size()):
        flash = evcosmicflash.at(ic)
        print "  cosmic[%d] petot=%.2f"%(ic,flash.TotalPE())

    break
    




