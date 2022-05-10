import sys
from array import array
from ROOT import TCanvas, TGraph2D, TH1D
from larlite import larlite

if len(sys.argv) < 3:
    msg  = '\n'
    msg += "Usage 1: %s $INPUT_ROOT_FILE(s) $ENTRY\n" % sys.argv[0]
    msg += '\n'
    sys.stderr.write(msg)
    sys.exit(1)

entry = int(sys.argv[2])
    
c1 = TCanvas( 'c1', 'Canvas', 200, 10, 700, 500 )
 
c1.SetFillColor( 42 )
c1.SetGrid()
    
ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
#ioll.set_data_to_read( larlite.data.kMCTrack,  "mcreco" )
ioll.add_in_filename( sys.argv[1] )
ioll.open()

ll_nentries = ioll.get_entries()
print("ll_nentries: ",ll_nentries)
ioll.go_to( entry )

ev_mctrack = ioll.get_data(larlite.data.kMCTrack,"mcreco");
print("Number of tracks in event: ", ev_mctrack.size() )

mctrack = ev_mctrack.at(0)

print("EventID is: ", ioll.event_id() )
print("TrackID is: ", mctrack.TrackID() )
print("Starting X position is: ", mctrack.Start().X())
print("Starting Y position is: ", mctrack.Start().Y())
print("Starting Z position is: ", mctrack.Start().Z())

print("First mcstep X:", mctrack.at(0).X() )
print("First mcstep Y:", mctrack.at(0).Y() )
print("First mcstep Z:", mctrack.at(0).Z() )

#vx = []
#vy = []
#vz = []

vx, vy, vz = array( 'd' ), array( 'd' ), array( 'd' )

for mcstep in mctrack:
    vx.append( mcstep.X() )
    vy.append( mcstep.Y() )
    vz.append( mcstep.Z() )
    print( "mcstep X values: ",mcstep.X() )
    print( "mcstep Y values: ",mcstep.Y() )
    print( "mcstep Z values: ",mcstep.Z() )

print(vx)

gr = TGraph2D( len(vx), vx, vy, vz )

gr.SetLineColor( 2 )
gr.SetLineWidth( 4 )
gr.SetMarkerColor( 4 )
gr.SetMarkerStyle( 21 )
gr.SetTitle( 'MCTrack' )
gr.GetXaxis().SetTitle( 'X' )

gr.GetYaxis().SetTitle( 'Y' )
gr.GetYaxis().SetTitle( 'Z' )
gr.Draw( 'P' )

input = input("Press enter to continue...")

#my_proc = larlite.ana_processor()
#my_proc.set_io_mode(larlite.storage_manager.kREAD)
#my_proc.set_ana_output_file("myGraph.root");

#event_mctracks=storage.get_data<event_mctrack>("mcreco");
