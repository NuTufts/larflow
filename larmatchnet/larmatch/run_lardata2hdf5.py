import os,sys,argparse,time

parser = argparse.ArgumentParser("run_lardata2hdf5: Convert LArCV/larlite data into HDF5 larmatch training data")
parser.add_argument("-ill", "--input-larlite",required=True,type=str,help="Input larlite file [required]")
parser.add_argument("-ilcv","--input-larcv",required=True,type=str,help="Input LArCV file [required]")
parser.add_argument("-o","--output",required=True,type=str,help="output file name [required]")
parser.add_argument("-adc", "--adc",type=str,default="wire",help="Name of tree with Wire ADC values [default: wire]")
parser.add_argument("-tb",  "--tick-backward",action='store_true',default=False,help="Input LArCV data is tick-backward [default: false]")
parser.add_argument("-tri", "--save-triplets",action='store_true',default=False,help="Save triplet data [default: false]")
parser.add_argument("-n",   "--nentries",type=int,default=-1,help="Number of entries to run [default: -1 (all)]")
parser.add_argument("-e",   "--start-entry",type=int,default=0,help="Entry to start [default: 0]")
parser.add_argument("-w",   "--allow-overwrite",default=False,action='store_true',help="If flag provided, allow overwrite of output file.")
parser.add_argument("-vt",   "--set-verbosity-truthfixer",type=int,default=2,help="Set verbosity level for module larflow::prep::TripletTruthFixer")
parser.add_argument("-vp",   "--set-verbosity-prepmatchtriplets",type=int,default=2,help="Give name of module to set verbosity level to larcv::msg::kDEBUG")
args = parser.parse_args()

from larmatch.data.larmatch_hdf5_writer import LArMatchHDF5Writer

writer = LArMatchHDF5Writer( args.adc )

writer.set_verbosity( args.set_verbosity_truthfixer, 'truthfixer' )
writer.set_verbosity( args.set_verbosity_prepmatchtriplets, 'preptriplets' )

print("======== START PROCESSING =================")
writer.process_larcv_larlite_file( args.input_larcv, args.input_larlite,
                                    args.output, start_entry=args.start_entry,
                                    num_entries=args.nentries,
                                    allow_output_overwrite=args.allow_overwrite)

print("============== FIN =========================")
