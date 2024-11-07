import os,sys

class LArMatchHDF5Writer:

    ModuleList = ['preptriplets','kpana','ssnet','kpflow','truthfixer']

    def __init__(self, treename_for_adc_image="wire" ):
        
        # import bindings for ROOT-based c++ classes and functions
        # ROOT analysis framework
        import ROOT as rt
        # ROOT-based IO for image-like data to LAr TPCs
        from larcv import larcv
        # ROOT-based IO for data objects used in LArSoft (common framework used in LArTPC experiments )        
        from larlite import larlite
        # ROOT-based c++ objects for algorithms used to convert larcv and larlite into data for larmatch
        from larflow import larflow
        # ROOT-based C++ objects for algorithms to handle common simulation data parsing or other data analysis functions
        from ublarcvapp import ublarcvapp
        # file IO interface we are saving to
        import h5py 
        # turn off the stat box legend for plots
        rt.gStyle.SetOptStat(0)

        # larflow classes we use to convert larcv and larlite datat into larmatch training data

        # bad channel/gap channel maker
        self.badchmaker = ublarcvapp.EmptyChannelAlgo()

        # makes spacepoint proposals from larcv images and provides functions to provide spacepoints with groundtruth labels
        self.preptriplets = larflow.prep.PrepMatchTriplets()

        # keypoint score data
        self.kpana = larflow.keypoints.PrepKeypointData()
        #self.kpana.set_verbosity( larcv.msg.kDEBUG )
        self.kpana.setADCimageTreeName( treename_for_adc_image )
        self.adc_treename = treename_for_adc_image
        #tmp.cd()
        #kpana.defineAnaTree()

        # ssnet label data
        self.ssnet = larflow.prep.PrepSSNetTriplet()
        #tmp.cd()
        #ssnet.defineAnaTree()

        # direction field
        self.kpflow = larflow.keypoints.PrepAffinityField()
        #tmp.cd()
        #kpflow.defineAnaTree()

        # truth label corrections
        self.truthfixer = larflow.prep.TripletTruthFixer()    

        self.tick_backward = True
        self.entry_data = []


    def set_verbosity(self,verbosity,module):
        if module not in LArMatchHDF5Writer.ModuleList:
            print("WARNING: [set_verbosity] module asked to set verbosity for does not exist")
            print("possible modules: ",LArMatchHDF5Writer.ModuleList)
            return
        
        if module=='truthfixer':
            print("Setting TripletTruthFixer module, truthfixer, to verbosity level: ",verbosity)
            self.truthfixer.set_verbosity(verbosity)
        else:
            pass
        
        return


    def larlite_larcv_to_hdf5_entry( self, ioll, iolcv,
                                    run_process_truthlabels=False,
                                    num_max_spacepoints=10000000 ):

        # ROOT-based IO for image-like data to LAr TPCs
        from larcv import larcv
        # ROOT-based IO for data objects used in LArSoft (common framework used in LArTPC experiments )        
        from larlite import larlite
        # ROOT-based c++ objects for algorithms used to convert larcv and larlite into data for larmatch
        from larflow import larflow

        print("==== [[ LArMatchHDF5Writer ]] ========================")
        
        self.preptriplets.clear()
        #self.kpana
        #self.ssnet
        #self.kpflow
        #self.truthfixer
        
        ev_adc = iolcv.get_data( larcv.kProductImage2D, self.adc_treename )
        print("number of images: ",ev_adc.Image2DArray().size())
        adc_v = ev_adc.Image2DArray()
        for p in range(adc_v.size()):
            print(" image[",p,"] ",adc_v[p].meta().dump())
        sys.stdout.flush()
        
        ev_chstatus = iolcv.get_data( larcv.kProductChStatus, self.adc_treename )
        ev_larflow = iolcv.get_data( larcv.kProductImage2D, "larflow" )
        larflow_v  = ev_larflow.Image2DArray()
    
        badch_v = self.badchmaker.makeGapChannelImage( adc_v, ev_chstatus,
                                                  4, 3, 2400, 1008*6, 3456, 6, 1,
                                                  1.0, 100, -1.0 );
        print("  made badch_v, size=",badch_v.size())
        sys.stdout.flush()
    
        # make triplet proposals
        self.preptriplets.process( adc_v, badch_v, 10.0, True )

        if run_process_truthlabels:
            """ run code to make truth labels and convert them into numpy arrays """
            #print("Process truth labels")
            self.process_truthlabels( iolcv, ioll )

            # At this point, the spacepoints and labels are made.
            # we convert them into numpy arrays
            array_dict = self.convert_spacepoints_and_labels_to_numpy( self.preptriplets,
                                                                        self.kpana,
                                                                        self.ssnet,
                                                                        self.kpflow,
                                                                        num_max_spacepoints=num_max_spacepoints )
        else:
            """ no truth labels to make, so we just convert the triplet, wireplane, and spacepoints to numpy """
            array_dict = {}
            withtruth = False
            array_dict['matchtriplet_v'] = self.preptriplets.get_all_triplet_data( withtruth )
            for p in range(3):
                array_dict['wireimage_plane%d'%(p)] = self.preptriplets.make_sparse_image( p )

        self.entry_data.append( array_dict )

    def process_truthlabels(self, iolcv, ioll):

        # make good/bad triplet ground truth
        self.preptriplets.process_truth_labels( iolcv, ioll, self.adc_treename )

        # fix up some labels: handles edge cases due to the unoptimal way we stored information
        print("RUN TRIPLET TRUTHFIXER")
        self.truthfixer.calc_reassignments( self.preptriplets, iolcv, ioll )

        # make keypoint score ground truth
        print("RUN PrepKeypoint")    
        self.kpana.process( iolcv, ioll, self.preptriplets )
        self.kpana.make_proposal_labels( self.preptriplets )
        #self.kpana.fillAnaTree()

        # make ssnet ground truth
        print("Make SSNet ground truth")
        self.ssnet.make_ssnet_labels( iolcv, ioll, self.preptriplets )
    
        # fill happens automatically (ugh so ugly)
        # make affinity field ground truth
        self.kpflow.process( iolcv, ioll, self.preptriplets )
        #self.kpflow.fillAnaTree()

        #ev_mcshower = ioll.get_data( larlite.data.kMCShower, "mcreco" )
        #print("number of mcshower objects: ",ev_mcshower.size())
        #for i in range(ev_mcshower.size()):
        #    shower = ev_mcshower.at(i)
        #    print("[",i,"] geantid=",shower.TrackID()," pid=",shower.PdgCode(),")")
        #ev_mctrack = ioll.get_data( larlite.data.kMCTrack, "mcreco" )
        #print("number of mctrack objects: ",ev_mctrack.size())
        #for i in range(ev_mctrack.size()):
        #    track = ev_mctrack.at(i)
        #    print("[",i,"] geantid=",track.TrackID()," pid=",track.PdgCode(),")")
    
        #if args.save_triplets:
        #    triptree.Fill()
        
        return True
        


    def convert_spacepoints_and_labels_to_numpy( self, preptriplets, kpana,
                                                 ssnet, kpflow, 
                                                 num_max_spacepoints=10000000 ):
        # ROOT-based c++ objects for algorithms used to convert larcv and larlite into data for larmatch
        from larflow import larflow
        from ctypes import c_int
        loader = larflow.keypoints.LoaderKeypointData()
        loader.provide_entry_data( preptriplets, kpana, ssnet, kpflow )
         
        nfilled = c_int(0)
        array_dict = loader.sample_data( num_max_spacepoints, nfilled, True )
        return array_dict
        

    def writedata(self, output_filepath ):
        import h5py 

        with h5py.File(output_filepath, 'w') as hf:
            for ientry,entrydict in enumerate(self.entry_data):
                print("writing entry[",ientry,"]")
                for name in entrydict:
                    n = name+"_%d"%(ientry)
                    print("  write ",n," ",entrydict[name].shape)
                    hf.create_dataset( n, data=entrydict[name], compression='gzip', compression_opts=9 )
    
    def process_larcv_larlite_file( self, input_larcv, input_larlite, 
                                    output_filepath,
                                    start_entry=0, num_entries=-1,
                                    num_max_spacepoints=10000000,
                                    allow_output_overwrite=False ):
        from larcv import larcv
        # ROOT-based IO for data objects used in LArSoft (common framework used in LArTPC experiments )        
        from larlite import larlite
        # ROOT-based c++ objects for algorithms used to convert larcv and larlite into data for larmatch

        # check file paths, input and outpiut
        if not os.path.exists( input_larcv ):
            print("LARCV input file does not exist. path given: ",input_larcv)
            return 1
        if not os.path.exists( input_larlite ):
            print("larlite input file does not exist. path given: ", input_larlite)
            return 1
        if not allow_output_overwrite and os.path.exists( output_filepath ):
            print("output file exists. not allowing overwrites. path given: ",output_filepath)
            print("provide True to keyword argument 'allow_output_overwrite'")
            return 1
        if not os.path.exists( os.path.dirname(output_filepath) ):
            print("directory for output file does not exist. Given: ",os.path.dirname(output_filepath))
            return 1

        ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
        ioll.add_in_filename( input_larlite )
        ioll.open()

        if self.tick_backward:
            iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )
        else:
            iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickForward )
        iolcv.add_in_file( input_larcv )
        iolcv.reverse_all_products()
        iolcv.initialize()

        nentries_larcv = iolcv.get_n_entries()
        print("Number of entries in file: ",nentries_larcv)
        if start_entry>=nentries_larcv:
            print("Asking to start after last entry (%d) in file"%(nentries_larcv-1))
            return 1
        
        end_entry = start_entry + nentries_larcv
        if num_entries>0:
            end_entry = start_entry + num_entries
        if end_entry >= nentries_larcv:
            end_entry = nentries_larcv

        # event loop
        for ientry in range(start_entry,end_entry):
            print("[[ RUN ENTRY %d ]]"%(ientry))
            ioll.go_to(ientry)
            iolcv.read_entry(ientry)
            # convert the data and store into self.entry_data
            self.larlite_larcv_to_hdf5_entry( ioll, iolcv, True, num_max_spacepoints )

        # after we're done, we write the data
        self.writedata(output_filepath)
        return 0
