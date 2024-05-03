import numpy as np
from larlite import larlite

def get_reco_flash_vectors( ioll ):
    """
    Gets the reconstructed flashes we will match charge clusters to.
    We get the data from the larlite data structures.
    We assume that an event has already been loaded before
      this is called.
    """
    
    # reco flash vectors
    producer_v = ["simpleFlashBeam","simpleFlashCosmic"]
    flash_np_v = {}
    for iproducer,producer in enumerate(producer_v):
        
        flash_beam_v = ioll.get_data( larlite.data.kOpFlash, producer )
    
        for iflash in range( flash_beam_v.size() ):
            flash = flash_beam_v.at(iflash)

            # we need to make the flash vector, the target output
            flash_np = np.zeros( flash.nOpDets(), dtype=np.float32 )
            
            for iopdet in range( flash.nOpDets() ):
                flash_np[iopdet] = flash.PE(iopdet)

            # uboone has 4 pmt groupings
            score_group = {}
            for igroup in range(4):
                score_group[igroup] = flash_np[ 100*igroup: 100*igroup+32 ].sum()
            print(" [",producer,"] iflash[",iflash,"]: ",score_group)
            
            if producer=="simpleFlashBeam":
                flash_np_v[(iproducer,iflash)] = flash_np[0:32]
            elif producer=="simpleFlashCosmic":
                flash_np_v[(iproducer,iflash)] = flash_np[200:232]
    return flash_np_v
