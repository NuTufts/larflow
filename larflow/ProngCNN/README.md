# ProngCNN interface

This module is meant to provide tools to run the LArPID CNN through its C++ interface.

The LArPID network is a convolutional neural network that provides a classification score for five particle types:
electron, photon, muon, proton, and pions (+other mesons).

The ProngCNN interface is in the [prongCNN repository](http://github.com/nutufts/prongCNN)
and is currently local at /home/twongjirad/working/larbys/gen2/container_u20_env/work/prongCNN/larpid/larpid/.

We run the network for each particle associated to neutrino interaction candidates.
The candidates are represented by the larflow::reco::NuVertexCandidate class.
For each candidate there is a list of track-like particles and shower-like particles.
We assume that both have spacepoints associated with them, so the interface for both is the same.

For each, we make two images.
The first only includes pixels on the three wire plane images that correspond to the particle spaceoints.
The second includes all pixels around the start or end point of the particle.
For both, we crop a square region around the point.

These then get passed into a convolutional neural network (CNN).

There are some conditions for which particles are evaluated.

We have an example of applying LArPID from the make_dlgen2_flat_ntuple.py script:

```
... for each track ...

      # make image crop and count number of pixels mapped to particle in each wire plane image
      # apply a lower-bound threshold for applying the network.
      skip = True
      n_below_threshold = 0 
      if goodTrack:
        skip = False
        cropPt = vertex.track_v[iTrk].End()
        print(" track loop[",iTrk,"] calling make_cropped_initial_sparse_prong_image_reco(...)",flush=True)
        prong_vv = flowTriples.make_cropped_initial_sparse_prong_image_reco(adc_v,thrumu_v,trackCls,cropPt,10.,512,512)
        print(" track loop[",iTrk,"] prong_vv made",flush=True)        
        # 2024/11/27: weaken threshold to allow for one dead-plane
        for p in range(3):
          print("  plane[",p,"] track prong num of pixels: ",prong_vv[p].size())
          if prong_vv[p].size() < 10:
            #skip = True
            n_below_threshold += 1
        if n_below_threshold>1:
          skip = True
        sys.stdout.flush()
      trackNPlanesAbove[0] = n_below_threshold


      with torch.no_grad():
        print("make prong image: ",prong_vv.size(),flush=True)
        prongImage = makeImage(prong_vv).to(args.device)
        print("run prongCNN on track image",flush=True)
        prongCNN_out = model(prongImage)
      trackClassified[iTrk] = 1
      trackPID[iTrk] = getPID(prongCNN_out[0].argmax(1).item())
      trackElScore[iTrk] = prongCNN_out[0][0][0].item()
      trackPhScore[iTrk] = prongCNN_out[0][0][1].item()
      trackMuScore[iTrk] = prongCNN_out[0][0][2].item()
      trackPiScore[iTrk] = prongCNN_out[0][0][3].item()
      trackPrScore[iTrk] = prongCNN_out[0][0][4].item()
      trackComp[iTrk] = prongCNN_out[1].item()
      trackPurity[iTrk] = prongCNN_out[2].item()
      trackProcess[iTrk] = prongCNN_out[3].argmax(1).item()
      trackPrimaryScore[iTrk] = prongCNN_out[3][0][0].item()
      trackFromNeutralScore[iTrk] = prongCNN_out[3][0][1].item()
      trackFromChargedScore[iTrk] = prongCNN_out[3][0][2].item()
```

To Do:
  1. Create interace class that loads model [first draft done]
  2. Provide function to pass in a container of spacepoints for track and showers, and return scores. [draft done]
  3. Make executable program to run that takes in reco file and image file and runs larmatch, saving scores and other info to tree. 
  4. Run executable on one file and confirm that it produces same scores as those stored in existing (and vetted) ntuple.