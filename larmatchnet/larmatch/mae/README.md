# MAE Training

This folder is dedicated to modules for Masked Auto-Encoder training for spacepoints.

Each spacepoint can be mapped to a pixel within each of the three wire-plane images.

The training data is loaded using the LArMatchSimChHDF5Dataset class found in larmatch.data.larmatch_simchhdf5_reader.py.

During training we want to:

  1. First pass the sparse images into a common UNet image feature extractor. 
     We apply the LArMatchMinkowski model found in larmatch.model.larmatchminkowski.py.
      - We need to refactor the forward function to include the (a) image processing part, 
        (b) spacepoint image feature vector extraction
        (c) the larmatch tasks [which we wont use for this model]
  2. We then form a feature vector for each spacepoint, 
     by concatenating the feature vector from each of the 3 pixels the spacepoint is associated with.
  3. The spacepoint image feature vector needs to be mixed with a sinusoidal encoding of the (x,y,z) position
  4. We also have for each spacepoint
     - the pixel values in each images (observable)
     - the true energy deposited, measured for each pixel (latent variable label coming from simulation)
     - a particle ID label (latent)
     - an instance label, where spacepoints with the same label are from the same particle (latent)
     - six keypont scores, indicating how close a spacepoint is to six different keypoint types (latent)
  5. We then mask some fraction of the pixel values for each image
  6. We take the unmasked token and put them in a transformer encoder.
  7. We put some fraction of the masked tokens back into the sequence. 
     - The masked tokens are made by a global 'masked' vector added with the sinusoidal position embedding vector
  8. We use a much shallower decoder to then predicts the pixel values.
  9. We also want to include a contrastive loss for same-particle versus different-particle representation.
     - We reward a high cosine-similarity score for the encoder vectors for spacepoints of the same particle
     - We can limit this to non-ghost spacepoints
  10. We also include an auxillary supervised loss, using the encoder output to predict various labels
  11. We also want to try using a co-distillation loss to help with the training.
  12. Use wandb to log training and validation metrics.

Issues:
  1. The number of spacepoints in each event can be large: 200k-600k per example
  2. The number of ghost to true spacepoint ratio is very high at around 10:1 to 20:1
  3. The number of spacepoints for a particle instance can vary widely.