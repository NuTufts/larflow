# CRTMatch

Tools to match CRT hit and track objects to TPC tracks and PMT flashes.

# Contents

## `larflow::crtmatch::CRTTrackMach`

In `CRTTrackMatch.cxx/.h`

Use CRT Track objects to find path through images with charge. Produces though-going muon candidates.

### Configuring

# Guide


# To do

* use pca-cluster output to match to CRT hit [done]
* use crt-track to match to larflow hits via 2D pixels. This means projecting CRT track to image, collecting pixels near track, use pixels to collect space points. Associate flash. Save.
* for CRT hits, save spacepoints associated to CRT hits. Associate flash.