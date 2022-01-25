// Want to histogram hits in 3D: # hits as a function of position for the 3 planes 

#include <iostream>
#include <string>
#include <cmath>
#include <vector>

#include "TFile.h"
#include "TTree.h"

#include "TH1D.h"
#include "TH2D.h"
#include "TH3D.h"

#include "larlite/core/DataFormat/storage_manager.h"
#include "larlite/core/DataFormat/larflow3dhit.h"
#include "larlite/core/DataFormat/larflowcluster.h"

int main( int nargs, char** argv ) {

  int hit_U = 0;
  int hit_V = 0;
  int hit_Y = 0;
  int tick = 0;

  float hit_x = 0.;
  float hit_y = 0.;
  float hit_z = 0.;

  int wireBins[] = {250, 250, 350};
  int wireMax[] = {2500, 2500, 3500};

  float xyzMin[] = {0., -130., 0.};
  float xyzMax[] = {250., 130., 1041.};
  int xyzBins[] = {250, 260, 1041};
  
  float xzzMin[] = {0., 0., 0.}; 
  float xzzMax[] = {250., 1041., 1041.};
  int xzzBins[] = {250, 1041, 1041};
  
  float yyxMin[] = {-130., -130., 0.}; 
  float yyxMax[] = {130., 130., 250.};
  int yyxBins[] = {260, 260, 250};

  // Input for ttree
  int hitsPerVoxel;
  
  std::string input_crtfile = argv[1];
  int startentry = atoi(argv[2]);
  int maxentries = atoi(argv[3]);

  larlite::storage_manager llio( larlite::storage_manager::kREAD );
  llio.add_in_filename( input_crtfile );
  llio.open();

  int nentries = llio.get_entries();
  
  TFile* outfile = new TFile(Form("crt_%d-%d.root",startentry,startentry+maxentries-1),"recreate");
  TTree *tree = new TTree("tree","tree of hits per voxel");
  tree->Branch("hitsPerVoxel", &hitsPerVoxel, "hitsPerVoxel/I");
 
  // Define histograms
  const int nhists = 3;
  std::string str1[3] = {"U","V","Y"};
  std::string str2[3] = {"x","y","z"};
  std::string str3[3] = {"x","z","z"};
  std::string str4[3] = {"y","y","x"};
  
  // wire hists
  TH1D* hitcount_wire_hist[ nhists ] = {nullptr};
  for (int n = 0; n < 3; n++ ) {
    char name[100];
    sprintf( name, "hitcount_wire_hist_%s", str1[n].c_str() );
    hitcount_wire_hist[n] = new TH1D( name, "wire #", wireBins[n], 0, wireMax[n]);
  }

  TH2D* hitcount_wire_th2d[ nhists ] = {nullptr};
  for (int n = 0; n < 3; n++ ) {
    char name[100];
    sprintf( name, "hitcount_wire_th2d_%s", str1[n].c_str() );
    hitcount_wire_th2d[n] = new TH2D( name, "wire #; tick", wireBins[n]*10, 0, wireMax[n], 1008, 2400, 8448);
  }

  // xyz hists
  TH1D* hitcount_xyz_hist[ nhists ] = {nullptr};
  for (int n = 0; n < 3; n++ ) {
    char name[100];
    sprintf( name, "hitcount_xyz_hist_%s", str2[n].c_str() );
    hitcount_xyz_hist[n] = new TH1D( name, ";position", xyzBins[n], xyzMin[n], xyzMax[n]);
  }

  TH2D* hitcount_xyz_th2d[ nhists ] = {nullptr};
  for (int n = 0; n < 3; n++ ) {
    char name[100];
    sprintf( name, "hitcount_xyz_th2d_%s%s", str3[n].c_str(), str4[n].c_str() );
    hitcount_xyz_th2d[n] = new TH2D( name, ";position ; position", xzzBins[n], xzzMin[n], xzzMax[n], yyxBins[n], yyxMin[n], yyxMax[n]);
  }

  TH3D* hitcount_xyz_th3d = nullptr;
  char name[100];
  sprintf( name, "hitcount_xyz_th3d");
  hitcount_xyz_th3d = new TH3D( name, ";position ;position ; position", (250), 0., 250., (260), -130., 130., (1041), 0., 1041.);

  // Loop over events
  for (int i = startentry; i < (startentry + maxentries); i++) {
    
    std::cout << "===========================================" << std::endl;
    std::cout << "[ Entry " << i << " ]" << std::endl;

    llio.go_to(i);

    larlite::event_larflowcluster* clusters_v = (larlite::event_larflowcluster*)llio.get_data(larlite::data::kLArFlowCluster,"fitcrttrack_larmatchhits");

    // loop thru clusters
    for ( size_t iCluster = 0; iCluster < clusters_v->size(); iCluster++ ) {

      const larlite::larflowcluster& cluster = clusters_v->at( iCluster );

      std::cout << "I'm in cluster: " << iCluster << std::endl;
      
      //      larlite::event_larflow3dhit* lfhits_v = (larlite::event_larflow3dhit*)llio.get_data(larlite::data::kLArFlow3DHit,"larmatch");

      // loop thru hits in this cluster
      for ( size_t iHit = 0; iHit < cluster.size(); iHit++ ) {

	const larlite::larflow3dhit& lfhit = cluster.at( iHit );

	hit_U = lfhit.targetwire[0];
	hit_V = lfhit.targetwire[1];
	hit_Y = lfhit.targetwire[2];
	tick = lfhit.tick;
	hit_x = lfhit[0];
	hit_y = lfhit[1];
	hit_z = lfhit[2];

	// fill wire hists
	hitcount_wire_hist[0]->Fill(hit_U);
	hitcount_wire_hist[1]->Fill(hit_V);
	hitcount_wire_hist[2]->Fill(hit_Y);

	hitcount_wire_th2d[0]->Fill(hit_U, tick);
	hitcount_wire_th2d[1]->Fill(hit_V, tick);
	hitcount_wire_th2d[2]->Fill(hit_Y, tick);

	// fill xyz hists
	hitcount_xyz_hist[0]->Fill(hit_x);
	hitcount_xyz_hist[1]->Fill(hit_y);
	hitcount_xyz_hist[2]->Fill(hit_z);

	hitcount_xyz_th2d[0]->Fill(hit_x, hit_y);
Last login: Mon Aug 17 12:47:52 on ttys000
MacBook-Pro:~ abratenko$ ls
Applications              Movies                    dllee_unified
ChimeraDev                Music                     lardly
Desktop                   Pictures                  matlab_crash_dump.47479-1
Documents                 Public                    temp
Downloads                 Testing                   ubdl
Dropbox                   cifar_net.pth
Library                   data
MacBook-Pro:~ abratenko$ ssh-XY  abratenko@lnstrex.mit.edu
-bash: ssh-XY: command not found
MacBook-Pro:~ abratenko$ ssh -XY abratenko@lnstrex.mit.edu
abratenko@lnstrex.mit.edu's password: 
Welcome to Ubuntu 18.04.2 LTS (GNU/Linux 5.0.0-25-generic x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage

 * Are you ready for Kubernetes 1.19? It's nearly here! Try RC3 with
   sudo snap install microk8s --channel=1.19/candidate --classic

   https://microk8s.io/ has docs and details.

 * Canonical Livepatch is available for installation.
   - Reduce system reboots and improve kernel security. Activate at:
     https://ubuntu.com/livepatch

271 packages can be updated.
0 updates are security updates.

Your Hardware Enablement Stack (HWE) is supported until April 2023.
*** System restart required ***
Last login: Tue Aug 18 14:18:24 2020 from 108.46.163.10
abratenko@trex:~$ ls
'#.bashrc#'   history.txt   lardly       larflow.sh~     setup_ubdl.sh~   ubdl
 data         h.txt         larflow.sh   setup_ubdl.sh   sparse_larflow
abratenko@trex:~$ source larflow.sh
SETUP TREX
LARLITE_BASEDIR = /home/abratenko/ubdl/larlite
LARLITE_PYTHON_VERSION =  2
LARLITE_PYTHON = python2
Specified UserDev packages to be compiled by a user:

Finish configuration. To build, type:
> cd $LARLITE_BASEDIR
> make

GEO2D_BASEDIR = /home/abratenko/ubdl/Geo2D
GEO2D_PYTHON_VERSION =  2
GEO2D_PYTHON = python2

Finish configuration. To build, type:
> cd $GEO2D_BASEDIR
> make

/home/abratenko/ubdl/LArOpenCV
LAROPENCV_PYTHON_VERSION =  2
LAROPENCV_PYTHON = python2
ANN: approximate nearest neighboor
    Found ANN package
Warning ... missing libTorch support. Build without them.

LArCV FYI shell env. may useful for external packages:
    LARCV_INCDIR   = /home/abratenko/ubdl/larcv/build/installed/include
    LARCV_LIBDIR   = /home/abratenko/ubdl/larcv/build/installed/lib
    LARCV_BUILDDIR = /home/abratenko/ubdl/larcv/build

Finish configuration. To build, type:
> cd $LARCV_BUILDDIR
> make 

abratenko@trex:~/ubdl/larflow$ ls
build           deprecated    Pangolin        start_tufts_container.sh
cilantro        docs          postprocessor   submit_compilation_to_grid.sh
cmake           GNUmakefile   python          testdata
CMakeLists.txt  larcvdataset  README.md       utils
configure.sh    larflow       serverfeed
container       larmatchnet   sparse_larflow
abratenko@trex:~/ubdl/larflow$ cd larflow/Ana/
abratenko@trex:~/ubdl/larflow/larflow/Ana$ sl

Command 'sl' not found, but can be installed with:

apt install sl
Please ask your administrator.

abratenko@trex:~/ubdl/larflow/larflow/Ana$ ls
crt_0-0.root                crt_0-9.root      hitsPerVoxel.root
crt_0-1318_10cmVoxels.root  CRTana            keypoint_recoana.cxx
crt_0-1318_1cmVoxels.root   CRTana.cxx        keypoint_truthana.cxx
crt_0-1318_3cmVoxels.root   CRTana.cxx~       kpsreco_vertexana
crt_0-1318_5cmVoxels.root   CRTvoxelHits.py   kpsreco_vertexana.cxx
crt_0-29.root               CRTvoxelHits.py~  README.md
crt_0-2.root                GNUmakefile
abratenko@trex:~/ubdl/larflow/larflow/Ana$ e CRTana.cxx
abratenko@trex:~/ubdl/larflow/larflow/Ana$ make
<< compile CRTana >>
g++ -g -fPIC `root-config --cflags` `larlite-config --includes` -I/home/abratenko/ubdl/larlite/../ `larcv-config --includes` `ublarcvapp-config --includes` -I/home/abratenko/ubdl/larflow/build/include  CRTana.cxx -o CRTana -L/home/abratenko/ubdl/larflow/build/lib -lLArFlow_LArFlowConstants -lLArFlow_PrepFlowMatchData -lLArFlow_KeyPoints `ublarcvapp-config --libs` -lLArCVApp_MCTools -lLArCVApp_ubdllee -lLArCVApp_UBWireTool -lLArCVApp_LArliteHandler `larcv-config --libs` -lLArCVCorePyUtil `larlite-config --libs` `root-config --libs`
CRTana.cxx: In function ‘int main(int, char**)’:
CRTana.cxx:124:44: error: base operand of ‘->’ has non-pointer type ‘const larlite::larflowcluster’
       for ( size_t iHit = 0; iHit < cluster->size(); iHit++ ) {
                                            ^~
GNUmakefile:21: recipe for target 'CRTana' failed
make: *** [CRTana] Error 1
abratenko@trex:~/ubdl/larflow/larflow/Ana$ e CRTana.cxx
abratenko@trex:~/ubdl/larflow/larflow/Ana$ make
<< compile CRTana >>
g++ -g -fPIC `root-config --cflags` `larlite-config --includes` -I/home/abratenko/ubdl/larlite/../ `larcv-config --includes` `ublarcvapp-config --includes` -I/home/abratenko/ubdl/larflow/build/include  CRTana.cxx -o CRTana -L/home/abratenko/ubdl/larflow/build/lib -lLArFlow_LArFlowConstants -lLArFlow_PrepFlowMatchData -lLArFlow_KeyPoints `ublarcvapp-config --libs` -lLArCVApp_MCTools -lLArCVApp_ubdllee -lLArCVApp_UBWireTool -lLArCVApp_LArliteHandler `larcv-config --libs` -lLArCVCorePyUtil `larlite-config --libs` `root-config --libs`
CRTana.cxx: In function ‘int main(int, char**)’:
CRTana.cxx:124:44: error: base operand of ‘->’ has non-pointer type ‘const larlite::larflowcluster’
       for ( size_t iHit = 0; iHit < cluster->size(); iHit++ ) {
                                            ^~
CRTana.cxx:126:46: error: base operand of ‘->’ has non-pointer type ‘const larlite::larflowcluster’
  const larlite::larflow3dhit& lfhit = cluster->at( iHit );
                                              ^~
GNUmakefile:21: recipe for target 'CRTana' failed
make: *** [CRTana] Error 1
abratenko@trex:~/ubdl/larflow/larflow/Ana$ e CRTana.cxx
abratenko@trex:~/ubdl/larflow/larflow/Ana$ make
<< compile CRTana >>
g++ -g -fPIC `root-config --cflags` `larlite-config --includes` -I/home/abratenko/ubdl/larlite/../ `larcv-config --includes` `ublarcvapp-config --includes` -I/home/abratenko/ubdl/larflow/build/include  CRTana.cxx -o CRTana -L/home/abratenko/ubdl/larflow/build/lib -lLArFlow_LArFlowConstants -lLArFlow_PrepFlowMatchData -lLArFlow_KeyPoints `ublarcvapp-config --libs` -lLArCVApp_MCTools -lLArCVApp_ubdllee -lLArCVApp_UBWireTool -lLArCVApp_LArliteHandler `larcv-config --libs` -lLArCVCorePyUtil `larlite-config --libs` `root-config --libs`
abratenko@trex:~/ubdl/larflow/larflow/Ana$ ls -plrt
total 139536
-rw-rw-r-- 1 abratenko abratenko      217 Aug  4 17:45 README.md
-rw-rw-r-- 1 abratenko abratenko    13937 Aug  4 17:45 kpsreco_vertexana.cxx
-rw-rw-r-- 1 abratenko abratenko     3533 Aug  4 17:45 keypoint_truthana.cxx
-rw-rw-r-- 1 abratenko abratenko    12714 Aug  4 17:45 keypoint_recoana.cxx
-rwxrwxr-x 1 abratenko abratenko  1476240 Aug  7 13:55 kpsreco_vertexana
-rw-r--r-- 1 abratenko abratenko      411 Aug  7 14:37 crt_0-0.root
-rw-r--r-- 1 abratenko abratenko      411 Aug 11 12:48 crt_0-2.root
-rw-r--r-- 1 abratenko abratenko   598080 Aug 13 13:10 crt_0-29.root
-rw-rw-r-- 1 abratenko abratenko     5089 Aug 17 15:27 CRTana.cxx~
-rw-rw-r-- 1 abratenko abratenko      766 Aug 17 15:49 GNUmakefile
-rw-rw-r-- 1 abratenko abratenko     1125 Aug 17 17:09 CRTvoxelHits.py~
-rw-rw-r-- 1 abratenko abratenko     1125 Aug 17 17:10 CRTvoxelHits.py
-rw-r--r-- 1 abratenko abratenko     5308 Aug 17 17:10 hitsPerVoxel.root
-rw-r--r-- 1 abratenko abratenko  6025901 Aug 17 17:16 crt_0-9.root
-rw-r--r-- 1 abratenko abratenko 69908217 Aug 17 17:26 crt_0-1318_1cmVoxels.root
-rw-r--r-- 1 abratenko abratenko 25095222 Aug 18 15:43 crt_0-1318_3cmVoxels.root
-rw-r--r-- 1 abratenko abratenko 20335364 Aug 18 15:52 crt_0-1318_5cmVoxels.root
-rw-r--r-- 1 abratenko abratenko 18418468 Aug 18 16:20 crt_0-1318_10cmVoxels.root
-rw-rw-r-- 1 abratenko abratenko     5266 Aug 21 10:54 CRTana.cxx
-rwxrwxr-x 1 abratenko abratenko   924328 Aug 21 10:54 CRTana
abratenko@trex:~/ubdl/larflow/larflow/Ana$ ls -lrt
total 139536
-rw-rw-r-- 1 abratenko abratenko      217 Aug  4 17:45 README.md
-rw-rw-r-- 1 abratenko abratenko    13937 Aug  4 17:45 kpsreco_vertexana.cxx
-rw-rw-r-- 1 abratenko abratenko     3533 Aug  4 17:45 keypoint_truthana.cxx
-rw-rw-r-- 1 abratenko abratenko    12714 Aug  4 17:45 keypoint_recoana.cxx
-rwxrwxr-x 1 abratenko abratenko  1476240 Aug  7 13:55 kpsreco_vertexana
-rw-r--r-- 1 abratenko abratenko      411 Aug  7 14:37 crt_0-0.root
-rw-r--r-- 1 abratenko abratenko      411 Aug 11 12:48 crt_0-2.root
-rw-r--r-- 1 abratenko abratenko   598080 Aug 13 13:10 crt_0-29.root
-rw-rw-r-- 1 abratenko abratenko     5089 Aug 17 15:27 CRTana.cxx~
-rw-rw-r-- 1 abratenko abratenko      766 Aug 17 15:49 GNUmakefile
-rw-rw-r-- 1 abratenko abratenko     1125 Aug 17 17:09 CRTvoxelHits.py~
-rw-rw-r-- 1 abratenko abratenko     1125 Aug 17 17:10 CRTvoxelHits.py
-rw-r--r-- 1 abratenko abratenko     5308 Aug 17 17:10 hitsPerVoxel.root
-rw-r--r-- 1 abratenko abratenko  6025901 Aug 17 17:16 crt_0-9.root
-rw-r--r-- 1 abratenko abratenko 69908217 Aug 17 17:26 crt_0-1318_1cmVoxels.root
-rw-r--r-- 1 abratenko abratenko 25095222 Aug 18 15:43 crt_0-1318_3cmVoxels.root
-rw-r--r-- 1 abratenko abratenko 20335364 Aug 18 15:52 crt_0-1318_5cmVoxels.root
-rw-r--r-- 1 abratenko abratenko 18418468 Aug 18 16:20 crt_0-1318_10cmVoxels.root
-rw-rw-r-- 1 abratenko abratenko     5266 Aug 21 10:54 CRTana.cxx
-rwxrwxr-x 1 abratenko abratenko   924328 Aug 21 10:54 CRTana
abratenko@trex:~/ubdl/larflow/larflow/Ana$ make
make: Nothing to be done for 'all'.
abratenko@trex:~/ubdl/larflow/larflow/Ana$ ./CRTana ~/data/hadded_crtFiles.root 0 1319
    [NORMAL]  <open> Opening a file in READ mode: /home/abratenko/data/hadded_crtFiles.root
===========================================
[ Entry 0 ]
I'm in cluster: 0
===========================================
[ Entry 1 ]
===========================================
[ Entry 2 ]
I'm in cluster: 0
===========================================
[ Entry 3 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 4 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 5 ]
===========================================
[ Entry 6 ]
I'm in cluster: 0
===========================================
[ Entry 7 ]
I'm in cluster: 0
===========================================
[ Entry 8 ]
===========================================
[ Entry 9 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 10 ]
I'm in cluster: 0
===========================================
[ Entry 11 ]
===========================================
[ Entry 12 ]
===========================================
[ Entry 13 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 14 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 15 ]
===========================================
[ Entry 16 ]
I'm in cluster: 0
===========================================
[ Entry 17 ]
I'm in cluster: 0
===========================================
[ Entry 18 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 19 ]
===========================================
[ Entry 20 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 21 ]
I'm in cluster: 0
===========================================
[ Entry 22 ]
===========================================
[ Entry 23 ]
===========================================
[ Entry 24 ]
I'm in cluster: 0
===========================================
[ Entry 25 ]
===========================================
[ Entry 26 ]
I'm in cluster: 0
===========================================
[ Entry 27 ]
I'm in cluster: 0
===========================================
[ Entry 28 ]
I'm in cluster: 0
===========================================
[ Entry 29 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 30 ]
===========================================
[ Entry 31 ]
===========================================
[ Entry 32 ]
I'm in cluster: 0
===========================================
[ Entry 33 ]
===========================================
[ Entry 34 ]
I'm in cluster: 0
===========================================
[ Entry 35 ]
I'm in cluster: 0
===========================================
[ Entry 36 ]
I'm in cluster: 0
===========================================
[ Entry 37 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 38 ]
===========================================
[ Entry 39 ]
===========================================
[ Entry 40 ]
I'm in cluster: 0
===========================================
[ Entry 41 ]
===========================================
[ Entry 42 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 43 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 44 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 45 ]
===========================================
[ Entry 46 ]
===========================================
[ Entry 47 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 48 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 49 ]
I'm in cluster: 0
===========================================
[ Entry 50 ]
===========================================
[ Entry 51 ]
===========================================
[ Entry 52 ]
===========================================
[ Entry 53 ]
===========================================
[ Entry 54 ]
I'm in cluster: 0
===========================================
[ Entry 55 ]
===========================================
[ Entry 56 ]
===========================================
[ Entry 57 ]
I'm in cluster: 0
===========================================
[ Entry 58 ]
I'm in cluster: 0
===========================================
[ Entry 59 ]
I'm in cluster: 0
===========================================
[ Entry 60 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 61 ]
===========================================
[ Entry 62 ]
===========================================
[ Entry 63 ]
===========================================
[ Entry 64 ]
===========================================
[ Entry 65 ]
I'm in cluster: 0
===========================================
[ Entry 66 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 67 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 68 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 69 ]
===========================================
[ Entry 70 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 71 ]
I'm in cluster: 0
===========================================
[ Entry 72 ]
===========================================
[ Entry 73 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 74 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 75 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 76 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 77 ]
===========================================
[ Entry 78 ]
I'm in cluster: 0
===========================================
[ Entry 79 ]
I'm in cluster: 0
===========================================
[ Entry 80 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 81 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 82 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 83 ]
===========================================
[ Entry 84 ]
I'm in cluster: 0
===========================================
[ Entry 85 ]
I'm in cluster: 0
===========================================
[ Entry 86 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 87 ]
===========================================
[ Entry 88 ]
===========================================
[ Entry 89 ]
I'm in cluster: 0
===========================================
[ Entry 90 ]
I'm in cluster: 0
===========================================
[ Entry 91 ]
===========================================
[ Entry 92 ]
I'm in cluster: 0
===========================================
[ Entry 93 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 94 ]
I'm in cluster: 0
===========================================
[ Entry 95 ]
I'm in cluster: 0
===========================================
[ Entry 96 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 97 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 98 ]
I'm in cluster: 0
===========================================
[ Entry 99 ]
===========================================
[ Entry 100 ]
I'm in cluster: 0
===========================================
[ Entry 101 ]
===========================================
[ Entry 102 ]
I'm in cluster: 0
===========================================
[ Entry 103 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 104 ]
I'm in cluster: 0
===========================================
[ Entry 105 ]
I'm in cluster: 0
===========================================
[ Entry 106 ]
===========================================
[ Entry 107 ]
I'm in cluster: 0
===========================================
[ Entry 108 ]
I'm in cluster: 0
===========================================
[ Entry 109 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 110 ]
I'm in cluster: 0
===========================================
[ Entry 111 ]
I'm in cluster: 0
===========================================
[ Entry 112 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 113 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 114 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 115 ]
===========================================
[ Entry 116 ]
I'm in cluster: 0
===========================================
[ Entry 117 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 118 ]
===========================================
[ Entry 119 ]
I'm in cluster: 0
===========================================
[ Entry 120 ]
===========================================
[ Entry 121 ]
===========================================
[ Entry 122 ]
I'm in cluster: 0
===========================================
[ Entry 123 ]
===========================================
[ Entry 124 ]
I'm in cluster: 0
===========================================
[ Entry 125 ]
===========================================
[ Entry 126 ]
===========================================
[ Entry 127 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 128 ]
===========================================
[ Entry 129 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 130 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 131 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 132 ]
I'm in cluster: 0
===========================================
[ Entry 133 ]
I'm in cluster: 0
===========================================
[ Entry 134 ]
I'm in cluster: 0
===========================================
[ Entry 135 ]
===========================================
[ Entry 136 ]
I'm in cluster: 0
===========================================
[ Entry 137 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 138 ]
===========================================
[ Entry 139 ]
I'm in cluster: 0
===========================================
[ Entry 140 ]
===========================================
[ Entry 141 ]
===========================================
[ Entry 142 ]
I'm in cluster: 0
===========================================
[ Entry 143 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 144 ]
I'm in cluster: 0
===========================================
[ Entry 145 ]
I'm in cluster: 0
===========================================
[ Entry 146 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 147 ]
I'm in cluster: 0
===========================================
[ Entry 148 ]
I'm in cluster: 0
===========================================
[ Entry 149 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 150 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 151 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 152 ]
I'm in cluster: 0
===========================================
[ Entry 153 ]
===========================================
[ Entry 154 ]
===========================================
[ Entry 155 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 156 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 157 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 158 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 159 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 160 ]
I'm in cluster: 0
===========================================
[ Entry 161 ]
I'm in cluster: 0
===========================================
[ Entry 162 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 163 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 164 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 165 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
I'm in cluster: 5
===========================================
[ Entry 166 ]
I'm in cluster: 0
===========================================
[ Entry 167 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 168 ]
I'm in cluster: 0
===========================================
[ Entry 169 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 170 ]
===========================================
[ Entry 171 ]
===========================================
[ Entry 172 ]
I'm in cluster: 0
===========================================
[ Entry 173 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 174 ]
I'm in cluster: 0
===========================================
[ Entry 175 ]
I'm in cluster: 0
===========================================
[ Entry 176 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 177 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 178 ]
===========================================
[ Entry 179 ]
===========================================
[ Entry 180 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 181 ]
I'm in cluster: 0
===========================================
[ Entry 182 ]
===========================================
[ Entry 183 ]
I'm in cluster: 0
===========================================
[ Entry 184 ]
I'm in cluster: 0
===========================================
[ Entry 185 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 186 ]
===========================================
[ Entry 187 ]
===========================================
[ Entry 188 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 189 ]
I'm in cluster: 0
===========================================
[ Entry 190 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 191 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 192 ]
===========================================
[ Entry 193 ]
===========================================
[ Entry 194 ]
===========================================
[ Entry 195 ]
I'm in cluster: 0
===========================================
[ Entry 196 ]
I'm in cluster: 0
===========================================
[ Entry 197 ]
I'm in cluster: 0
===========================================
[ Entry 198 ]
I'm in cluster: 0
===========================================
[ Entry 199 ]
I'm in cluster: 0
===========================================
[ Entry 200 ]
I'm in cluster: 0
===========================================
[ Entry 201 ]
I'm in cluster: 0
===========================================
[ Entry 202 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 203 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 204 ]
I'm in cluster: 0
===========================================
[ Entry 205 ]
===========================================
[ Entry 206 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 207 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 208 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 209 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 210 ]
I'm in cluster: 0
===========================================
[ Entry 211 ]
I'm in cluster: 0
===========================================
[ Entry 212 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 213 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 214 ]
===========================================
[ Entry 215 ]
I'm in cluster: 0
===========================================
[ Entry 216 ]
I'm in cluster: 0
===========================================
[ Entry 217 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 218 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 219 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 220 ]
I'm in cluster: 0
===========================================
[ Entry 221 ]
===========================================
[ Entry 222 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 223 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 224 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 225 ]
===========================================
[ Entry 226 ]
I'm in cluster: 0
===========================================
[ Entry 227 ]
I'm in cluster: 0
===========================================
[ Entry 228 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 229 ]
I'm in cluster: 0
===========================================
[ Entry 230 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 231 ]
===========================================
[ Entry 232 ]
I'm in cluster: 0
===========================================
[ Entry 233 ]
===========================================
[ Entry 234 ]
===========================================
[ Entry 235 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 236 ]
===========================================
[ Entry 237 ]
I'm in cluster: 0
===========================================
[ Entry 238 ]
===========================================
[ Entry 239 ]
I'm in cluster: 0
===========================================
[ Entry 240 ]
I'm in cluster: 0
===========================================
[ Entry 241 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 242 ]
===========================================
[ Entry 243 ]
===========================================
[ Entry 244 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 245 ]
===========================================
[ Entry 246 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 247 ]
I'm in cluster: 0
===========================================
[ Entry 248 ]
===========================================
[ Entry 249 ]
===========================================
[ Entry 250 ]
===========================================
[ Entry 251 ]
===========================================
[ Entry 252 ]
===========================================
[ Entry 253 ]
===========================================
[ Entry 254 ]
===========================================
[ Entry 255 ]
===========================================
[ Entry 256 ]
===========================================
[ Entry 257 ]
===========================================
[ Entry 258 ]
===========================================
[ Entry 259 ]
===========================================
[ Entry 260 ]
===========================================
[ Entry 261 ]
===========================================
[ Entry 262 ]
===========================================
[ Entry 263 ]
===========================================
[ Entry 264 ]
===========================================
[ Entry 265 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 266 ]
I'm in cluster: 0
===========================================
[ Entry 267 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 268 ]
===========================================
[ Entry 269 ]
I'm in cluster: 0
===========================================
[ Entry 270 ]
===========================================
[ Entry 271 ]
===========================================
[ Entry 272 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 273 ]
I'm in cluster: 0
===========================================
[ Entry 274 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 275 ]
===========================================
[ Entry 276 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 277 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 278 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 279 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 280 ]
I'm in cluster: 0
===========================================
[ Entry 281 ]
I'm in cluster: 0
===========================================
[ Entry 282 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 283 ]
===========================================
[ Entry 284 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 285 ]
I'm in cluster: 0
===========================================
[ Entry 286 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 287 ]
===========================================
[ Entry 288 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 289 ]
===========================================
[ Entry 290 ]
I'm in cluster: 0
===========================================
[ Entry 291 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 292 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 293 ]
===========================================
[ Entry 294 ]
===========================================
[ Entry 295 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 296 ]
===========================================
[ Entry 297 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 298 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 299 ]
===========================================
[ Entry 300 ]
===========================================
[ Entry 301 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 302 ]
I'm in cluster: 0
===========================================
[ Entry 303 ]
===========================================
[ Entry 304 ]
===========================================
[ Entry 305 ]
I'm in cluster: 0
===========================================
[ Entry 306 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 307 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 308 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 309 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 310 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 311 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 312 ]
===========================================
[ Entry 313 ]
===========================================
[ Entry 314 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 315 ]
===========================================
[ Entry 316 ]
I'm in cluster: 0
===========================================
[ Entry 317 ]
===========================================
[ Entry 318 ]
===========================================
[ Entry 319 ]
I'm in cluster: 0
===========================================
[ Entry 320 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 321 ]
I'm in cluster: 0
===========================================
[ Entry 322 ]
===========================================
[ Entry 323 ]
I'm in cluster: 0
===========================================
[ Entry 324 ]
I'm in cluster: 0
===========================================
[ Entry 325 ]
I'm in cluster: 0
===========================================
[ Entry 326 ]
===========================================
[ Entry 327 ]
I'm in cluster: 0
===========================================
[ Entry 328 ]
I'm in cluster: 0
===========================================
[ Entry 329 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 330 ]
===========================================
[ Entry 331 ]
===========================================
[ Entry 332 ]
===========================================
[ Entry 333 ]
I'm in cluster: 0
===========================================
[ Entry 334 ]
I'm in cluster: 0
===========================================
[ Entry 335 ]
I'm in cluster: 0
===========================================
[ Entry 336 ]
===========================================
[ Entry 337 ]
I'm in cluster: 0
===========================================
[ Entry 338 ]
===========================================
[ Entry 339 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 340 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 341 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 342 ]
I'm in cluster: 0
===========================================
[ Entry 343 ]
I'm in cluster: 0
===========================================
[ Entry 344 ]
I'm in cluster: 0
===========================================
[ Entry 345 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 346 ]
I'm in cluster: 0
===========================================
[ Entry 347 ]
I'm in cluster: 0
===========================================
[ Entry 348 ]
I'm in cluster: 0
===========================================
[ Entry 349 ]
I'm in cluster: 0
===========================================
[ Entry 350 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 351 ]
I'm in cluster: 0
===========================================
[ Entry 352 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 353 ]
I'm in cluster: 0
===========================================
[ Entry 354 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 355 ]
===========================================
[ Entry 356 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 357 ]
I'm in cluster: 0
===========================================
[ Entry 358 ]
===========================================
[ Entry 359 ]
===========================================
[ Entry 360 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 361 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 362 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 363 ]
I'm in cluster: 0
===========================================
[ Entry 364 ]
I'm in cluster: 0
===========================================
[ Entry 365 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 366 ]
===========================================
[ Entry 367 ]
===========================================
[ Entry 368 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 369 ]
I'm in cluster: 0
===========================================
[ Entry 370 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 371 ]
===========================================
[ Entry 372 ]
I'm in cluster: 0
===========================================
[ Entry 373 ]
I'm in cluster: 0
===========================================
[ Entry 374 ]
I'm in cluster: 0
===========================================
[ Entry 375 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 376 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 377 ]
===========================================
[ Entry 378 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 379 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 380 ]
===========================================
[ Entry 381 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 382 ]
I'm in cluster: 0
===========================================
[ Entry 383 ]
I'm in cluster: 0
===========================================
[ Entry 384 ]
I'm in cluster: 0
===========================================
[ Entry 385 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 386 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 387 ]
I'm in cluster: 0
===========================================
[ Entry 388 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 389 ]
I'm in cluster: 0
===========================================
[ Entry 390 ]
I'm in cluster: 0
===========================================
[ Entry 391 ]
I'm in cluster: 0
===========================================
[ Entry 392 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 393 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 394 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 395 ]
I'm in cluster: 0
===========================================
[ Entry 396 ]
I'm in cluster: 0
===========================================
[ Entry 397 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 398 ]
I'm in cluster: 0
===========================================
[ Entry 399 ]
===========================================
[ Entry 400 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 401 ]
===========================================
[ Entry 402 ]
I'm in cluster: 0
===========================================
[ Entry 403 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 404 ]
I'm in cluster: 0
===========================================
[ Entry 405 ]
===========================================
[ Entry 406 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 407 ]
I'm in cluster: 0
===========================================
[ Entry 408 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 409 ]
I'm in cluster: 0
===========================================
[ Entry 410 ]
I'm in cluster: 0
===========================================
[ Entry 411 ]
===========================================
[ Entry 412 ]
===========================================
[ Entry 413 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 414 ]
===========================================
[ Entry 415 ]
===========================================
[ Entry 416 ]
I'm in cluster: 0
===========================================
[ Entry 417 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 418 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 419 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 420 ]
===========================================
[ Entry 421 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 422 ]
===========================================
[ Entry 423 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 424 ]
===========================================
[ Entry 425 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 426 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 427 ]
I'm in cluster: 0
===========================================
[ Entry 428 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 429 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 430 ]
===========================================
[ Entry 431 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 432 ]
I'm in cluster: 0
===========================================
[ Entry 433 ]
I'm in cluster: 0
===========================================
[ Entry 434 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 435 ]
===========================================
[ Entry 436 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 437 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 438 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 439 ]
===========================================
[ Entry 440 ]
===========================================
[ Entry 441 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 442 ]
===========================================
[ Entry 443 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 444 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 445 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 446 ]
===========================================
[ Entry 447 ]
===========================================
[ Entry 448 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 449 ]
===========================================
[ Entry 450 ]
===========================================
[ Entry 451 ]
===========================================
[ Entry 452 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 453 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 454 ]
===========================================
[ Entry 455 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 456 ]
===========================================
[ Entry 457 ]
===========================================
[ Entry 458 ]
I'm in cluster: 0
===========================================
[ Entry 459 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 460 ]
I'm in cluster: 0
===========================================
[ Entry 461 ]
I'm in cluster: 0
===========================================
[ Entry 462 ]
===========================================
[ Entry 463 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 464 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 465 ]
I'm in cluster: 0
===========================================
[ Entry 466 ]
I'm in cluster: 0
===========================================
[ Entry 467 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 468 ]
I'm in cluster: 0
===========================================
[ Entry 469 ]
===========================================
[ Entry 470 ]
I'm in cluster: 0
===========================================
[ Entry 471 ]
I'm in cluster: 0
===========================================
[ Entry 472 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 473 ]
===========================================
[ Entry 474 ]
I'm in cluster: 0
===========================================
[ Entry 475 ]
===========================================
[ Entry 476 ]
I'm in cluster: 0
===========================================
[ Entry 477 ]
I'm in cluster: 0
===========================================
[ Entry 478 ]
===========================================
[ Entry 479 ]
===========================================
[ Entry 480 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 481 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 482 ]
I'm in cluster: 0
===========================================
[ Entry 483 ]
===========================================
[ Entry 484 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 485 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 486 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 487 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 488 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 489 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 490 ]
I'm in cluster: 0
===========================================
[ Entry 491 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 492 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 493 ]
===========================================
[ Entry 494 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 495 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 496 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 497 ]
I'm in cluster: 0
===========================================
[ Entry 498 ]
I'm in cluster: 0
===========================================
[ Entry 499 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 500 ]
===========================================
[ Entry 501 ]
===========================================
[ Entry 502 ]
I'm in cluster: 0
===========================================
[ Entry 503 ]
===========================================
[ Entry 504 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 505 ]
I'm in cluster: 0
===========================================
[ Entry 506 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 507 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 508 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 509 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 510 ]
===========================================
[ Entry 511 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 512 ]
I'm in cluster: 0
===========================================
[ Entry 513 ]
===========================================
[ Entry 514 ]
I'm in cluster: 0
===========================================
[ Entry 515 ]
I'm in cluster: 0
===========================================
[ Entry 516 ]
I'm in cluster: 0
===========================================
[ Entry 517 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 518 ]
===========================================
[ Entry 519 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 520 ]
I'm in cluster: 0
===========================================
[ Entry 521 ]
I'm in cluster: 0
===========================================
[ Entry 522 ]
===========================================
[ Entry 523 ]
I'm in cluster: 0
===========================================
[ Entry 524 ]
I'm in cluster: 0
===========================================
[ Entry 525 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 526 ]
===========================================
[ Entry 527 ]
===========================================
[ Entry 528 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 529 ]
I'm in cluster: 0
===========================================
[ Entry 530 ]
I'm in cluster: 0
===========================================
[ Entry 531 ]
===========================================
[ Entry 532 ]
I'm in cluster: 0
===========================================
[ Entry 533 ]
I'm in cluster: 0
===========================================
[ Entry 534 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 535 ]
===========================================
[ Entry 536 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 537 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 538 ]
I'm in cluster: 0
===========================================
[ Entry 539 ]
===========================================
[ Entry 540 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 541 ]
I'm in cluster: 0
===========================================
[ Entry 542 ]
I'm in cluster: 0
===========================================
[ Entry 543 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 544 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 545 ]
I'm in cluster: 0
===========================================
[ Entry 546 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 547 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 548 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 549 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 550 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 551 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 552 ]
I'm in cluster: 0
===========================================
[ Entry 553 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 554 ]
===========================================
[ Entry 555 ]
===========================================
[ Entry 556 ]
I'm in cluster: 0
===========================================
[ Entry 557 ]
I'm in cluster: 0
===========================================
[ Entry 558 ]
I'm in cluster: 0
===========================================
[ Entry 559 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 560 ]
I'm in cluster: 0
===========================================
[ Entry 561 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 562 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 563 ]
===========================================
[ Entry 564 ]
I'm in cluster: 0
===========================================
[ Entry 565 ]
I'm in cluster: 0
===========================================
[ Entry 566 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 567 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 568 ]
I'm in cluster: 0
===========================================
[ Entry 569 ]
I'm in cluster: 0
===========================================
[ Entry 570 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 571 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 572 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 573 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 574 ]
I'm in cluster: 0
===========================================
[ Entry 575 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 576 ]
===========================================
[ Entry 577 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 578 ]
I'm in cluster: 0
===========================================
[ Entry 579 ]
===========================================
[ Entry 580 ]
===========================================
[ Entry 581 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 582 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 583 ]
I'm in cluster: 0
===========================================
[ Entry 584 ]
I'm in cluster: 0
===========================================
[ Entry 585 ]
I'm in cluster: 0
===========================================
[ Entry 586 ]
===========================================
[ Entry 587 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 588 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 589 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 590 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 591 ]
===========================================
[ Entry 592 ]
I'm in cluster: 0
===========================================
[ Entry 593 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 594 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 595 ]
===========================================
[ Entry 596 ]
I'm in cluster: 0
===========================================
[ Entry 597 ]
I'm in cluster: 0
===========================================
[ Entry 598 ]
I'm in cluster: 0
===========================================
[ Entry 599 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 600 ]
===========================================
[ Entry 601 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 602 ]
I'm in cluster: 0
===========================================
[ Entry 603 ]
===========================================
[ Entry 604 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 605 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 606 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 607 ]
I'm in cluster: 0
===========================================
[ Entry 608 ]
I'm in cluster: 0
===========================================
[ Entry 609 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 610 ]
I'm in cluster: 0
===========================================
[ Entry 611 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 612 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 613 ]
I'm in cluster: 0
===========================================
[ Entry 614 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 615 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 616 ]
===========================================
[ Entry 617 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 618 ]
===========================================
[ Entry 619 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 620 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 621 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 622 ]
I'm in cluster: 0
===========================================
[ Entry 623 ]
===========================================
[ Entry 624 ]
===========================================
[ Entry 625 ]
I'm in cluster: 0
===========================================
[ Entry 626 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 627 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 628 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 629 ]
===========================================
[ Entry 630 ]
===========================================
[ Entry 631 ]
===========================================
[ Entry 632 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 633 ]
I'm in cluster: 0
===========================================
[ Entry 634 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 635 ]
I'm in cluster: 0
===========================================
[ Entry 636 ]
===========================================
[ Entry 637 ]
I'm in cluster: 0
===========================================
[ Entry 638 ]
I'm in cluster: 0
===========================================
[ Entry 639 ]
===========================================
[ Entry 640 ]
I'm in cluster: 0
===========================================
[ Entry 641 ]
===========================================
[ Entry 642 ]
I'm in cluster: 0
===========================================
[ Entry 643 ]
===========================================
[ Entry 644 ]
===========================================
[ Entry 645 ]
===========================================
[ Entry 646 ]
===========================================
[ Entry 647 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 648 ]
I'm in cluster: 0
===========================================
[ Entry 649 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 650 ]
===========================================
[ Entry 651 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 652 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 653 ]
I'm in cluster: 0
===========================================
[ Entry 654 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 655 ]
===========================================
[ Entry 656 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 657 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 658 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 659 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 660 ]
===========================================
[ Entry 661 ]
I'm in cluster: 0
===========================================
[ Entry 662 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 663 ]
I'm in cluster: 0
===========================================
[ Entry 664 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 665 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 666 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 667 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 668 ]
I'm in cluster: 0
===========================================
[ Entry 669 ]
===========================================
[ Entry 670 ]
===========================================
[ Entry 671 ]
I'm in cluster: 0
===========================================
[ Entry 672 ]
I'm in cluster: 0
===========================================
[ Entry 673 ]
I'm in cluster: 0
===========================================
[ Entry 674 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 675 ]
I'm in cluster: 0
===========================================
[ Entry 676 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 677 ]
===========================================
[ Entry 678 ]
I'm in cluster: 0
===========================================
[ Entry 679 ]
I'm in cluster: 0
===========================================
[ Entry 680 ]
I'm in cluster: 0
===========================================
[ Entry 681 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 682 ]
===========================================
[ Entry 683 ]
I'm in cluster: 0
===========================================
[ Entry 684 ]
I'm in cluster: 0
===========================================
[ Entry 685 ]
===========================================
[ Entry 686 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 687 ]
I'm in cluster: 0
===========================================
[ Entry 688 ]
I'm in cluster: 0
===========================================
[ Entry 689 ]
===========================================
[ Entry 690 ]
===========================================
[ Entry 691 ]
I'm in cluster: 0
===========================================
[ Entry 692 ]
I'm in cluster: 0
===========================================
[ Entry 693 ]
I'm in cluster: 0
===========================================
[ Entry 694 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 695 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 696 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 697 ]
I'm in cluster: 0
===========================================
[ Entry 698 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 699 ]
I'm in cluster: 0
===========================================
[ Entry 700 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 701 ]
===========================================
[ Entry 702 ]
===========================================
[ Entry 703 ]
===========================================
[ Entry 704 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 705 ]
I'm in cluster: 0
===========================================
[ Entry 706 ]
I'm in cluster: 0
===========================================
[ Entry 707 ]
I'm in cluster: 0
===========================================
[ Entry 708 ]
===========================================
[ Entry 709 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 710 ]
I'm in cluster: 0
===========================================
[ Entry 711 ]
I'm in cluster: 0
===========================================
[ Entry 712 ]
I'm in cluster: 0
===========================================
[ Entry 713 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 714 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 715 ]
===========================================
[ Entry 716 ]
I'm in cluster: 0
===========================================
[ Entry 717 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 718 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 719 ]
I'm in cluster: 0
===========================================
[ Entry 720 ]
===========================================
[ Entry 721 ]
I'm in cluster: 0
===========================================
[ Entry 722 ]
I'm in cluster: 0
===========================================
[ Entry 723 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 724 ]
I'm in cluster: 0
===========================================
[ Entry 725 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 726 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 727 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 728 ]
I'm in cluster: 0
===========================================
[ Entry 729 ]
I'm in cluster: 0
===========================================
[ Entry 730 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 731 ]
===========================================
[ Entry 732 ]
I'm in cluster: 0
===========================================
[ Entry 733 ]
===========================================
[ Entry 734 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 735 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 736 ]
===========================================
[ Entry 737 ]
I'm in cluster: 0
===========================================
[ Entry 738 ]
I'm in cluster: 0
===========================================
[ Entry 739 ]
===========================================
[ Entry 740 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 741 ]
I'm in cluster: 0
===========================================
[ Entry 742 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 743 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 744 ]
I'm in cluster: 0
===========================================
[ Entry 745 ]
I'm in cluster: 0
===========================================
[ Entry 746 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 747 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 748 ]
I'm in cluster: 0
===========================================
[ Entry 749 ]
I'm in cluster: 0
===========================================
[ Entry 750 ]
I'm in cluster: 0
===========================================
[ Entry 751 ]
I'm in cluster: 0
===========================================
[ Entry 752 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 753 ]
I'm in cluster: 0
===========================================
[ Entry 754 ]
I'm in cluster: 0
===========================================
[ Entry 755 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 756 ]
I'm in cluster: 0
===========================================
[ Entry 757 ]
===========================================
[ Entry 758 ]
===========================================
[ Entry 759 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 760 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 761 ]
I'm in cluster: 0
===========================================
[ Entry 762 ]
===========================================
[ Entry 763 ]
I'm in cluster: 0
===========================================
[ Entry 764 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 765 ]
I'm in cluster: 0
===========================================
[ Entry 766 ]
I'm in cluster: 0
===========================================
[ Entry 767 ]
I'm in cluster: 0
===========================================
[ Entry 768 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 769 ]
I'm in cluster: 0
===========================================
[ Entry 770 ]
I'm in cluster: 0
===========================================
[ Entry 771 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 772 ]
===========================================
[ Entry 773 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 774 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 775 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 776 ]
===========================================
[ Entry 777 ]
I'm in cluster: 0
===========================================
[ Entry 778 ]
I'm in cluster: 0
===========================================
[ Entry 779 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 780 ]
I'm in cluster: 0
===========================================
[ Entry 781 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 782 ]
I'm in cluster: 0
===========================================
[ Entry 783 ]
I'm in cluster: 0
===========================================
[ Entry 784 ]
I'm in cluster: 0
===========================================
[ Entry 785 ]
I'm in cluster: 0
===========================================
[ Entry 786 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 787 ]
I'm in cluster: 0
===========================================
[ Entry 788 ]
I'm in cluster: 0
===========================================
[ Entry 789 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 790 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
I'm in cluster: 5
===========================================
[ Entry 791 ]
===========================================
[ Entry 792 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 793 ]
I'm in cluster: 0
===========================================
[ Entry 794 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 795 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 796 ]
I'm in cluster: 0
===========================================
[ Entry 797 ]
===========================================
[ Entry 798 ]
I'm in cluster: 0
===========================================
[ Entry 799 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 800 ]
===========================================
[ Entry 801 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 802 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 803 ]
I'm in cluster: 0
===========================================
[ Entry 804 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 805 ]
I'm in cluster: 0
===========================================
[ Entry 806 ]
I'm in cluster: 0
===========================================
[ Entry 807 ]
===========================================
[ Entry 808 ]
I'm in cluster: 0
===========================================
[ Entry 809 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 810 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 811 ]
===========================================
[ Entry 812 ]
===========================================
[ Entry 813 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 814 ]
===========================================
[ Entry 815 ]
I'm in cluster: 0
===========================================
[ Entry 816 ]
===========================================
[ Entry 817 ]
I'm in cluster: 0
===========================================
[ Entry 818 ]
I'm in cluster: 0
===========================================
[ Entry 819 ]
I'm in cluster: 0
===========================================
[ Entry 820 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 821 ]
I'm in cluster: 0
===========================================
[ Entry 822 ]
I'm in cluster: 0
===========================================
[ Entry 823 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 824 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 825 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 826 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 827 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 828 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 829 ]
I'm in cluster: 0
===========================================
[ Entry 830 ]
I'm in cluster: 0
===========================================
[ Entry 831 ]
I'm in cluster: 0
===========================================
[ Entry 832 ]
I'm in cluster: 0
===========================================
[ Entry 833 ]
===========================================
[ Entry 834 ]
===========================================
[ Entry 835 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 836 ]
I'm in cluster: 0
===========================================
[ Entry 837 ]
I'm in cluster: 0
===========================================
[ Entry 838 ]
===========================================
[ Entry 839 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 840 ]
===========================================
[ Entry 841 ]
I'm in cluster: 0
===========================================
[ Entry 842 ]
I'm in cluster: 0
===========================================
[ Entry 843 ]
I'm in cluster: 0
===========================================
[ Entry 844 ]
===========================================
[ Entry 845 ]
===========================================
[ Entry 846 ]
I'm in cluster: 0
===========================================
[ Entry 847 ]
===========================================
[ Entry 848 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 849 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 850 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 851 ]
I'm in cluster: 0
===========================================
[ Entry 852 ]
I'm in cluster: 0
===========================================
[ Entry 853 ]
I'm in cluster: 0
===========================================
[ Entry 854 ]
===========================================
[ Entry 855 ]
===========================================
[ Entry 856 ]
I'm in cluster: 0
===========================================
[ Entry 857 ]
I'm in cluster: 0
===========================================
[ Entry 858 ]
I'm in cluster: 0
===========================================
[ Entry 859 ]
I'm in cluster: 0
===========================================
[ Entry 860 ]
===========================================
[ Entry 861 ]
I'm in cluster: 0
===========================================
[ Entry 862 ]
===========================================
[ Entry 863 ]
===========================================
[ Entry 864 ]
I'm in cluster: 0
===========================================
[ Entry 865 ]
I'm in cluster: 0
===========================================
[ Entry 866 ]
I'm in cluster: 0
===========================================
[ Entry 867 ]
===========================================
[ Entry 868 ]
I'm in cluster: 0
===========================================
[ Entry 869 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 870 ]
===========================================
[ Entry 871 ]
I'm in cluster: 0
===========================================
[ Entry 872 ]
I'm in cluster: 0
===========================================
[ Entry 873 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 874 ]
I'm in cluster: 0
===========================================
[ Entry 875 ]
===========================================
[ Entry 876 ]
I'm in cluster: 0
===========================================
[ Entry 877 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 878 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 879 ]
I'm in cluster: 0
===========================================
[ Entry 880 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 881 ]
I'm in cluster: 0
===========================================
[ Entry 882 ]
I'm in cluster: 0
===========================================
[ Entry 883 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 884 ]
I'm in cluster: 0
===========================================
[ Entry 885 ]
===========================================
[ Entry 886 ]
I'm in cluster: 0
===========================================
[ Entry 887 ]
===========================================
[ Entry 888 ]
I'm in cluster: 0
===========================================
[ Entry 889 ]
===========================================
[ Entry 890 ]
I'm in cluster: 0
===========================================
[ Entry 891 ]
I'm in cluster: 0
===========================================
[ Entry 892 ]
===========================================
[ Entry 893 ]
===========================================
[ Entry 894 ]
I'm in cluster: 0
===========================================
[ Entry 895 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 896 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 897 ]
===========================================
[ Entry 898 ]
I'm in cluster: 0
===========================================
[ Entry 899 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 900 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 901 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 902 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 903 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 904 ]
I'm in cluster: 0
===========================================
[ Entry 905 ]
===========================================
[ Entry 906 ]
===========================================
[ Entry 907 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 908 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 909 ]
I'm in cluster: 0
===========================================
[ Entry 910 ]
===========================================
[ Entry 911 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 912 ]
I'm in cluster: 0
===========================================
[ Entry 913 ]
===========================================
[ Entry 914 ]
I'm in cluster: 0
===========================================
[ Entry 915 ]
I'm in cluster: 0
===========================================
[ Entry 916 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 917 ]
I'm in cluster: 0
===========================================
[ Entry 918 ]
I'm in cluster: 0
===========================================
[ Entry 919 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 920 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 921 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 922 ]
===========================================
[ Entry 923 ]
===========================================
[ Entry 924 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 925 ]
I'm in cluster: 0
===========================================
[ Entry 926 ]
===========================================
[ Entry 927 ]
I'm in cluster: 0
===========================================
[ Entry 928 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 929 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 930 ]
===========================================
[ Entry 931 ]
I'm in cluster: 0
===========================================
[ Entry 932 ]
I'm in cluster: 0
===========================================
[ Entry 933 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 934 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 935 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 936 ]
I'm in cluster: 0
===========================================
[ Entry 937 ]
===========================================
[ Entry 938 ]
I'm in cluster: 0
===========================================
[ Entry 939 ]
===========================================
[ Entry 940 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 941 ]
===========================================
[ Entry 942 ]
===========================================
[ Entry 943 ]
===========================================
[ Entry 944 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 945 ]
===========================================
[ Entry 946 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 947 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 948 ]
===========================================
[ Entry 949 ]
I'm in cluster: 0
===========================================
[ Entry 950 ]
I'm in cluster: 0
===========================================
[ Entry 951 ]
===========================================
[ Entry 952 ]
===========================================
[ Entry 953 ]
===========================================
[ Entry 954 ]
I'm in cluster: 0
===========================================
[ Entry 955 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 956 ]
I'm in cluster: 0
===========================================
[ Entry 957 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 958 ]
I'm in cluster: 0
===========================================
[ Entry 959 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 960 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 961 ]
I'm in cluster: 0
===========================================
[ Entry 962 ]
I'm in cluster: 0
===========================================
[ Entry 963 ]
I'm in cluster: 0
===========================================
[ Entry 964 ]
===========================================
[ Entry 965 ]
I'm in cluster: 0
===========================================
[ Entry 966 ]
===========================================
[ Entry 967 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 968 ]
===========================================
[ Entry 969 ]
I'm in cluster: 0
===========================================
[ Entry 970 ]
I'm in cluster: 0
===========================================
[ Entry 971 ]
I'm in cluster: 0
===========================================
[ Entry 972 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 973 ]
I'm in cluster: 0
===========================================
[ Entry 974 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 975 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 976 ]
I'm in cluster: 0
===========================================
[ Entry 977 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 978 ]
I'm in cluster: 0
===========================================
[ Entry 979 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 980 ]
===========================================
[ Entry 981 ]
I'm in cluster: 0
===========================================
[ Entry 982 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 983 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 984 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 985 ]
I'm in cluster: 0
===========================================
[ Entry 986 ]
I'm in cluster: 0
===========================================
[ Entry 987 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 988 ]
I'm in cluster: 0
===========================================
[ Entry 989 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 990 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 991 ]
I'm in cluster: 0
===========================================
[ Entry 992 ]
I'm in cluster: 0
===========================================
[ Entry 993 ]
I'm in cluster: 0
===========================================
[ Entry 994 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 995 ]
I'm in cluster: 0
===========================================
[ Entry 996 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 997 ]
===========================================
[ Entry 998 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 999 ]
I'm in cluster: 0
===========================================
[ Entry 1000 ]
I'm in cluster: 0
===========================================
[ Entry 1001 ]
I'm in cluster: 0
===========================================
[ Entry 1002 ]
===========================================
[ Entry 1003 ]
===========================================
[ Entry 1004 ]
I'm in cluster: 0
===========================================
[ Entry 1005 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1006 ]
===========================================
[ Entry 1007 ]
===========================================
[ Entry 1008 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1009 ]
I'm in cluster: 0
===========================================
[ Entry 1010 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1011 ]
===========================================
[ Entry 1012 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1013 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1014 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
I'm in cluster: 5
===========================================
[ Entry 1015 ]
I'm in cluster: 0
===========================================
[ Entry 1016 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1017 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1018 ]
===========================================
[ Entry 1019 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1020 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1021 ]
I'm in cluster: 0
===========================================
[ Entry 1022 ]
===========================================
[ Entry 1023 ]
===========================================
[ Entry 1024 ]
===========================================
[ Entry 1025 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1026 ]
===========================================
[ Entry 1027 ]
I'm in cluster: 0
===========================================
[ Entry 1028 ]
I'm in cluster: 0
===========================================
[ Entry 1029 ]
I'm in cluster: 0
===========================================
[ Entry 1030 ]
===========================================
[ Entry 1031 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1032 ]
I'm in cluster: 0
===========================================
[ Entry 1033 ]
===========================================
[ Entry 1034 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1035 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1036 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1037 ]
I'm in cluster: 0
===========================================
[ Entry 1038 ]
I'm in cluster: 0
===========================================
[ Entry 1039 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1040 ]
I'm in cluster: 0
===========================================
[ Entry 1041 ]
===========================================
[ Entry 1042 ]
===========================================
[ Entry 1043 ]
===========================================
[ Entry 1044 ]
===========================================
[ Entry 1045 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1046 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1047 ]
===========================================
[ Entry 1048 ]
===========================================
[ Entry 1049 ]
===========================================
[ Entry 1050 ]
===========================================
[ Entry 1051 ]
===========================================
[ Entry 1052 ]
===========================================
[ Entry 1053 ]
===========================================
[ Entry 1054 ]
I'm in cluster: 0
===========================================
[ Entry 1055 ]
===========================================
[ Entry 1056 ]
===========================================
[ Entry 1057 ]
===========================================
[ Entry 1058 ]
===========================================
[ Entry 1059 ]
===========================================
[ Entry 1060 ]
===========================================
[ Entry 1061 ]
===========================================
[ Entry 1062 ]
===========================================
[ Entry 1063 ]
===========================================
[ Entry 1064 ]
===========================================
[ Entry 1065 ]
I'm in cluster: 0
===========================================
[ Entry 1066 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1067 ]
I'm in cluster: 0
===========================================
[ Entry 1068 ]
I'm in cluster: 0
===========================================
[ Entry 1069 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1070 ]
I'm in cluster: 0
===========================================
[ Entry 1071 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1072 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1073 ]
===========================================
[ Entry 1074 ]
I'm in cluster: 0
===========================================
[ Entry 1075 ]
===========================================
[ Entry 1076 ]
I'm in cluster: 0
===========================================
[ Entry 1077 ]
===========================================
[ Entry 1078 ]
===========================================
[ Entry 1079 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1080 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1081 ]
===========================================
[ Entry 1082 ]
===========================================
[ Entry 1083 ]
I'm in cluster: 0
===========================================
[ Entry 1084 ]
===========================================
[ Entry 1085 ]
I'm in cluster: 0
===========================================
[ Entry 1086 ]
===========================================
[ Entry 1087 ]
I'm in cluster: 0
===========================================
[ Entry 1088 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1089 ]
===========================================
[ Entry 1090 ]
===========================================
[ Entry 1091 ]
I'm in cluster: 0
===========================================
[ Entry 1092 ]
===========================================
[ Entry 1093 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1094 ]
I'm in cluster: 0
===========================================
[ Entry 1095 ]
I'm in cluster: 0
===========================================
[ Entry 1096 ]
===========================================
[ Entry 1097 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1098 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1099 ]
===========================================
[ Entry 1100 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1101 ]
===========================================
[ Entry 1102 ]
I'm in cluster: 0
===========================================
[ Entry 1103 ]
I'm in cluster: 0
===========================================
[ Entry 1104 ]
===========================================
[ Entry 1105 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1106 ]
I'm in cluster: 0
===========================================
[ Entry 1107 ]
I'm in cluster: 0
===========================================
[ Entry 1108 ]
I'm in cluster: 0
===========================================
[ Entry 1109 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1110 ]
I'm in cluster: 0
===========================================
[ Entry 1111 ]
I'm in cluster: 0
===========================================
[ Entry 1112 ]
===========================================
[ Entry 1113 ]
I'm in cluster: 0
===========================================
[ Entry 1114 ]
===========================================
[ Entry 1115 ]
I'm in cluster: 0
===========================================
[ Entry 1116 ]
===========================================
[ Entry 1117 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1118 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1119 ]
I'm in cluster: 0
===========================================
[ Entry 1120 ]
I'm in cluster: 0
===========================================
[ Entry 1121 ]
===========================================
[ Entry 1122 ]
===========================================
[ Entry 1123 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1124 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1125 ]
===========================================
[ Entry 1126 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1127 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1128 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1129 ]
I'm in cluster: 0
===========================================
[ Entry 1130 ]
===========================================
[ Entry 1131 ]
I'm in cluster: 0
===========================================
[ Entry 1132 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1133 ]
===========================================
[ Entry 1134 ]
I'm in cluster: 0
===========================================
[ Entry 1135 ]
===========================================
[ Entry 1136 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1137 ]
===========================================
[ Entry 1138 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1139 ]
===========================================
[ Entry 1140 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1141 ]
I'm in cluster: 0
===========================================
[ Entry 1142 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1143 ]
I'm in cluster: 0
===========================================
[ Entry 1144 ]
I'm in cluster: 0
===========================================
[ Entry 1145 ]
I'm in cluster: 0
===========================================
[ Entry 1146 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1147 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1148 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1149 ]
===========================================
[ Entry 1150 ]
I'm in cluster: 0
===========================================
[ Entry 1151 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1152 ]
I'm in cluster: 0
===========================================
[ Entry 1153 ]
===========================================
[ Entry 1154 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1155 ]
===========================================
[ Entry 1156 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1157 ]
===========================================
[ Entry 1158 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1159 ]
I'm in cluster: 0
===========================================
[ Entry 1160 ]
I'm in cluster: 0
===========================================
[ Entry 1161 ]
===========================================
[ Entry 1162 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1163 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1164 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1165 ]
===========================================
[ Entry 1166 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1167 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1168 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1169 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1170 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1171 ]
I'm in cluster: 0
===========================================
[ Entry 1172 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1173 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1174 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1175 ]
===========================================
[ Entry 1176 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1177 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1178 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1179 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1180 ]
===========================================
[ Entry 1181 ]
===========================================
[ Entry 1182 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1183 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1184 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1185 ]
I'm in cluster: 0
===========================================
[ Entry 1186 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1187 ]
I'm in cluster: 0
===========================================
[ Entry 1188 ]
I'm in cluster: 0
===========================================
[ Entry 1189 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1190 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1191 ]
===========================================
[ Entry 1192 ]
I'm in cluster: 0
===========================================
[ Entry 1193 ]
===========================================
[ Entry 1194 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1195 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1196 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1197 ]
===========================================
[ Entry 1198 ]
I'm in cluster: 0
===========================================
[ Entry 1199 ]
===========================================
[ Entry 1200 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1201 ]
I'm in cluster: 0
===========================================
[ Entry 1202 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
I'm in cluster: 5
===========================================
[ Entry 1203 ]
I'm in cluster: 0
===========================================
[ Entry 1204 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1205 ]
===========================================
[ Entry 1206 ]
I'm in cluster: 0
===========================================
[ Entry 1207 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1208 ]
I'm in cluster: 0
===========================================
[ Entry 1209 ]
===========================================
[ Entry 1210 ]
===========================================
[ Entry 1211 ]
I'm in cluster: 0
===========================================
[ Entry 1212 ]
I'm in cluster: 0
===========================================
[ Entry 1213 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1214 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1215 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1216 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1217 ]
===========================================
[ Entry 1218 ]
===========================================
[ Entry 1219 ]
I'm in cluster: 0
===========================================
[ Entry 1220 ]
===========================================
[ Entry 1221 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1222 ]
I'm in cluster: 0
===========================================
[ Entry 1223 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1224 ]
I'm in cluster: 0
===========================================
[ Entry 1225 ]
===========================================
[ Entry 1226 ]
I'm in cluster: 0
===========================================
[ Entry 1227 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1228 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1229 ]
===========================================
[ Entry 1230 ]
I'm in cluster: 0
===========================================
[ Entry 1231 ]
I'm in cluster: 0
===========================================
[ Entry 1232 ]
===========================================
[ Entry 1233 ]
===========================================
[ Entry 1234 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1235 ]
===========================================
[ Entry 1236 ]
===========================================
[ Entry 1237 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1238 ]
===========================================
[ Entry 1239 ]
I'm in cluster: 0
===========================================
[ Entry 1240 ]
===========================================
[ Entry 1241 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1242 ]
===========================================
[ Entry 1243 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1244 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1245 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1246 ]
===========================================
[ Entry 1247 ]
===========================================
[ Entry 1248 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 1249 ]
I'm in cluster: 0
===========================================
[ Entry 1250 ]
I'm in cluster: 0
===========================================
[ Entry 1251 ]
I'm in cluster: 0
===========================================
[ Entry 1252 ]
===========================================
[ Entry 1253 ]
===========================================
[ Entry 1254 ]
I'm in cluster: 0
===========================================
[ Entry 1255 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1256 ]
===========================================
[ Entry 1257 ]
===========================================
[ Entry 1258 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1259 ]
I'm in cluster: 0
===========================================
[ Entry 1260 ]
===========================================
[ Entry 1261 ]
I'm in cluster: 0
===========================================
[ Entry 1262 ]
I'm in cluster: 0
===========================================
[ Entry 1263 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1264 ]
===========================================
[ Entry 1265 ]
===========================================
[ Entry 1266 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1267 ]
===========================================
[ Entry 1268 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1269 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1270 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1271 ]
I'm in cluster: 0
===========================================
[ Entry 1272 ]
I'm in cluster: 0
===========================================
[ Entry 1273 ]
I'm in cluster: 0
===========================================
[ Entry 1274 ]
I'm in cluster: 0
===========================================
[ Entry 1275 ]
I'm in cluster: 0
===========================================
[ Entry 1276 ]
===========================================
[ Entry 1277 ]
===========================================
[ Entry 1278 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1279 ]
I'm in cluster: 0
===========================================
[ Entry 1280 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1281 ]
I'm in cluster: 0
===========================================
[ Entry 1282 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1283 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1284 ]
===========================================
[ Entry 1285 ]
===========================================
[ Entry 1286 ]
===========================================
[ Entry 1287 ]
===========================================
[ Entry 1288 ]
===========================================
[ Entry 1289 ]
I'm in cluster: 0
===========================================
[ Entry 1290 ]
===========================================
[ Entry 1291 ]
===========================================
[ Entry 1292 ]
===========================================
[ Entry 1293 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 1294 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1295 ]
I'm in cluster: 0
===========================================
[ Entry 1296 ]
===========================================
[ Entry 1297 ]
I'm in cluster: 0
===========================================
[ Entry 1298 ]
I'm in cluster: 0
===========================================
[ Entry 1299 ]
I'm in cluster: 0
===========================================
[ Entry 1300 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1301 ]
I'm in cluster: 0
===========================================
[ Entry 1302 ]
I'm in cluster: 0
===========================================
[ Entry 1303 ]
I'm in cluster: 0
===========================================
[ Entry 1304 ]
I'm in cluster: 0
===========================================
[ Entry 1305 ]
===========================================
[ Entry 1306 ]
I'm in cluster: 0
===========================================
[ Entry 1307 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1308 ]
I'm in cluster: 0
===========================================
[ Entry 1309 ]
===========================================
[ Entry 1310 ]
===========================================
[ Entry 1311 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1312 ]
I'm in cluster: 0
===========================================
[ Entry 1313 ]
===========================================
[ Entry 1314 ]
===========================================
[ Entry 1315 ]
===========================================
[ Entry 1316 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1317 ]
===========================================
[ Entry 1318 ]
I'm in cluster: 0
abratenko@trex:~/ubdl/larflow/larflow/Ana$ root -l crt_0-1318
crt_0-1318_10cmVoxels.root  crt_0-1318_1cmVoxels.root   crt_0-1318_3cmVoxels.root   crt_0-1318_5cmVoxels.root   crt_0-1318.root
abratenko@trex:~/ubdl/larflow/larflow/Ana$ root -l crt_0-1318.root 
root [0] 
Attaching file crt_0-1318.root as _file0...
(TFile *) 0x55cf45cabc10
root [1] .ls
TFile**		crt_0-1318.root	
 TFile*		crt_0-1318.root	
  KEY: TTree	tree;1	tree of hits per voxel
  KEY: TH1D	hitcount_wire_hist_U;1	wire #
  KEY: TH1D	hitcount_wire_hist_V;1	wire #
  KEY: TH1D	hitcount_wire_hist_Y;1	wire #
  KEY: TH2D	hitcount_wire_th2d_U;1	wire ; tick
  KEY: TH2D	hitcount_wire_th2d_V;1	wire ; tick
  KEY: TH2D	hitcount_wire_th2d_Y;1	wire ; tick
  KEY: TH1D	hitcount_xyz_hist_x;1	
  KEY: TH1D	hitcount_xyz_hist_y;1	
  KEY: TH1D	hitcount_xyz_hist_z;1	
  KEY: TH2D	hitcount_xyz_th2d_xy;1	
  KEY: TH2D	hitcount_xyz_th2d_zy;1	
  KEY: TH2D	hitcount_xyz_th2d_zx;1	
  KEY: TH3D	hitcount_xyz_th3d;1	
root [2] hitcount_xyz_th3d->Draw("hitsPerVoxel")
Info in <TCanvas::MakeDefCanvas>:  created default TCanvas with name c1
root [3] tree->Draw("hitsPerVoxel")
Info in <TCanvas::MakeDefCanvas>:  created default TCanvas with name c1
root [4] tree->Draw("hitsPerVoxel>>htemp(100, 0, 100)")
root [5] .q
abratenko@trex:~/ubdl/larflow/larflow/Ana$ e CRTana.cxx
abratenko@trex:~/ubdl/larflow/larflow/Ana$ make
make: Nothing to be done for 'all'.
abratenko@trex:~/ubdl/larflow/larflow/Ana$ e CRTana.cxx
abratenko@trex:~/ubdl/larflow/larflow/Ana$ ls
crt_0-0.root                crt_0-1318_3cmVoxels.root  crt_0-29.root  CRTana       CRTvoxelHits.py   hitsPerVoxel.root      kpsreco_vertexana
crt_0-1318_10cmVoxels.root  crt_0-1318_5cmVoxels.root  crt_0-2.root   CRTana.cxx   CRTvoxelHits.py~  keypoint_recoana.cxx   kpsreco_vertexana.cxx
crt_0-1318_1cmVoxels.root   crt_0-1318.root            crt_0-9.root   CRTana.cxx~  GNUmakefile       keypoint_truthana.cxx  README.md
abratenko@trex:~/ubdl/larflow/larflow/Ana$ mkdir oldBadPlots
abratenko@trex:~/ubdl/larflow/larflow/Ana$ mv crt_0-1318_
crt_0-1318_10cmVoxels.root  crt_0-1318_1cmVoxels.root   crt_0-1318_3cmVoxels.root   crt_0-1318_5cmVoxels.root   
abratenko@trex:~/ubdl/larflow/larflow/Ana$ mv crt_0-1318_
crt_0-1318_10cmVoxels.root  crt_0-1318_1cmVoxels.root   crt_0-1318_3cmVoxels.root   crt_0-1318_5cmVoxels.root   
abratenko@trex:~/ubdl/larflow/larflow/Ana$ mv crt_0-1318_* oldBadPlots/
abratenko@trex:~/ubdl/larflow/larflow/Ana$ ls
crt_0-0.root     crt_0-29.root  crt_0-9.root  CRTana.cxx   CRTvoxelHits.py   GNUmakefile        keypoint_recoana.cxx   kpsreco_vertexana      oldBadPlots
crt_0-1318.root  crt_0-2.root   CRTana        CRTana.cxx~  CRTvoxelHits.py~  hitsPerVoxel.root  keypoint_truthana.cxx  kpsreco_vertexana.cxx  README.md
abratenko@trex:~/ubdl/larflow/larflow/Ana$ ls oldBadPlots/
crt_0-1318_10cmVoxels.root  crt_0-1318_1cmVoxels.root  crt_0-1318_3cmVoxels.root  crt_0-1318_5cmVoxels.root
abratenko@trex:~/ubdl/larflow/larflow/Ana$ ls -lrt
total 15864
-rw-rw-r-- 1 abratenko abratenko     217 Aug  4 17:45 README.md
-rw-rw-r-- 1 abratenko abratenko   13937 Aug  4 17:45 kpsreco_vertexana.cxx
-rw-rw-r-- 1 abratenko abratenko    3533 Aug  4 17:45 keypoint_truthana.cxx
-rw-rw-r-- 1 abratenko abratenko   12714 Aug  4 17:45 keypoint_recoana.cxx
-rwxrwxr-x 1 abratenko abratenko 1476240 Aug  7 13:55 kpsreco_vertexana
-rw-r--r-- 1 abratenko abratenko     411 Aug  7 14:37 crt_0-0.root
-rw-r--r-- 1 abratenko abratenko     411 Aug 11 12:48 crt_0-2.root
-rw-r--r-- 1 abratenko abratenko  598080 Aug 13 13:10 crt_0-29.root
-rw-rw-r-- 1 abratenko abratenko    5089 Aug 17 15:27 CRTana.cxx~
-rw-rw-r-- 1 abratenko abratenko     766 Aug 17 15:49 GNUmakefile
-rw-rw-r-- 1 abratenko abratenko    1125 Aug 17 17:09 CRTvoxelHits.py~
-rw-rw-r-- 1 abratenko abratenko    1125 Aug 17 17:10 CRTvoxelHits.py
-rw-r--r-- 1 abratenko abratenko    5308 Aug 17 17:10 hitsPerVoxel.root
-rw-r--r-- 1 abratenko abratenko 6025901 Aug 17 17:16 crt_0-9.root
-rw-rw-r-- 1 abratenko abratenko    5266 Aug 21 10:54 CRTana.cxx
-rwxrwxr-x 1 abratenko abratenko  924328 Aug 21 10:54 CRTana
-rw-r--r-- 1 abratenko abratenko 7115474 Aug 21 10:55 crt_0-1318.root
drwxrwxr-x 2 abratenko abratenko    4096 Aug 21 11:09 oldBadPlots
abratenko@trex:~/ubdl/larflow/larflow/Ana$ mv crt_0-1318.root crt_0-1318_10cmVoxels.root
abratenko@trex:~/ubdl/larflow/larflow/Ana$ root -l crt_0-1318_10cmVoxels.root
root [0] 
Attaching file crt_0-1318_10cmVoxels.root as _file0...
(TFile *) 0x5651820f4240
root [1] .ls
TFile**		crt_0-1318_10cmVoxels.root	
 TFile*		crt_0-1318_10cmVoxels.root	
  KEY: TTree	tree;1	tree of hits per voxel
  KEY: TH1D	hitcount_wire_hist_U;1	wire #
  KEY: TH1D	hitcount_wire_hist_V;1	wire #
  KEY: TH1D	hitcount_wire_hist_Y;1	wire #
  KEY: TH2D	hitcount_wire_th2d_U;1	wire ; tick
  KEY: TH2D	hitcount_wire_th2d_V;1	wire ; tick
  KEY: TH2D	hitcount_wire_th2d_Y;1	wire ; tick
  KEY: TH1D	hitcount_xyz_hist_x;1	
  KEY: TH1D	hitcount_xyz_hist_y;1	
  KEY: TH1D	hitcount_xyz_hist_z;1	
  KEY: TH2D	hitcount_xyz_th2d_xy;1	
  KEY: TH2D	hitcount_xyz_th2d_zy;1	
  KEY: TH2D	hitcount_xyz_th2d_zx;1	
  KEY: TH3D	hitcount_xyz_th3d;1	
root [2] tree->Draw("hitsPerVoxel")
Info in <TCanvas::MakeDefCanvas>:  created default TCanvas with name c1
root [3] tree->Draw("hitsPerVoxel>>htemp(100, 0, 100)")
Info in <TCanvas::MakeDefCanvas>:  created default TCanvas with name c1
root [4] .q
abratenko@trex:~/ubdl/larflow/larflow/Ana$ ls
crt_0-0.root                crt_0-2.root  CRTana.cxx       CRTvoxelHits.py~   keypoint_recoana.cxx   kpsreco_vertexana.cxx
crt_0-1318_10cmVoxels.root  crt_0-9.root  CRTana.cxx~      GNUmakefile        keypoint_truthana.cxx  oldBadPlots
crt_0-29.root               CRTana        CRTvoxelHits.py  hitsPerVoxel.root  kpsreco_vertexana      README.md
abratenko@trex:~/ubdl/larflow/larflow/Ana$ e CRTana.cxx
abratenko@trex:~/ubdl/larflow/larflow/Ana$ make
<< compile CRTana >>
g++ -g -fPIC `root-config --cflags` `larlite-config --includes` -I/home/abratenko/ubdl/larlite/../ `larcv-config --includes` `ublarcvapp-config --includes` -I/home/abratenko/ubdl/larflow/build/include  CRTana.cxx -o CRTana -L/home/abratenko/ubdl/larflow/build/lib -lLArFlow_LArFlowConstants -lLArFlow_PrepFlowMatchData -lLArFlow_KeyPoints `ublarcvapp-config --libs` -lLArCVApp_MCTools -lLArCVApp_ubdllee -lLArCVApp_UBWireTool -lLArCVApp_LArliteHandler `larcv-config --libs` -lLArCVCorePyUtil `larlite-config --libs` `root-config --libs`
abratenko@trex:~/ubdl/larflow/larflow/Ana$ ./CRTana ~/data/hadded_crtFiles.root 0 1319
    [NORMAL]  <open> Opening a file in READ mode: /home/abratenko/data/hadded_crtFiles.root
===========================================
[ Entry 0 ]
I'm in cluster: 0
===========================================
[ Entry 1 ]
===========================================
[ Entry 2 ]
I'm in cluster: 0
===========================================
[ Entry 3 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 4 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 5 ]
===========================================
[ Entry 6 ]
I'm in cluster: 0
===========================================
[ Entry 7 ]
I'm in cluster: 0
===========================================
[ Entry 8 ]
===========================================
[ Entry 9 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 10 ]
I'm in cluster: 0
===========================================
[ Entry 11 ]
===========================================
[ Entry 12 ]
===========================================
[ Entry 13 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 14 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 15 ]
===========================================
[ Entry 16 ]
I'm in cluster: 0
===========================================
[ Entry 17 ]
I'm in cluster: 0
===========================================
[ Entry 18 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 19 ]
===========================================
[ Entry 20 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 21 ]
I'm in cluster: 0
===========================================
[ Entry 22 ]
===========================================
[ Entry 23 ]
===========================================
[ Entry 24 ]
I'm in cluster: 0
===========================================
[ Entry 25 ]
===========================================
[ Entry 26 ]
I'm in cluster: 0
===========================================
[ Entry 27 ]
I'm in cluster: 0
===========================================
[ Entry 28 ]
I'm in cluster: 0
===========================================
[ Entry 29 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 30 ]
===========================================
[ Entry 31 ]
===========================================
[ Entry 32 ]
I'm in cluster: 0
===========================================
[ Entry 33 ]
===========================================
[ Entry 34 ]
I'm in cluster: 0
===========================================
[ Entry 35 ]
I'm in cluster: 0
===========================================
[ Entry 36 ]
I'm in cluster: 0
===========================================
[ Entry 37 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 38 ]
===========================================
[ Entry 39 ]
===========================================
[ Entry 40 ]
I'm in cluster: 0
===========================================
[ Entry 41 ]
===========================================
[ Entry 42 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 43 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 44 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 45 ]
===========================================
[ Entry 46 ]
===========================================
[ Entry 47 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 48 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 49 ]
I'm in cluster: 0
===========================================
[ Entry 50 ]
===========================================
[ Entry 51 ]
===========================================
[ Entry 52 ]
===========================================
[ Entry 53 ]
===========================================
[ Entry 54 ]
I'm in cluster: 0
===========================================
[ Entry 55 ]
===========================================
[ Entry 56 ]
===========================================
[ Entry 57 ]
I'm in cluster: 0
===========================================
[ Entry 58 ]
I'm in cluster: 0
===========================================
[ Entry 59 ]
I'm in cluster: 0
===========================================
[ Entry 60 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 61 ]
===========================================
[ Entry 62 ]
===========================================
[ Entry 63 ]
===========================================
[ Entry 64 ]
===========================================
[ Entry 65 ]
I'm in cluster: 0
===========================================
[ Entry 66 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 67 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 68 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 69 ]
===========================================
[ Entry 70 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 71 ]
I'm in cluster: 0
===========================================
[ Entry 72 ]
===========================================
[ Entry 73 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 74 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 75 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 76 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 77 ]
===========================================
[ Entry 78 ]
I'm in cluster: 0
===========================================
[ Entry 79 ]
I'm in cluster: 0
===========================================
[ Entry 80 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 81 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 82 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 83 ]
===========================================
[ Entry 84 ]
I'm in cluster: 0
===========================================
[ Entry 85 ]
I'm in cluster: 0
===========================================
[ Entry 86 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 87 ]
===========================================
[ Entry 88 ]
===========================================
[ Entry 89 ]
I'm in cluster: 0
===========================================
[ Entry 90 ]
I'm in cluster: 0
===========================================
[ Entry 91 ]
===========================================
[ Entry 92 ]
I'm in cluster: 0
===========================================
[ Entry 93 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 94 ]
I'm in cluster: 0
===========================================
[ Entry 95 ]
I'm in cluster: 0
===========================================
[ Entry 96 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 97 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 98 ]
I'm in cluster: 0
===========================================
[ Entry 99 ]
===========================================
[ Entry 100 ]
I'm in cluster: 0
===========================================
[ Entry 101 ]
===========================================
[ Entry 102 ]
I'm in cluster: 0
===========================================
[ Entry 103 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 104 ]
I'm in cluster: 0
===========================================
[ Entry 105 ]
I'm in cluster: 0
===========================================
[ Entry 106 ]
===========================================
[ Entry 107 ]
I'm in cluster: 0
===========================================
[ Entry 108 ]
I'm in cluster: 0
===========================================
[ Entry 109 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 110 ]
I'm in cluster: 0
===========================================
[ Entry 111 ]
I'm in cluster: 0
===========================================
[ Entry 112 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 113 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 114 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 115 ]
===========================================
[ Entry 116 ]
I'm in cluster: 0
===========================================
[ Entry 117 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 118 ]
===========================================
[ Entry 119 ]
I'm in cluster: 0
===========================================
[ Entry 120 ]
===========================================
[ Entry 121 ]
===========================================
[ Entry 122 ]
I'm in cluster: 0
===========================================
[ Entry 123 ]
===========================================
[ Entry 124 ]
I'm in cluster: 0
===========================================
[ Entry 125 ]
===========================================
[ Entry 126 ]
===========================================
[ Entry 127 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 128 ]
===========================================
[ Entry 129 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 130 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 131 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 132 ]
I'm in cluster: 0
===========================================
[ Entry 133 ]
I'm in cluster: 0
===========================================
[ Entry 134 ]
I'm in cluster: 0
===========================================
[ Entry 135 ]
===========================================
[ Entry 136 ]
I'm in cluster: 0
===========================================
[ Entry 137 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 138 ]
===========================================
[ Entry 139 ]
I'm in cluster: 0
===========================================
[ Entry 140 ]
===========================================
[ Entry 141 ]
===========================================
[ Entry 142 ]
I'm in cluster: 0
===========================================
[ Entry 143 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 144 ]
I'm in cluster: 0
===========================================
[ Entry 145 ]
I'm in cluster: 0
===========================================
[ Entry 146 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 147 ]
I'm in cluster: 0
===========================================
[ Entry 148 ]
I'm in cluster: 0
===========================================
[ Entry 149 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 150 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 151 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 152 ]
I'm in cluster: 0
===========================================
[ Entry 153 ]
===========================================
[ Entry 154 ]
===========================================
[ Entry 155 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 156 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 157 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 158 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 159 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 160 ]
I'm in cluster: 0
===========================================
[ Entry 161 ]
I'm in cluster: 0
===========================================
[ Entry 162 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 163 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 164 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 165 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
I'm in cluster: 5
===========================================
[ Entry 166 ]
I'm in cluster: 0
===========================================
[ Entry 167 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 168 ]
I'm in cluster: 0
===========================================
[ Entry 169 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 170 ]
===========================================
[ Entry 171 ]
===========================================
[ Entry 172 ]
I'm in cluster: 0
===========================================
[ Entry 173 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 174 ]
I'm in cluster: 0
===========================================
[ Entry 175 ]
I'm in cluster: 0
===========================================
[ Entry 176 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 177 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 178 ]
===========================================
[ Entry 179 ]
===========================================
[ Entry 180 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 181 ]
I'm in cluster: 0
===========================================
[ Entry 182 ]
===========================================
[ Entry 183 ]
I'm in cluster: 0
===========================================
[ Entry 184 ]
I'm in cluster: 0
===========================================
[ Entry 185 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 186 ]
===========================================
[ Entry 187 ]
===========================================
[ Entry 188 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 189 ]
I'm in cluster: 0
===========================================
[ Entry 190 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 191 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 192 ]
===========================================
[ Entry 193 ]
===========================================
[ Entry 194 ]
===========================================
[ Entry 195 ]
I'm in cluster: 0
===========================================
[ Entry 196 ]
I'm in cluster: 0
===========================================
[ Entry 197 ]
I'm in cluster: 0
===========================================
[ Entry 198 ]
I'm in cluster: 0
===========================================
[ Entry 199 ]
I'm in cluster: 0
===========================================
[ Entry 200 ]
I'm in cluster: 0
===========================================
[ Entry 201 ]
I'm in cluster: 0
===========================================
[ Entry 202 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 203 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 204 ]
I'm in cluster: 0
===========================================
[ Entry 205 ]
===========================================
[ Entry 206 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 207 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 208 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 209 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 210 ]
I'm in cluster: 0
===========================================
[ Entry 211 ]
I'm in cluster: 0
===========================================
[ Entry 212 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 213 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 214 ]
===========================================
[ Entry 215 ]
I'm in cluster: 0
===========================================
[ Entry 216 ]
I'm in cluster: 0
===========================================
[ Entry 217 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 218 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 219 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 220 ]
I'm in cluster: 0
===========================================
[ Entry 221 ]
===========================================
[ Entry 222 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 223 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 224 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 225 ]
===========================================
[ Entry 226 ]
I'm in cluster: 0
===========================================
[ Entry 227 ]
I'm in cluster: 0
===========================================
[ Entry 228 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 229 ]
I'm in cluster: 0
===========================================
[ Entry 230 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 231 ]
===========================================
[ Entry 232 ]
I'm in cluster: 0
===========================================
[ Entry 233 ]
===========================================
[ Entry 234 ]
===========================================
[ Entry 235 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 236 ]
===========================================
[ Entry 237 ]
I'm in cluster: 0
===========================================
[ Entry 238 ]
===========================================
[ Entry 239 ]
I'm in cluster: 0
===========================================
[ Entry 240 ]
I'm in cluster: 0
===========================================
[ Entry 241 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 242 ]
===========================================
[ Entry 243 ]
===========================================
[ Entry 244 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 245 ]
===========================================
[ Entry 246 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 247 ]
I'm in cluster: 0
===========================================
[ Entry 248 ]
===========================================
[ Entry 249 ]
===========================================
[ Entry 250 ]
===========================================
[ Entry 251 ]
===========================================
[ Entry 252 ]
===========================================
[ Entry 253 ]
===========================================
[ Entry 254 ]
===========================================
[ Entry 255 ]
===========================================
[ Entry 256 ]
===========================================
[ Entry 257 ]
===========================================
[ Entry 258 ]
===========================================
[ Entry 259 ]
===========================================
[ Entry 260 ]
===========================================
[ Entry 261 ]
===========================================
[ Entry 262 ]
===========================================
[ Entry 263 ]
===========================================
[ Entry 264 ]
===========================================
[ Entry 265 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 266 ]
I'm in cluster: 0
===========================================
[ Entry 267 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 268 ]
===========================================
[ Entry 269 ]
I'm in cluster: 0
===========================================
[ Entry 270 ]
===========================================
[ Entry 271 ]
===========================================
[ Entry 272 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 273 ]
I'm in cluster: 0
===========================================
[ Entry 274 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 275 ]
===========================================
[ Entry 276 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 277 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 278 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 279 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 280 ]
I'm in cluster: 0
===========================================
[ Entry 281 ]
I'm in cluster: 0
===========================================
[ Entry 282 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 283 ]
===========================================
[ Entry 284 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 285 ]
I'm in cluster: 0
===========================================
[ Entry 286 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 287 ]
===========================================
[ Entry 288 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 289 ]
===========================================
[ Entry 290 ]
I'm in cluster: 0
===========================================
[ Entry 291 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 292 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 293 ]
===========================================
[ Entry 294 ]
===========================================
[ Entry 295 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 296 ]
===========================================
[ Entry 297 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 298 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 299 ]
===========================================
[ Entry 300 ]
===========================================
[ Entry 301 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 302 ]
I'm in cluster: 0
===========================================
[ Entry 303 ]
===========================================
[ Entry 304 ]
===========================================
[ Entry 305 ]
I'm in cluster: 0
===========================================
[ Entry 306 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 307 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 308 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 309 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 310 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 311 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 312 ]
===========================================
[ Entry 313 ]
===========================================
[ Entry 314 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 315 ]
===========================================
[ Entry 316 ]
I'm in cluster: 0
===========================================
[ Entry 317 ]
===========================================
[ Entry 318 ]
===========================================
[ Entry 319 ]
I'm in cluster: 0
===========================================
[ Entry 320 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 321 ]
I'm in cluster: 0
===========================================
[ Entry 322 ]
===========================================
[ Entry 323 ]
I'm in cluster: 0
===========================================
[ Entry 324 ]
I'm in cluster: 0
===========================================
[ Entry 325 ]
I'm in cluster: 0
===========================================
[ Entry 326 ]
===========================================
[ Entry 327 ]
I'm in cluster: 0
===========================================
[ Entry 328 ]
I'm in cluster: 0
===========================================
[ Entry 329 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 330 ]
===========================================
[ Entry 331 ]
===========================================
[ Entry 332 ]
===========================================
[ Entry 333 ]
I'm in cluster: 0
===========================================
[ Entry 334 ]
I'm in cluster: 0
===========================================
[ Entry 335 ]
I'm in cluster: 0
===========================================
[ Entry 336 ]
===========================================
[ Entry 337 ]
I'm in cluster: 0
===========================================
[ Entry 338 ]
===========================================
[ Entry 339 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 340 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 341 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 342 ]
I'm in cluster: 0
===========================================
[ Entry 343 ]
I'm in cluster: 0
===========================================
[ Entry 344 ]
I'm in cluster: 0
===========================================
[ Entry 345 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 346 ]
I'm in cluster: 0
===========================================
[ Entry 347 ]
I'm in cluster: 0
===========================================
[ Entry 348 ]
I'm in cluster: 0
===========================================
[ Entry 349 ]
I'm in cluster: 0
===========================================
[ Entry 350 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 351 ]
I'm in cluster: 0
===========================================
[ Entry 352 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 353 ]
I'm in cluster: 0
===========================================
[ Entry 354 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 355 ]
===========================================
[ Entry 356 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 357 ]
I'm in cluster: 0
===========================================
[ Entry 358 ]
===========================================
[ Entry 359 ]
===========================================
[ Entry 360 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 361 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 362 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 363 ]
I'm in cluster: 0
===========================================
[ Entry 364 ]
I'm in cluster: 0
===========================================
[ Entry 365 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 366 ]
===========================================
[ Entry 367 ]
===========================================
[ Entry 368 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 369 ]
I'm in cluster: 0
===========================================
[ Entry 370 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 371 ]
===========================================
[ Entry 372 ]
I'm in cluster: 0
===========================================
[ Entry 373 ]
I'm in cluster: 0
===========================================
[ Entry 374 ]
I'm in cluster: 0
===========================================
[ Entry 375 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 376 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 377 ]
===========================================
[ Entry 378 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 379 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 380 ]
===========================================
[ Entry 381 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 382 ]
I'm in cluster: 0
===========================================
[ Entry 383 ]
I'm in cluster: 0
===========================================
[ Entry 384 ]
I'm in cluster: 0
===========================================
[ Entry 385 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 386 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 387 ]
I'm in cluster: 0
===========================================
[ Entry 388 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 389 ]
I'm in cluster: 0
===========================================
[ Entry 390 ]
I'm in cluster: 0
===========================================
[ Entry 391 ]
I'm in cluster: 0
===========================================
[ Entry 392 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 393 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 394 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 395 ]
I'm in cluster: 0
===========================================
[ Entry 396 ]
I'm in cluster: 0
===========================================
[ Entry 397 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 398 ]
I'm in cluster: 0
===========================================
[ Entry 399 ]
===========================================
[ Entry 400 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 401 ]
===========================================
[ Entry 402 ]
I'm in cluster: 0
===========================================
[ Entry 403 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 404 ]
I'm in cluster: 0
===========================================
[ Entry 405 ]
===========================================
[ Entry 406 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 407 ]
I'm in cluster: 0
===========================================
[ Entry 408 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 409 ]
I'm in cluster: 0
===========================================
[ Entry 410 ]
I'm in cluster: 0
===========================================
[ Entry 411 ]
===========================================
[ Entry 412 ]
===========================================
[ Entry 413 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 414 ]
===========================================
[ Entry 415 ]
===========================================
[ Entry 416 ]
I'm in cluster: 0
===========================================
[ Entry 417 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 418 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 419 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 420 ]
===========================================
[ Entry 421 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 422 ]
===========================================
[ Entry 423 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 424 ]
===========================================
[ Entry 425 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 426 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 427 ]
I'm in cluster: 0
===========================================
[ Entry 428 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 429 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 430 ]
===========================================
[ Entry 431 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 432 ]
I'm in cluster: 0
===========================================
[ Entry 433 ]
I'm in cluster: 0
===========================================
[ Entry 434 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 435 ]
===========================================
[ Entry 436 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 437 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 438 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 439 ]
===========================================
[ Entry 440 ]
===========================================
[ Entry 441 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 442 ]
===========================================
[ Entry 443 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 444 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 445 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 446 ]
===========================================
[ Entry 447 ]
===========================================
[ Entry 448 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 449 ]
===========================================
[ Entry 450 ]
===========================================
[ Entry 451 ]
===========================================
[ Entry 452 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 453 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 454 ]
===========================================
[ Entry 455 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 456 ]
===========================================
[ Entry 457 ]
===========================================
[ Entry 458 ]
I'm in cluster: 0
===========================================
[ Entry 459 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 460 ]
I'm in cluster: 0
===========================================
[ Entry 461 ]
I'm in cluster: 0
===========================================
[ Entry 462 ]
===========================================
[ Entry 463 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 464 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 465 ]
I'm in cluster: 0
===========================================
[ Entry 466 ]
I'm in cluster: 0
===========================================
[ Entry 467 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 468 ]
I'm in cluster: 0
===========================================
[ Entry 469 ]
===========================================
[ Entry 470 ]
I'm in cluster: 0
===========================================
[ Entry 471 ]
I'm in cluster: 0
===========================================
[ Entry 472 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 473 ]
===========================================
[ Entry 474 ]
I'm in cluster: 0
===========================================
[ Entry 475 ]
===========================================
[ Entry 476 ]
I'm in cluster: 0
===========================================
[ Entry 477 ]
I'm in cluster: 0
===========================================
[ Entry 478 ]
===========================================
[ Entry 479 ]
===========================================
[ Entry 480 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 481 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 482 ]
I'm in cluster: 0
===========================================
[ Entry 483 ]
===========================================
[ Entry 484 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 485 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 486 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 487 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 488 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 489 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 490 ]
I'm in cluster: 0
===========================================
[ Entry 491 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 492 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 493 ]
===========================================
[ Entry 494 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 495 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 496 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 497 ]
I'm in cluster: 0
===========================================
[ Entry 498 ]
I'm in cluster: 0
===========================================
[ Entry 499 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 500 ]
===========================================
[ Entry 501 ]
===========================================
[ Entry 502 ]
I'm in cluster: 0
===========================================
[ Entry 503 ]
===========================================
[ Entry 504 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 505 ]
I'm in cluster: 0
===========================================
[ Entry 506 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 507 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 508 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 509 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 510 ]
===========================================
[ Entry 511 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 512 ]
I'm in cluster: 0
===========================================
[ Entry 513 ]
===========================================
[ Entry 514 ]
I'm in cluster: 0
===========================================
[ Entry 515 ]
I'm in cluster: 0
===========================================
[ Entry 516 ]
I'm in cluster: 0
===========================================
[ Entry 517 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 518 ]
===========================================
[ Entry 519 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 520 ]
I'm in cluster: 0
===========================================
[ Entry 521 ]
I'm in cluster: 0
===========================================
[ Entry 522 ]
===========================================
[ Entry 523 ]
I'm in cluster: 0
===========================================
[ Entry 524 ]
I'm in cluster: 0
===========================================
[ Entry 525 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 526 ]
===========================================
[ Entry 527 ]
===========================================
[ Entry 528 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 529 ]
I'm in cluster: 0
===========================================
[ Entry 530 ]
I'm in cluster: 0
===========================================
[ Entry 531 ]
===========================================
[ Entry 532 ]
I'm in cluster: 0
===========================================
[ Entry 533 ]
I'm in cluster: 0
===========================================
[ Entry 534 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 535 ]
===========================================
[ Entry 536 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 537 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 538 ]
I'm in cluster: 0
===========================================
[ Entry 539 ]
===========================================
[ Entry 540 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 541 ]
I'm in cluster: 0
===========================================
[ Entry 542 ]
I'm in cluster: 0
===========================================
[ Entry 543 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 544 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 545 ]
I'm in cluster: 0
===========================================
[ Entry 546 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 547 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 548 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 549 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 550 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 551 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 552 ]
I'm in cluster: 0
===========================================
[ Entry 553 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 554 ]
===========================================
[ Entry 555 ]
===========================================
[ Entry 556 ]
I'm in cluster: 0
===========================================
[ Entry 557 ]
I'm in cluster: 0
===========================================
[ Entry 558 ]
I'm in cluster: 0
===========================================
[ Entry 559 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 560 ]
I'm in cluster: 0
===========================================
[ Entry 561 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 562 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 563 ]
===========================================
[ Entry 564 ]
I'm in cluster: 0
===========================================
[ Entry 565 ]
I'm in cluster: 0
===========================================
[ Entry 566 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 567 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 568 ]
I'm in cluster: 0
===========================================
[ Entry 569 ]
I'm in cluster: 0
===========================================
[ Entry 570 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 571 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 572 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 573 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 574 ]
I'm in cluster: 0
===========================================
[ Entry 575 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 576 ]
===========================================
[ Entry 577 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 578 ]
I'm in cluster: 0
===========================================
[ Entry 579 ]
===========================================
[ Entry 580 ]
===========================================
[ Entry 581 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 582 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 583 ]
I'm in cluster: 0
===========================================
[ Entry 584 ]
I'm in cluster: 0
===========================================
[ Entry 585 ]
I'm in cluster: 0
===========================================
[ Entry 586 ]
===========================================
[ Entry 587 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 588 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 589 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 590 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 591 ]
===========================================
[ Entry 592 ]
I'm in cluster: 0
===========================================
[ Entry 593 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 594 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 595 ]
===========================================
[ Entry 596 ]
I'm in cluster: 0
===========================================
[ Entry 597 ]
I'm in cluster: 0
===========================================
[ Entry 598 ]
I'm in cluster: 0
===========================================
[ Entry 599 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 600 ]
===========================================
[ Entry 601 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 602 ]
I'm in cluster: 0
===========================================
[ Entry 603 ]
===========================================
[ Entry 604 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 605 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 606 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 607 ]
I'm in cluster: 0
===========================================
[ Entry 608 ]
I'm in cluster: 0
===========================================
[ Entry 609 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 610 ]
I'm in cluster: 0
===========================================
[ Entry 611 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 612 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 613 ]
I'm in cluster: 0
===========================================
[ Entry 614 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 615 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 616 ]
===========================================
[ Entry 617 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 618 ]
===========================================
[ Entry 619 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 620 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 621 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 622 ]
I'm in cluster: 0
===========================================
[ Entry 623 ]
===========================================
[ Entry 624 ]
===========================================
[ Entry 625 ]
I'm in cluster: 0
===========================================
[ Entry 626 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 627 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 628 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 629 ]
===========================================
[ Entry 630 ]
===========================================
[ Entry 631 ]
===========================================
[ Entry 632 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 633 ]
I'm in cluster: 0
===========================================
[ Entry 634 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 635 ]
I'm in cluster: 0
===========================================
[ Entry 636 ]
===========================================
[ Entry 637 ]
I'm in cluster: 0
===========================================
[ Entry 638 ]
I'm in cluster: 0
===========================================
[ Entry 639 ]
===========================================
[ Entry 640 ]
I'm in cluster: 0
===========================================
[ Entry 641 ]
===========================================
[ Entry 642 ]
I'm in cluster: 0
===========================================
[ Entry 643 ]
===========================================
[ Entry 644 ]
===========================================
[ Entry 645 ]
===========================================
[ Entry 646 ]
===========================================
[ Entry 647 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 648 ]
I'm in cluster: 0
===========================================
[ Entry 649 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 650 ]
===========================================
[ Entry 651 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 652 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 653 ]
I'm in cluster: 0
===========================================
[ Entry 654 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 655 ]
===========================================
[ Entry 656 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 657 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 658 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 659 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 660 ]
===========================================
[ Entry 661 ]
I'm in cluster: 0
===========================================
[ Entry 662 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 663 ]
I'm in cluster: 0
===========================================
[ Entry 664 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 665 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 666 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 667 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 668 ]
I'm in cluster: 0
===========================================
[ Entry 669 ]
===========================================
[ Entry 670 ]
===========================================
[ Entry 671 ]
I'm in cluster: 0
===========================================
[ Entry 672 ]
I'm in cluster: 0
===========================================
[ Entry 673 ]
I'm in cluster: 0
===========================================
[ Entry 674 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 675 ]
I'm in cluster: 0
===========================================
[ Entry 676 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 677 ]
===========================================
[ Entry 678 ]
I'm in cluster: 0
===========================================
[ Entry 679 ]
I'm in cluster: 0
===========================================
[ Entry 680 ]
I'm in cluster: 0
===========================================
[ Entry 681 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 682 ]
===========================================
[ Entry 683 ]
I'm in cluster: 0
===========================================
[ Entry 684 ]
I'm in cluster: 0
===========================================
[ Entry 685 ]
===========================================
[ Entry 686 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 687 ]
I'm in cluster: 0
===========================================
[ Entry 688 ]
I'm in cluster: 0
===========================================
[ Entry 689 ]
===========================================
[ Entry 690 ]
===========================================
[ Entry 691 ]
I'm in cluster: 0
===========================================
[ Entry 692 ]
I'm in cluster: 0
===========================================
[ Entry 693 ]
I'm in cluster: 0
===========================================
[ Entry 694 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 695 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 696 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 697 ]
I'm in cluster: 0
===========================================
[ Entry 698 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 699 ]
I'm in cluster: 0
===========================================
[ Entry 700 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 701 ]
===========================================
[ Entry 702 ]
===========================================
[ Entry 703 ]
===========================================
[ Entry 704 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 705 ]
I'm in cluster: 0
===========================================
[ Entry 706 ]
I'm in cluster: 0
===========================================
[ Entry 707 ]
I'm in cluster: 0
===========================================
[ Entry 708 ]
===========================================
[ Entry 709 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 710 ]
I'm in cluster: 0
===========================================
[ Entry 711 ]
I'm in cluster: 0
===========================================
[ Entry 712 ]
I'm in cluster: 0
===========================================
[ Entry 713 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 714 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 715 ]
===========================================
[ Entry 716 ]
I'm in cluster: 0
===========================================
[ Entry 717 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 718 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 719 ]
I'm in cluster: 0
===========================================
[ Entry 720 ]
===========================================
[ Entry 721 ]
I'm in cluster: 0
===========================================
[ Entry 722 ]
I'm in cluster: 0
===========================================
[ Entry 723 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 724 ]
I'm in cluster: 0
===========================================
[ Entry 725 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 726 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 727 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 728 ]
I'm in cluster: 0
===========================================
[ Entry 729 ]
I'm in cluster: 0
===========================================
[ Entry 730 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 731 ]
===========================================
[ Entry 732 ]
I'm in cluster: 0
===========================================
[ Entry 733 ]
===========================================
[ Entry 734 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 735 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 736 ]
===========================================
[ Entry 737 ]
I'm in cluster: 0
===========================================
[ Entry 738 ]
I'm in cluster: 0
===========================================
[ Entry 739 ]
===========================================
[ Entry 740 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 741 ]
I'm in cluster: 0
===========================================
[ Entry 742 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 743 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 744 ]
I'm in cluster: 0
===========================================
[ Entry 745 ]
I'm in cluster: 0
===========================================
[ Entry 746 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 747 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 748 ]
I'm in cluster: 0
===========================================
[ Entry 749 ]
I'm in cluster: 0
===========================================
[ Entry 750 ]
I'm in cluster: 0
===========================================
[ Entry 751 ]
I'm in cluster: 0
===========================================
[ Entry 752 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 753 ]
I'm in cluster: 0
===========================================
[ Entry 754 ]
I'm in cluster: 0
===========================================
[ Entry 755 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 756 ]
I'm in cluster: 0
===========================================
[ Entry 757 ]
===========================================
[ Entry 758 ]
===========================================
[ Entry 759 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 760 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 761 ]
I'm in cluster: 0
===========================================
[ Entry 762 ]
===========================================
[ Entry 763 ]
I'm in cluster: 0
===========================================
[ Entry 764 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 765 ]
I'm in cluster: 0
===========================================
[ Entry 766 ]
I'm in cluster: 0
===========================================
[ Entry 767 ]
I'm in cluster: 0
===========================================
[ Entry 768 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 769 ]
I'm in cluster: 0
===========================================
[ Entry 770 ]
I'm in cluster: 0
===========================================
[ Entry 771 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 772 ]
===========================================
[ Entry 773 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 774 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 775 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 776 ]
===========================================
[ Entry 777 ]
I'm in cluster: 0
===========================================
[ Entry 778 ]
I'm in cluster: 0
===========================================
[ Entry 779 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 780 ]
I'm in cluster: 0
===========================================
[ Entry 781 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 782 ]
I'm in cluster: 0
===========================================
[ Entry 783 ]
I'm in cluster: 0
===========================================
[ Entry 784 ]
I'm in cluster: 0
===========================================
[ Entry 785 ]
I'm in cluster: 0
===========================================
[ Entry 786 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 787 ]
I'm in cluster: 0
===========================================
[ Entry 788 ]
I'm in cluster: 0
===========================================
[ Entry 789 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 790 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
I'm in cluster: 5
===========================================
[ Entry 791 ]
===========================================
[ Entry 792 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 793 ]
I'm in cluster: 0
===========================================
[ Entry 794 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 795 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 796 ]
I'm in cluster: 0
===========================================
[ Entry 797 ]
===========================================
[ Entry 798 ]
I'm in cluster: 0
===========================================
[ Entry 799 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 800 ]
===========================================
[ Entry 801 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 802 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 803 ]
I'm in cluster: 0
===========================================
[ Entry 804 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 805 ]
I'm in cluster: 0
===========================================
[ Entry 806 ]
I'm in cluster: 0
===========================================
[ Entry 807 ]
===========================================
[ Entry 808 ]
I'm in cluster: 0
===========================================
[ Entry 809 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 810 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 811 ]
===========================================
[ Entry 812 ]
===========================================
[ Entry 813 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 814 ]
===========================================
[ Entry 815 ]
I'm in cluster: 0
===========================================
[ Entry 816 ]
===========================================
[ Entry 817 ]
I'm in cluster: 0
===========================================
[ Entry 818 ]
I'm in cluster: 0
===========================================
[ Entry 819 ]
I'm in cluster: 0
===========================================
[ Entry 820 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 821 ]
I'm in cluster: 0
===========================================
[ Entry 822 ]
I'm in cluster: 0
===========================================
[ Entry 823 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 824 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 825 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 826 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 827 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 828 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 829 ]
I'm in cluster: 0
===========================================
[ Entry 830 ]
I'm in cluster: 0
===========================================
[ Entry 831 ]
I'm in cluster: 0
===========================================
[ Entry 832 ]
I'm in cluster: 0
===========================================
[ Entry 833 ]
===========================================
[ Entry 834 ]
===========================================
[ Entry 835 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 836 ]
I'm in cluster: 0
===========================================
[ Entry 837 ]
I'm in cluster: 0
===========================================
[ Entry 838 ]
===========================================
[ Entry 839 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 840 ]
===========================================
[ Entry 841 ]
I'm in cluster: 0
===========================================
[ Entry 842 ]
I'm in cluster: 0
===========================================
[ Entry 843 ]
I'm in cluster: 0
===========================================
[ Entry 844 ]
===========================================
[ Entry 845 ]
===========================================
[ Entry 846 ]
I'm in cluster: 0
===========================================
[ Entry 847 ]
===========================================
[ Entry 848 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 849 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 850 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 851 ]
I'm in cluster: 0
===========================================
[ Entry 852 ]
I'm in cluster: 0
===========================================
[ Entry 853 ]
I'm in cluster: 0
===========================================
[ Entry 854 ]
===========================================
[ Entry 855 ]
===========================================
[ Entry 856 ]
I'm in cluster: 0
===========================================
[ Entry 857 ]
I'm in cluster: 0
===========================================
[ Entry 858 ]
I'm in cluster: 0
===========================================
[ Entry 859 ]
I'm in cluster: 0
===========================================
[ Entry 860 ]
===========================================
[ Entry 861 ]
I'm in cluster: 0
===========================================
[ Entry 862 ]
===========================================
[ Entry 863 ]
===========================================
[ Entry 864 ]
I'm in cluster: 0
===========================================
[ Entry 865 ]
I'm in cluster: 0
===========================================
[ Entry 866 ]
I'm in cluster: 0
===========================================
[ Entry 867 ]
===========================================
[ Entry 868 ]
I'm in cluster: 0
===========================================
[ Entry 869 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 870 ]
===========================================
[ Entry 871 ]
I'm in cluster: 0
===========================================
[ Entry 872 ]
I'm in cluster: 0
===========================================
[ Entry 873 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 874 ]
I'm in cluster: 0
===========================================
[ Entry 875 ]
===========================================
[ Entry 876 ]
I'm in cluster: 0
===========================================
[ Entry 877 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 878 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 879 ]
I'm in cluster: 0
===========================================
[ Entry 880 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 881 ]
I'm in cluster: 0
===========================================
[ Entry 882 ]
I'm in cluster: 0
===========================================
[ Entry 883 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 884 ]
I'm in cluster: 0
===========================================
[ Entry 885 ]
===========================================
[ Entry 886 ]
I'm in cluster: 0
===========================================
[ Entry 887 ]
===========================================
[ Entry 888 ]
I'm in cluster: 0
===========================================
[ Entry 889 ]
===========================================
[ Entry 890 ]
I'm in cluster: 0
===========================================
[ Entry 891 ]
I'm in cluster: 0
===========================================
[ Entry 892 ]
===========================================
[ Entry 893 ]
===========================================
[ Entry 894 ]
I'm in cluster: 0
===========================================
[ Entry 895 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 896 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 897 ]
===========================================
[ Entry 898 ]
I'm in cluster: 0
===========================================
[ Entry 899 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 900 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 901 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 902 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 903 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 904 ]
I'm in cluster: 0
===========================================
[ Entry 905 ]
===========================================
[ Entry 906 ]
===========================================
[ Entry 907 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 908 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 909 ]
I'm in cluster: 0
===========================================
[ Entry 910 ]
===========================================
[ Entry 911 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 912 ]
I'm in cluster: 0
===========================================
[ Entry 913 ]
===========================================
[ Entry 914 ]
I'm in cluster: 0
===========================================
[ Entry 915 ]
I'm in cluster: 0
===========================================
[ Entry 916 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 917 ]
I'm in cluster: 0
===========================================
[ Entry 918 ]
I'm in cluster: 0
===========================================
[ Entry 919 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 920 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 921 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 922 ]
===========================================
[ Entry 923 ]
===========================================
[ Entry 924 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 925 ]
I'm in cluster: 0
===========================================
[ Entry 926 ]
===========================================
[ Entry 927 ]
I'm in cluster: 0
===========================================
[ Entry 928 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 929 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 930 ]
===========================================
[ Entry 931 ]
I'm in cluster: 0
===========================================
[ Entry 932 ]
I'm in cluster: 0
===========================================
[ Entry 933 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 934 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 935 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 936 ]
I'm in cluster: 0
===========================================
[ Entry 937 ]
===========================================
[ Entry 938 ]
I'm in cluster: 0
===========================================
[ Entry 939 ]
===========================================
[ Entry 940 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 941 ]
===========================================
[ Entry 942 ]
===========================================
[ Entry 943 ]
===========================================
[ Entry 944 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 945 ]
===========================================
[ Entry 946 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 947 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 948 ]
===========================================
[ Entry 949 ]
I'm in cluster: 0
===========================================
[ Entry 950 ]
I'm in cluster: 0
===========================================
[ Entry 951 ]
===========================================
[ Entry 952 ]
===========================================
[ Entry 953 ]
===========================================
[ Entry 954 ]
I'm in cluster: 0
===========================================
[ Entry 955 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 956 ]
I'm in cluster: 0
===========================================
[ Entry 957 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 958 ]
I'm in cluster: 0
===========================================
[ Entry 959 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 960 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 961 ]
I'm in cluster: 0
===========================================
[ Entry 962 ]
I'm in cluster: 0
===========================================
[ Entry 963 ]
I'm in cluster: 0
===========================================
[ Entry 964 ]
===========================================
[ Entry 965 ]
I'm in cluster: 0
===========================================
[ Entry 966 ]
===========================================
[ Entry 967 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 968 ]
===========================================
[ Entry 969 ]
I'm in cluster: 0
===========================================
[ Entry 970 ]
I'm in cluster: 0
===========================================
[ Entry 971 ]
I'm in cluster: 0
===========================================
[ Entry 972 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 973 ]
I'm in cluster: 0
===========================================
[ Entry 974 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 975 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 976 ]
I'm in cluster: 0
===========================================
[ Entry 977 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 978 ]
I'm in cluster: 0
===========================================
[ Entry 979 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 980 ]
===========================================
[ Entry 981 ]
I'm in cluster: 0
===========================================
[ Entry 982 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 983 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 984 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 985 ]
I'm in cluster: 0
===========================================
[ Entry 986 ]
I'm in cluster: 0
===========================================
[ Entry 987 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 988 ]
I'm in cluster: 0
===========================================
[ Entry 989 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 990 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 991 ]
I'm in cluster: 0
===========================================
[ Entry 992 ]
I'm in cluster: 0
===========================================
[ Entry 993 ]
I'm in cluster: 0
===========================================
[ Entry 994 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 995 ]
I'm in cluster: 0
===========================================
[ Entry 996 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 997 ]
===========================================
[ Entry 998 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 999 ]
I'm in cluster: 0
===========================================
[ Entry 1000 ]
I'm in cluster: 0
===========================================
[ Entry 1001 ]
I'm in cluster: 0
===========================================
[ Entry 1002 ]
===========================================
[ Entry 1003 ]
===========================================
[ Entry 1004 ]
I'm in cluster: 0
===========================================
[ Entry 1005 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1006 ]
===========================================
[ Entry 1007 ]
===========================================
[ Entry 1008 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1009 ]
I'm in cluster: 0
===========================================
[ Entry 1010 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1011 ]
===========================================
[ Entry 1012 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1013 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1014 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
I'm in cluster: 5
===========================================
[ Entry 1015 ]
I'm in cluster: 0
===========================================
[ Entry 1016 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1017 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1018 ]
===========================================
[ Entry 1019 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1020 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1021 ]
I'm in cluster: 0
===========================================
[ Entry 1022 ]
===========================================
[ Entry 1023 ]
===========================================
[ Entry 1024 ]
===========================================
[ Entry 1025 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1026 ]
===========================================
[ Entry 1027 ]
I'm in cluster: 0
===========================================
[ Entry 1028 ]
I'm in cluster: 0
===========================================
[ Entry 1029 ]
I'm in cluster: 0
===========================================
[ Entry 1030 ]
===========================================
[ Entry 1031 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1032 ]
I'm in cluster: 0
===========================================
[ Entry 1033 ]
===========================================
[ Entry 1034 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1035 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1036 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1037 ]
I'm in cluster: 0
===========================================
[ Entry 1038 ]
I'm in cluster: 0
===========================================
[ Entry 1039 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1040 ]
I'm in cluster: 0
===========================================
[ Entry 1041 ]
===========================================
[ Entry 1042 ]
===========================================
[ Entry 1043 ]
===========================================
[ Entry 1044 ]
===========================================
[ Entry 1045 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1046 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1047 ]
===========================================
[ Entry 1048 ]
===========================================
[ Entry 1049 ]
===========================================
[ Entry 1050 ]
===========================================
[ Entry 1051 ]
===========================================
[ Entry 1052 ]
===========================================
[ Entry 1053 ]
===========================================
[ Entry 1054 ]
I'm in cluster: 0
===========================================
[ Entry 1055 ]
===========================================
[ Entry 1056 ]
===========================================
[ Entry 1057 ]
===========================================
[ Entry 1058 ]
===========================================
[ Entry 1059 ]
===========================================
[ Entry 1060 ]
===========================================
[ Entry 1061 ]
===========================================
[ Entry 1062 ]
===========================================
[ Entry 1063 ]
===========================================
[ Entry 1064 ]
===========================================
[ Entry 1065 ]
I'm in cluster: 0
===========================================
[ Entry 1066 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1067 ]
I'm in cluster: 0
===========================================
[ Entry 1068 ]
I'm in cluster: 0
===========================================
[ Entry 1069 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1070 ]
I'm in cluster: 0
===========================================
[ Entry 1071 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1072 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1073 ]
===========================================
[ Entry 1074 ]
I'm in cluster: 0
===========================================
[ Entry 1075 ]
===========================================
[ Entry 1076 ]
I'm in cluster: 0
===========================================
[ Entry 1077 ]
===========================================
[ Entry 1078 ]
===========================================
[ Entry 1079 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1080 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1081 ]
===========================================
[ Entry 1082 ]
===========================================
[ Entry 1083 ]
I'm in cluster: 0
===========================================
[ Entry 1084 ]
===========================================
[ Entry 1085 ]
I'm in cluster: 0
===========================================
[ Entry 1086 ]
===========================================
[ Entry 1087 ]
I'm in cluster: 0
===========================================
[ Entry 1088 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1089 ]
===========================================
[ Entry 1090 ]
===========================================
[ Entry 1091 ]
I'm in cluster: 0
===========================================
[ Entry 1092 ]
===========================================
[ Entry 1093 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1094 ]
I'm in cluster: 0
===========================================
[ Entry 1095 ]
I'm in cluster: 0
===========================================
[ Entry 1096 ]
===========================================
[ Entry 1097 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1098 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1099 ]
===========================================
[ Entry 1100 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1101 ]
===========================================
[ Entry 1102 ]
I'm in cluster: 0
===========================================
[ Entry 1103 ]
I'm in cluster: 0
===========================================
[ Entry 1104 ]
===========================================
[ Entry 1105 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1106 ]
I'm in cluster: 0
===========================================
[ Entry 1107 ]
I'm in cluster: 0
===========================================
[ Entry 1108 ]
I'm in cluster: 0
===========================================
[ Entry 1109 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1110 ]
I'm in cluster: 0
===========================================
[ Entry 1111 ]
I'm in cluster: 0
===========================================
[ Entry 1112 ]
===========================================
[ Entry 1113 ]
I'm in cluster: 0
===========================================
[ Entry 1114 ]
===========================================
[ Entry 1115 ]
I'm in cluster: 0
===========================================
[ Entry 1116 ]
===========================================
[ Entry 1117 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1118 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1119 ]
I'm in cluster: 0
===========================================
[ Entry 1120 ]
I'm in cluster: 0
===========================================
[ Entry 1121 ]
===========================================
[ Entry 1122 ]
===========================================
[ Entry 1123 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1124 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1125 ]
===========================================
[ Entry 1126 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1127 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1128 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1129 ]
I'm in cluster: 0
===========================================
[ Entry 1130 ]
===========================================
[ Entry 1131 ]
I'm in cluster: 0
===========================================
[ Entry 1132 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1133 ]
===========================================
[ Entry 1134 ]
I'm in cluster: 0
===========================================
[ Entry 1135 ]
===========================================
[ Entry 1136 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1137 ]
===========================================
[ Entry 1138 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1139 ]
===========================================
[ Entry 1140 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1141 ]
I'm in cluster: 0
===========================================
[ Entry 1142 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1143 ]
I'm in cluster: 0
===========================================
[ Entry 1144 ]
I'm in cluster: 0
===========================================
[ Entry 1145 ]
I'm in cluster: 0
===========================================
[ Entry 1146 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1147 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1148 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1149 ]
===========================================
[ Entry 1150 ]
I'm in cluster: 0
===========================================
[ Entry 1151 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1152 ]
I'm in cluster: 0
===========================================
[ Entry 1153 ]
===========================================
[ Entry 1154 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1155 ]
===========================================
[ Entry 1156 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1157 ]
===========================================
[ Entry 1158 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1159 ]
I'm in cluster: 0
===========================================
[ Entry 1160 ]
I'm in cluster: 0
===========================================
[ Entry 1161 ]
===========================================
[ Entry 1162 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1163 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1164 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1165 ]
===========================================
[ Entry 1166 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1167 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1168 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1169 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1170 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1171 ]
I'm in cluster: 0
===========================================
[ Entry 1172 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1173 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1174 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1175 ]
===========================================
[ Entry 1176 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1177 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1178 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1179 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1180 ]
===========================================
[ Entry 1181 ]
===========================================
[ Entry 1182 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1183 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1184 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1185 ]
I'm in cluster: 0
===========================================
[ Entry 1186 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1187 ]
I'm in cluster: 0
===========================================
[ Entry 1188 ]
I'm in cluster: 0
===========================================
[ Entry 1189 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1190 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1191 ]
===========================================
[ Entry 1192 ]
I'm in cluster: 0
===========================================
[ Entry 1193 ]
===========================================
[ Entry 1194 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1195 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1196 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1197 ]
===========================================
[ Entry 1198 ]
I'm in cluster: 0
===========================================
[ Entry 1199 ]
===========================================
[ Entry 1200 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1201 ]
I'm in cluster: 0
===========================================
[ Entry 1202 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
I'm in cluster: 5
===========================================
[ Entry 1203 ]
I'm in cluster: 0
===========================================
[ Entry 1204 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1205 ]
===========================================
[ Entry 1206 ]
I'm in cluster: 0
===========================================
[ Entry 1207 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1208 ]
I'm in cluster: 0
===========================================
[ Entry 1209 ]
===========================================
[ Entry 1210 ]
===========================================
[ Entry 1211 ]
I'm in cluster: 0
===========================================
[ Entry 1212 ]
I'm in cluster: 0
===========================================
[ Entry 1213 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1214 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1215 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1216 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1217 ]
===========================================
[ Entry 1218 ]
===========================================
[ Entry 1219 ]
I'm in cluster: 0
===========================================
[ Entry 1220 ]
===========================================
[ Entry 1221 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1222 ]
I'm in cluster: 0
===========================================
[ Entry 1223 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1224 ]
I'm in cluster: 0
===========================================
[ Entry 1225 ]
===========================================
[ Entry 1226 ]
I'm in cluster: 0
===========================================
[ Entry 1227 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1228 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1229 ]
===========================================
[ Entry 1230 ]
I'm in cluster: 0
===========================================
[ Entry 1231 ]
I'm in cluster: 0
===========================================
[ Entry 1232 ]
===========================================
[ Entry 1233 ]
===========================================
[ Entry 1234 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1235 ]
===========================================
[ Entry 1236 ]
===========================================
[ Entry 1237 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1238 ]
===========================================
[ Entry 1239 ]
I'm in cluster: 0
===========================================
[ Entry 1240 ]
===========================================
[ Entry 1241 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1242 ]
===========================================
[ Entry 1243 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1244 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1245 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1246 ]
===========================================
[ Entry 1247 ]
===========================================
[ Entry 1248 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 1249 ]
I'm in cluster: 0
===========================================
[ Entry 1250 ]
I'm in cluster: 0
===========================================
[ Entry 1251 ]
I'm in cluster: 0
===========================================
[ Entry 1252 ]
===========================================
[ Entry 1253 ]
===========================================
[ Entry 1254 ]
I'm in cluster: 0
===========================================
[ Entry 1255 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1256 ]
===========================================
[ Entry 1257 ]
===========================================
[ Entry 1258 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1259 ]
I'm in cluster: 0
===========================================
[ Entry 1260 ]
===========================================
[ Entry 1261 ]
I'm in cluster: 0
===========================================
[ Entry 1262 ]
I'm in cluster: 0
===========================================
[ Entry 1263 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1264 ]
===========================================
[ Entry 1265 ]
===========================================
[ Entry 1266 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1267 ]
===========================================
[ Entry 1268 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1269 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1270 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1271 ]
I'm in cluster: 0
===========================================
[ Entry 1272 ]
I'm in cluster: 0
===========================================
[ Entry 1273 ]
I'm in cluster: 0
===========================================
[ Entry 1274 ]
I'm in cluster: 0
===========================================
[ Entry 1275 ]
I'm in cluster: 0
===========================================
[ Entry 1276 ]
===========================================
[ Entry 1277 ]
===========================================
[ Entry 1278 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1279 ]
I'm in cluster: 0
===========================================
[ Entry 1280 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1281 ]
I'm in cluster: 0
===========================================
[ Entry 1282 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1283 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1284 ]
===========================================
[ Entry 1285 ]
===========================================
[ Entry 1286 ]
===========================================
[ Entry 1287 ]
===========================================
[ Entry 1288 ]
===========================================
[ Entry 1289 ]
I'm in cluster: 0
===========================================
[ Entry 1290 ]
===========================================
[ Entry 1291 ]
===========================================
[ Entry 1292 ]
===========================================
[ Entry 1293 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 1294 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1295 ]
I'm in cluster: 0
===========================================
[ Entry 1296 ]
===========================================
[ Entry 1297 ]
I'm in cluster: 0
===========================================
[ Entry 1298 ]
I'm in cluster: 0
===========================================
[ Entry 1299 ]
I'm in cluster: 0
===========================================
[ Entry 1300 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1301 ]
I'm in cluster: 0
===========================================
[ Entry 1302 ]
I'm in cluster: 0
===========================================
[ Entry 1303 ]
I'm in cluster: 0
===========================================
[ Entry 1304 ]
I'm in cluster: 0
===========================================
[ Entry 1305 ]
===========================================
[ Entry 1306 ]
I'm in cluster: 0
===========================================
[ Entry 1307 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1308 ]
I'm in cluster: 0
===========================================
[ Entry 1309 ]
===========================================
[ Entry 1310 ]
===========================================
[ Entry 1311 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1312 ]
I'm in cluster: 0
===========================================
[ Entry 1313 ]
===========================================
[ Entry 1314 ]
===========================================
[ Entry 1315 ]
===========================================
[ Entry 1316 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1317 ]
===========================================
[ Entry 1318 ]
I'm in cluster: 0
abratenko@trex:~/ubdl/larflow/larflow/Ana$ ls -lrt
total 23340
-rw-rw-r-- 1 abratenko abratenko     217 Aug  4 17:45 README.md
-rw-rw-r-- 1 abratenko abratenko   13937 Aug  4 17:45 kpsreco_vertexana.cxx
-rw-rw-r-- 1 abratenko abratenko    3533 Aug  4 17:45 keypoint_truthana.cxx
-rw-rw-r-- 1 abratenko abratenko   12714 Aug  4 17:45 keypoint_recoana.cxx
-rwxrwxr-x 1 abratenko abratenko 1476240 Aug  7 13:55 kpsreco_vertexana
-rw-r--r-- 1 abratenko abratenko     411 Aug  7 14:37 crt_0-0.root
-rw-r--r-- 1 abratenko abratenko     411 Aug 11 12:48 crt_0-2.root
-rw-r--r-- 1 abratenko abratenko  598080 Aug 13 13:10 crt_0-29.root
-rw-rw-r-- 1 abratenko abratenko    5089 Aug 17 15:27 CRTana.cxx~
-rw-rw-r-- 1 abratenko abratenko     766 Aug 17 15:49 GNUmakefile
-rw-rw-r-- 1 abratenko abratenko    1125 Aug 17 17:09 CRTvoxelHits.py~
-rw-rw-r-- 1 abratenko abratenko    1125 Aug 17 17:10 CRTvoxelHits.py
-rw-r--r-- 1 abratenko abratenko    5308 Aug 17 17:10 hitsPerVoxel.root
-rw-r--r-- 1 abratenko abratenko 6025901 Aug 17 17:16 crt_0-9.root
-rw-r--r-- 1 abratenko abratenko 7115474 Aug 21 10:55 crt_0-1318_10cmVoxels.root
drwxrwxr-x 2 abratenko abratenko    4096 Aug 21 11:09 oldBadPlots
-rw-rw-r-- 1 abratenko abratenko    5260 Aug 21 11:19 CRTana.cxx
-rwxrwxr-x 1 abratenko abratenko  924328 Aug 21 11:19 CRTana
-rw-r--r-- 1 abratenko abratenko 7655417 Aug 21 11:19 crt_0-1318.root
abratenko@trex:~/ubdl/larflow/larflow/Ana$ ls
crt_0-0.root                crt_0-29.root  CRTana       CRTvoxelHits.py   hitsPerVoxel.root      kpsreco_vertexana      README.md
crt_0-1318_10cmVoxels.root  crt_0-2.root   CRTana.cxx   CRTvoxelHits.py~  keypoint_recoana.cxx   kpsreco_vertexana.cxx
crt_0-1318.root             crt_0-9.root   CRTana.cxx~  GNUmakefile       keypoint_truthana.cxx  oldBadPlots
abratenko@trex:~/ubdl/larflow/larflow/Ana$ mv crt_0-1318.root crt_0-1318_5cmVoxels.root
abratenko@trex:~/ubdl/larflow/larflow/Ana$ root -l crt_0-1318_5cmVoxels.root
root [0] 
Attaching file crt_0-1318_5cmVoxels.root as _file0...
(TFile *) 0x55e76f336090
root [1] .ls
TFile**		crt_0-1318_5cmVoxels.root	
 TFile*		crt_0-1318_5cmVoxels.root	
  KEY: TTree	tree;1	tree of hits per voxel
  KEY: TH1D	hitcount_wire_hist_U;1	wire #
  KEY: TH1D	hitcount_wire_hist_V;1	wire #
  KEY: TH1D	hitcount_wire_hist_Y;1	wire #
  KEY: TH2D	hitcount_wire_th2d_U;1	wire ; tick
  KEY: TH2D	hitcount_wire_th2d_V;1	wire ; tick
  KEY: TH2D	hitcount_wire_th2d_Y;1	wire ; tick
  KEY: TH1D	hitcount_xyz_hist_x;1	
  KEY: TH1D	hitcount_xyz_hist_y;1	
  KEY: TH1D	hitcount_xyz_hist_z;1	
  KEY: TH2D	hitcount_xyz_th2d_xy;1	
  KEY: TH2D	hitcount_xyz_th2d_zy;1	
  KEY: TH2D	hitcount_xyz_th2d_zx;1	
  KEY: TH3D	hitcount_xyz_th3d;1	
root [2] tree->Draw("hitsPerVoxel")
Info in <TCanvas::MakeDefCanvas>:  created default TCanvas with name c1
root [3] tree->Draw("hitsPerVoxel>>htemp(100, 0, 100)")
Info in <TCanvas::MakeDefCanvas>:  created default TCanvas with name c1
root [4] .q
abratenko@trex:~/ubdl/larflow/larflow/Ana$ e CRTana.cxx
abratenko@trex:~/ubdl/larflow/larflow/Ana$ make
<< compile CRTana >>
g++ -g -fPIC `root-config --cflags` `larlite-config --includes` -I/home/abratenko/ubdl/larlite/../ `larcv-config --includes` `ublarcvapp-config --includes` -I/home/abratenko/ubdl/larflow/build/include  CRTana.cxx -o CRTana -L/home/abratenko/ubdl/larflow/build/lib -lLArFlow_LArFlowConstants -lLArFlow_PrepFlowMatchData -lLArFlow_KeyPoints `ublarcvapp-config --libs` -lLArCVApp_MCTools -lLArCVApp_ubdllee -lLArCVApp_UBWireTool -lLArCVApp_LArliteHandler `larcv-config --libs` -lLArCVCorePyUtil `larlite-config --libs` `root-config --libs`
abratenko@trex:~/ubdl/larflow/larflow/Ana$ ls -lrt
total 23340
-rw-rw-r-- 1 abratenko abratenko     217 Aug  4 17:45 README.md
-rw-rw-r-- 1 abratenko abratenko   13937 Aug  4 17:45 kpsreco_vertexana.cxx
-rw-rw-r-- 1 abratenko abratenko    3533 Aug  4 17:45 keypoint_truthana.cxx
-rw-rw-r-- 1 abratenko abratenko   12714 Aug  4 17:45 keypoint_recoana.cxx
-rwxrwxr-x 1 abratenko abratenko 1476240 Aug  7 13:55 kpsreco_vertexana
-rw-r--r-- 1 abratenko abratenko     411 Aug  7 14:37 crt_0-0.root
-rw-r--r-- 1 abratenko abratenko     411 Aug 11 12:48 crt_0-2.root
-rw-r--r-- 1 abratenko abratenko  598080 Aug 13 13:10 crt_0-29.root
-rw-rw-r-- 1 abratenko abratenko    5089 Aug 17 15:27 CRTana.cxx~
-rw-rw-r-- 1 abratenko abratenko     766 Aug 17 15:49 GNUmakefile
-rw-rw-r-- 1 abratenko abratenko    1125 Aug 17 17:09 CRTvoxelHits.py~
-rw-rw-r-- 1 abratenko abratenko    1125 Aug 17 17:10 CRTvoxelHits.py
-rw-r--r-- 1 abratenko abratenko    5308 Aug 17 17:10 hitsPerVoxel.root
-rw-r--r-- 1 abratenko abratenko 6025901 Aug 17 17:16 crt_0-9.root
-rw-r--r-- 1 abratenko abratenko 7115474 Aug 21 10:55 crt_0-1318_10cmVoxels.root
drwxrwxr-x 2 abratenko abratenko    4096 Aug 21 11:09 oldBadPlots
-rw-r--r-- 1 abratenko abratenko 7655417 Aug 21 11:19 crt_0-1318_5cmVoxels.root
-rw-rw-r-- 1 abratenko abratenko    5260 Aug 21 11:23 CRTana.cxx
-rwxrwxr-x 1 abratenko abratenko  924328 Aug 21 11:23 CRTana
abratenko@trex:~/ubdl/larflow/larflow/Ana$ ./CRTana ~/data/hadded_crtFiles.root 0 1319
    [NORMAL]  <open> Opening a file in READ mode: /home/abratenko/data/hadded_crtFiles.root
===========================================
[ Entry 0 ]
I'm in cluster: 0
===========================================
[ Entry 1 ]
===========================================
[ Entry 2 ]
I'm in cluster: 0
===========================================
[ Entry 3 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 4 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 5 ]
===========================================
[ Entry 6 ]
I'm in cluster: 0
===========================================
[ Entry 7 ]
I'm in cluster: 0
===========================================
[ Entry 8 ]
===========================================
[ Entry 9 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 10 ]
I'm in cluster: 0
===========================================
[ Entry 11 ]
===========================================
[ Entry 12 ]
===========================================
[ Entry 13 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 14 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 15 ]
===========================================
[ Entry 16 ]
I'm in cluster: 0
===========================================
[ Entry 17 ]
I'm in cluster: 0
===========================================
[ Entry 18 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 19 ]
===========================================
[ Entry 20 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 21 ]
I'm in cluster: 0
===========================================
[ Entry 22 ]
===========================================
[ Entry 23 ]
===========================================
[ Entry 24 ]
I'm in cluster: 0
===========================================
[ Entry 25 ]
===========================================
[ Entry 26 ]
I'm in cluster: 0
===========================================
[ Entry 27 ]
I'm in cluster: 0
===========================================
[ Entry 28 ]
I'm in cluster: 0
===========================================
[ Entry 29 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 30 ]
===========================================
[ Entry 31 ]
===========================================
[ Entry 32 ]
I'm in cluster: 0
===========================================
[ Entry 33 ]
===========================================
[ Entry 34 ]
I'm in cluster: 0
===========================================
[ Entry 35 ]
I'm in cluster: 0
===========================================
[ Entry 36 ]
I'm in cluster: 0
===========================================
[ Entry 37 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 38 ]
===========================================
[ Entry 39 ]
===========================================
[ Entry 40 ]
I'm in cluster: 0
===========================================
[ Entry 41 ]
===========================================
[ Entry 42 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 43 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 44 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 45 ]
===========================================
[ Entry 46 ]
===========================================
[ Entry 47 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 48 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 49 ]
I'm in cluster: 0
===========================================
[ Entry 50 ]
===========================================
[ Entry 51 ]
===========================================
[ Entry 52 ]
===========================================
[ Entry 53 ]
===========================================
[ Entry 54 ]
I'm in cluster: 0
===========================================
[ Entry 55 ]
===========================================
[ Entry 56 ]
===========================================
[ Entry 57 ]
I'm in cluster: 0
===========================================
[ Entry 58 ]
I'm in cluster: 0
===========================================
[ Entry 59 ]
I'm in cluster: 0
===========================================
[ Entry 60 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 61 ]
===========================================
[ Entry 62 ]
===========================================
[ Entry 63 ]
===========================================
[ Entry 64 ]
===========================================
[ Entry 65 ]
I'm in cluster: 0
===========================================
[ Entry 66 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 67 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 68 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 69 ]
===========================================
[ Entry 70 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 71 ]
I'm in cluster: 0
===========================================
[ Entry 72 ]
===========================================
[ Entry 73 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 74 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 75 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 76 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 77 ]
===========================================
[ Entry 78 ]
I'm in cluster: 0
===========================================
[ Entry 79 ]
I'm in cluster: 0
===========================================
[ Entry 80 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 81 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 82 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 83 ]
===========================================
[ Entry 84 ]
I'm in cluster: 0
===========================================
[ Entry 85 ]
I'm in cluster: 0
===========================================
[ Entry 86 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 87 ]
===========================================
[ Entry 88 ]
===========================================
[ Entry 89 ]
I'm in cluster: 0
===========================================
[ Entry 90 ]
I'm in cluster: 0
===========================================
[ Entry 91 ]
===========================================
[ Entry 92 ]
I'm in cluster: 0
===========================================
[ Entry 93 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 94 ]
I'm in cluster: 0
===========================================
[ Entry 95 ]
I'm in cluster: 0
===========================================
[ Entry 96 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 97 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 98 ]
I'm in cluster: 0
===========================================
[ Entry 99 ]
===========================================
[ Entry 100 ]
I'm in cluster: 0
===========================================
[ Entry 101 ]
===========================================
[ Entry 102 ]
I'm in cluster: 0
===========================================
[ Entry 103 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 104 ]
I'm in cluster: 0
===========================================
[ Entry 105 ]
I'm in cluster: 0
===========================================
[ Entry 106 ]
===========================================
[ Entry 107 ]
I'm in cluster: 0
===========================================
[ Entry 108 ]
I'm in cluster: 0
===========================================
[ Entry 109 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 110 ]
I'm in cluster: 0
===========================================
[ Entry 111 ]
I'm in cluster: 0
===========================================
[ Entry 112 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 113 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 114 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 115 ]
===========================================
[ Entry 116 ]
I'm in cluster: 0
===========================================
[ Entry 117 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 118 ]
===========================================
[ Entry 119 ]
I'm in cluster: 0
===========================================
[ Entry 120 ]
===========================================
[ Entry 121 ]
===========================================
[ Entry 122 ]
I'm in cluster: 0
===========================================
[ Entry 123 ]
===========================================
[ Entry 124 ]
I'm in cluster: 0
===========================================
[ Entry 125 ]
===========================================
[ Entry 126 ]
===========================================
[ Entry 127 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 128 ]
===========================================
[ Entry 129 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 130 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 131 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 132 ]
I'm in cluster: 0
===========================================
[ Entry 133 ]
I'm in cluster: 0
===========================================
[ Entry 134 ]
I'm in cluster: 0
===========================================
[ Entry 135 ]
===========================================
[ Entry 136 ]
I'm in cluster: 0
===========================================
[ Entry 137 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 138 ]
===========================================
[ Entry 139 ]
I'm in cluster: 0
===========================================
[ Entry 140 ]
===========================================
[ Entry 141 ]
===========================================
[ Entry 142 ]
I'm in cluster: 0
===========================================
[ Entry 143 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 144 ]
I'm in cluster: 0
===========================================
[ Entry 145 ]
I'm in cluster: 0
===========================================
[ Entry 146 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 147 ]
I'm in cluster: 0
===========================================
[ Entry 148 ]
I'm in cluster: 0
===========================================
[ Entry 149 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 150 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 151 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 152 ]
I'm in cluster: 0
===========================================
[ Entry 153 ]
===========================================
[ Entry 154 ]
===========================================
[ Entry 155 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 156 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 157 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 158 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 159 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 160 ]
I'm in cluster: 0
===========================================
[ Entry 161 ]
I'm in cluster: 0
===========================================
[ Entry 162 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 163 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 164 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 165 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
I'm in cluster: 5
===========================================
[ Entry 166 ]
I'm in cluster: 0
===========================================
[ Entry 167 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 168 ]
I'm in cluster: 0
===========================================
[ Entry 169 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 170 ]
===========================================
[ Entry 171 ]
===========================================
[ Entry 172 ]
I'm in cluster: 0
===========================================
[ Entry 173 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 174 ]
I'm in cluster: 0
===========================================
[ Entry 175 ]
I'm in cluster: 0
===========================================
[ Entry 176 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 177 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 178 ]
===========================================
[ Entry 179 ]
===========================================
[ Entry 180 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 181 ]
I'm in cluster: 0
===========================================
[ Entry 182 ]
===========================================
[ Entry 183 ]
I'm in cluster: 0
===========================================
[ Entry 184 ]
I'm in cluster: 0
===========================================
[ Entry 185 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 186 ]
===========================================
[ Entry 187 ]
===========================================
[ Entry 188 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 189 ]
I'm in cluster: 0
===========================================
[ Entry 190 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 191 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 192 ]
===========================================
[ Entry 193 ]
===========================================
[ Entry 194 ]
===========================================
[ Entry 195 ]
I'm in cluster: 0
===========================================
[ Entry 196 ]
I'm in cluster: 0
===========================================
[ Entry 197 ]
I'm in cluster: 0
===========================================
[ Entry 198 ]
I'm in cluster: 0
===========================================
[ Entry 199 ]
I'm in cluster: 0
===========================================
[ Entry 200 ]
I'm in cluster: 0
===========================================
[ Entry 201 ]
I'm in cluster: 0
===========================================
[ Entry 202 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 203 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 204 ]
I'm in cluster: 0
===========================================
[ Entry 205 ]
===========================================
[ Entry 206 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 207 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 208 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 209 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 210 ]
I'm in cluster: 0
===========================================
[ Entry 211 ]
I'm in cluster: 0
===========================================
[ Entry 212 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 213 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 214 ]
===========================================
[ Entry 215 ]
I'm in cluster: 0
===========================================
[ Entry 216 ]
I'm in cluster: 0
===========================================
[ Entry 217 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 218 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 219 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 220 ]
I'm in cluster: 0
===========================================
[ Entry 221 ]
===========================================
[ Entry 222 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 223 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 224 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 225 ]
===========================================
[ Entry 226 ]
I'm in cluster: 0
===========================================
[ Entry 227 ]
I'm in cluster: 0
===========================================
[ Entry 228 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 229 ]
I'm in cluster: 0
===========================================
[ Entry 230 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 231 ]
===========================================
[ Entry 232 ]
I'm in cluster: 0
===========================================
[ Entry 233 ]
===========================================
[ Entry 234 ]
===========================================
[ Entry 235 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 236 ]
===========================================
[ Entry 237 ]
I'm in cluster: 0
===========================================
[ Entry 238 ]
===========================================
[ Entry 239 ]
I'm in cluster: 0
===========================================
[ Entry 240 ]
I'm in cluster: 0
===========================================
[ Entry 241 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 242 ]
===========================================
[ Entry 243 ]
===========================================
[ Entry 244 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 245 ]
===========================================
[ Entry 246 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 247 ]
I'm in cluster: 0
===========================================
[ Entry 248 ]
===========================================
[ Entry 249 ]
===========================================
[ Entry 250 ]
===========================================
[ Entry 251 ]
===========================================
[ Entry 252 ]
===========================================
[ Entry 253 ]
===========================================
[ Entry 254 ]
===========================================
[ Entry 255 ]
===========================================
[ Entry 256 ]
===========================================
[ Entry 257 ]
===========================================
[ Entry 258 ]
===========================================
[ Entry 259 ]
===========================================
[ Entry 260 ]
===========================================
[ Entry 261 ]
===========================================
[ Entry 262 ]
===========================================
[ Entry 263 ]
===========================================
[ Entry 264 ]
===========================================
[ Entry 265 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 266 ]
I'm in cluster: 0
===========================================
[ Entry 267 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 268 ]
===========================================
[ Entry 269 ]
I'm in cluster: 0
===========================================
[ Entry 270 ]
===========================================
[ Entry 271 ]
===========================================
[ Entry 272 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 273 ]
I'm in cluster: 0
===========================================
[ Entry 274 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 275 ]
===========================================
[ Entry 276 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 277 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 278 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 279 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 280 ]
I'm in cluster: 0
===========================================
[ Entry 281 ]
I'm in cluster: 0
===========================================
[ Entry 282 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 283 ]
===========================================
[ Entry 284 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 285 ]
I'm in cluster: 0
===========================================
[ Entry 286 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 287 ]
===========================================
[ Entry 288 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 289 ]
===========================================
[ Entry 290 ]
I'm in cluster: 0
===========================================
[ Entry 291 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 292 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 293 ]
===========================================
[ Entry 294 ]
===========================================
[ Entry 295 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 296 ]
===========================================
[ Entry 297 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 298 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 299 ]
===========================================
[ Entry 300 ]
===========================================
[ Entry 301 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 302 ]
I'm in cluster: 0
===========================================
[ Entry 303 ]
===========================================
[ Entry 304 ]
===========================================
[ Entry 305 ]
I'm in cluster: 0
===========================================
[ Entry 306 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 307 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 308 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 309 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 310 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 311 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 312 ]
===========================================
[ Entry 313 ]
===========================================
[ Entry 314 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 315 ]
===========================================
[ Entry 316 ]
I'm in cluster: 0
===========================================
[ Entry 317 ]
===========================================
[ Entry 318 ]
===========================================
[ Entry 319 ]
I'm in cluster: 0
===========================================
[ Entry 320 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 321 ]
I'm in cluster: 0
===========================================
[ Entry 322 ]
===========================================
[ Entry 323 ]
I'm in cluster: 0
===========================================
[ Entry 324 ]
I'm in cluster: 0
===========================================
[ Entry 325 ]
I'm in cluster: 0
===========================================
[ Entry 326 ]
===========================================
[ Entry 327 ]
I'm in cluster: 0
===========================================
[ Entry 328 ]
I'm in cluster: 0
===========================================
[ Entry 329 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 330 ]
===========================================
[ Entry 331 ]
===========================================
[ Entry 332 ]
===========================================
[ Entry 333 ]
I'm in cluster: 0
===========================================
[ Entry 334 ]
I'm in cluster: 0
===========================================
[ Entry 335 ]
I'm in cluster: 0
===========================================
[ Entry 336 ]
===========================================
[ Entry 337 ]
I'm in cluster: 0
===========================================
[ Entry 338 ]
===========================================
[ Entry 339 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 340 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 341 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 342 ]
I'm in cluster: 0
===========================================
[ Entry 343 ]
I'm in cluster: 0
===========================================
[ Entry 344 ]
I'm in cluster: 0
===========================================
[ Entry 345 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 346 ]
I'm in cluster: 0
===========================================
[ Entry 347 ]
I'm in cluster: 0
===========================================
[ Entry 348 ]
I'm in cluster: 0
===========================================
[ Entry 349 ]
I'm in cluster: 0
===========================================
[ Entry 350 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 351 ]
I'm in cluster: 0
===========================================
[ Entry 352 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 353 ]
I'm in cluster: 0
===========================================
[ Entry 354 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 355 ]
===========================================
[ Entry 356 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 357 ]
I'm in cluster: 0
===========================================
[ Entry 358 ]
===========================================
[ Entry 359 ]
===========================================
[ Entry 360 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 361 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 362 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 363 ]
I'm in cluster: 0
===========================================
[ Entry 364 ]
I'm in cluster: 0
===========================================
[ Entry 365 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 366 ]
===========================================
[ Entry 367 ]
===========================================
[ Entry 368 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 369 ]
I'm in cluster: 0
===========================================
[ Entry 370 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 371 ]
===========================================
[ Entry 372 ]
I'm in cluster: 0
===========================================
[ Entry 373 ]
I'm in cluster: 0
===========================================
[ Entry 374 ]
I'm in cluster: 0
===========================================
[ Entry 375 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 376 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 377 ]
===========================================
[ Entry 378 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 379 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 380 ]
===========================================
[ Entry 381 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 382 ]
I'm in cluster: 0
===========================================
[ Entry 383 ]
I'm in cluster: 0
===========================================
[ Entry 384 ]
I'm in cluster: 0
===========================================
[ Entry 385 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 386 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 387 ]
I'm in cluster: 0
===========================================
[ Entry 388 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 389 ]
I'm in cluster: 0
===========================================
[ Entry 390 ]
I'm in cluster: 0
===========================================
[ Entry 391 ]
I'm in cluster: 0
===========================================
[ Entry 392 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 393 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 394 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 395 ]
I'm in cluster: 0
===========================================
[ Entry 396 ]
I'm in cluster: 0
===========================================
[ Entry 397 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 398 ]
I'm in cluster: 0
===========================================
[ Entry 399 ]
===========================================
[ Entry 400 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 401 ]
===========================================
[ Entry 402 ]
I'm in cluster: 0
===========================================
[ Entry 403 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 404 ]
I'm in cluster: 0
===========================================
[ Entry 405 ]
===========================================
[ Entry 406 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 407 ]
I'm in cluster: 0
===========================================
[ Entry 408 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 409 ]
I'm in cluster: 0
===========================================
[ Entry 410 ]
I'm in cluster: 0
===========================================
[ Entry 411 ]
===========================================
[ Entry 412 ]
===========================================
[ Entry 413 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 414 ]
===========================================
[ Entry 415 ]
===========================================
[ Entry 416 ]
I'm in cluster: 0
===========================================
[ Entry 417 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 418 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 419 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 420 ]
===========================================
[ Entry 421 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 422 ]
===========================================
[ Entry 423 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 424 ]
===========================================
[ Entry 425 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 426 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 427 ]
I'm in cluster: 0
===========================================
[ Entry 428 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 429 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 430 ]
===========================================
[ Entry 431 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 432 ]
I'm in cluster: 0
===========================================
[ Entry 433 ]
I'm in cluster: 0
===========================================
[ Entry 434 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 435 ]
===========================================
[ Entry 436 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 437 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 438 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 439 ]
===========================================
[ Entry 440 ]
===========================================
[ Entry 441 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 442 ]
===========================================
[ Entry 443 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 444 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 445 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 446 ]
===========================================
[ Entry 447 ]
===========================================
[ Entry 448 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 449 ]
===========================================
[ Entry 450 ]
===========================================
[ Entry 451 ]
===========================================
[ Entry 452 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 453 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 454 ]
===========================================
[ Entry 455 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 456 ]
===========================================
[ Entry 457 ]
===========================================
[ Entry 458 ]
I'm in cluster: 0
===========================================
[ Entry 459 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 460 ]
I'm in cluster: 0
===========================================
[ Entry 461 ]
I'm in cluster: 0
===========================================
[ Entry 462 ]
===========================================
[ Entry 463 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 464 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 465 ]
I'm in cluster: 0
===========================================
[ Entry 466 ]
I'm in cluster: 0
===========================================
[ Entry 467 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 468 ]
I'm in cluster: 0
===========================================
[ Entry 469 ]
===========================================
[ Entry 470 ]
I'm in cluster: 0
===========================================
[ Entry 471 ]
I'm in cluster: 0
===========================================
[ Entry 472 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 473 ]
===========================================
[ Entry 474 ]
I'm in cluster: 0
===========================================
[ Entry 475 ]
===========================================
[ Entry 476 ]
I'm in cluster: 0
===========================================
[ Entry 477 ]
I'm in cluster: 0
===========================================
[ Entry 478 ]
===========================================
[ Entry 479 ]
===========================================
[ Entry 480 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 481 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 482 ]
I'm in cluster: 0
===========================================
[ Entry 483 ]
===========================================
[ Entry 484 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 485 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 486 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 487 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 488 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 489 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 490 ]
I'm in cluster: 0
===========================================
[ Entry 491 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 492 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 493 ]
===========================================
[ Entry 494 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 495 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 496 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 497 ]
I'm in cluster: 0
===========================================
[ Entry 498 ]
I'm in cluster: 0
===========================================
[ Entry 499 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 500 ]
===========================================
[ Entry 501 ]
===========================================
[ Entry 502 ]
I'm in cluster: 0
===========================================
[ Entry 503 ]
===========================================
[ Entry 504 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 505 ]
I'm in cluster: 0
===========================================
[ Entry 506 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 507 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 508 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 509 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 510 ]
===========================================
[ Entry 511 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 512 ]
I'm in cluster: 0
===========================================
[ Entry 513 ]
===========================================
[ Entry 514 ]
I'm in cluster: 0
===========================================
[ Entry 515 ]
I'm in cluster: 0
===========================================
[ Entry 516 ]
I'm in cluster: 0
===========================================
[ Entry 517 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 518 ]
===========================================
[ Entry 519 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 520 ]
I'm in cluster: 0
===========================================
[ Entry 521 ]
I'm in cluster: 0
===========================================
[ Entry 522 ]
===========================================
[ Entry 523 ]
I'm in cluster: 0
===========================================
[ Entry 524 ]
I'm in cluster: 0
===========================================
[ Entry 525 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 526 ]
===========================================
[ Entry 527 ]
===========================================
[ Entry 528 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 529 ]
I'm in cluster: 0
===========================================
[ Entry 530 ]
I'm in cluster: 0
===========================================
[ Entry 531 ]
===========================================
[ Entry 532 ]
I'm in cluster: 0
===========================================
[ Entry 533 ]
I'm in cluster: 0
===========================================
[ Entry 534 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 535 ]
===========================================
[ Entry 536 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 537 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 538 ]
I'm in cluster: 0
===========================================
[ Entry 539 ]
===========================================
[ Entry 540 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 541 ]
I'm in cluster: 0
===========================================
[ Entry 542 ]
I'm in cluster: 0
===========================================
[ Entry 543 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 544 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 545 ]
I'm in cluster: 0
===========================================
[ Entry 546 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 547 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 548 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 549 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 550 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 551 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 552 ]
I'm in cluster: 0
===========================================
[ Entry 553 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 554 ]
===========================================
[ Entry 555 ]
===========================================
[ Entry 556 ]
I'm in cluster: 0
===========================================
[ Entry 557 ]
I'm in cluster: 0
===========================================
[ Entry 558 ]
I'm in cluster: 0
===========================================
[ Entry 559 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 560 ]
I'm in cluster: 0
===========================================
[ Entry 561 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 562 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 563 ]
===========================================
[ Entry 564 ]
I'm in cluster: 0
===========================================
[ Entry 565 ]
I'm in cluster: 0
===========================================
[ Entry 566 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 567 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 568 ]
I'm in cluster: 0
===========================================
[ Entry 569 ]
I'm in cluster: 0
===========================================
[ Entry 570 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 571 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 572 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 573 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 574 ]
I'm in cluster: 0
===========================================
[ Entry 575 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 576 ]
===========================================
[ Entry 577 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 578 ]
I'm in cluster: 0
===========================================
[ Entry 579 ]
===========================================
[ Entry 580 ]
===========================================
[ Entry 581 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 582 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 583 ]
I'm in cluster: 0
===========================================
[ Entry 584 ]
I'm in cluster: 0
===========================================
[ Entry 585 ]
I'm in cluster: 0
===========================================
[ Entry 586 ]
===========================================
[ Entry 587 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 588 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 589 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 590 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 591 ]
===========================================
[ Entry 592 ]
I'm in cluster: 0
===========================================
[ Entry 593 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 594 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 595 ]
===========================================
[ Entry 596 ]
I'm in cluster: 0
===========================================
[ Entry 597 ]
I'm in cluster: 0
===========================================
[ Entry 598 ]
I'm in cluster: 0
===========================================
[ Entry 599 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 600 ]
===========================================
[ Entry 601 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 602 ]
I'm in cluster: 0
===========================================
[ Entry 603 ]
===========================================
[ Entry 604 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 605 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 606 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 607 ]
I'm in cluster: 0
===========================================
[ Entry 608 ]
I'm in cluster: 0
===========================================
[ Entry 609 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 610 ]
I'm in cluster: 0
===========================================
[ Entry 611 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 612 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 613 ]
I'm in cluster: 0
===========================================
[ Entry 614 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 615 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 616 ]
===========================================
[ Entry 617 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 618 ]
===========================================
[ Entry 619 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 620 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 621 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 622 ]
I'm in cluster: 0
===========================================
[ Entry 623 ]
===========================================
[ Entry 624 ]
===========================================
[ Entry 625 ]
I'm in cluster: 0
===========================================
[ Entry 626 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 627 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 628 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 629 ]
===========================================
[ Entry 630 ]
===========================================
[ Entry 631 ]
===========================================
[ Entry 632 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 633 ]
I'm in cluster: 0
===========================================
[ Entry 634 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 635 ]
I'm in cluster: 0
===========================================
[ Entry 636 ]
===========================================
[ Entry 637 ]
I'm in cluster: 0
===========================================
[ Entry 638 ]
I'm in cluster: 0
===========================================
[ Entry 639 ]
===========================================
[ Entry 640 ]
I'm in cluster: 0
===========================================
[ Entry 641 ]
===========================================
[ Entry 642 ]
I'm in cluster: 0
===========================================
[ Entry 643 ]
===========================================
[ Entry 644 ]
===========================================
[ Entry 645 ]
===========================================
[ Entry 646 ]
===========================================
[ Entry 647 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 648 ]
I'm in cluster: 0
===========================================
[ Entry 649 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 650 ]
===========================================
[ Entry 651 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 652 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 653 ]
I'm in cluster: 0
===========================================
[ Entry 654 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 655 ]
===========================================
[ Entry 656 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 657 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 658 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 659 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 660 ]
===========================================
[ Entry 661 ]
I'm in cluster: 0
===========================================
[ Entry 662 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 663 ]
I'm in cluster: 0
===========================================
[ Entry 664 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 665 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 666 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 667 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 668 ]
I'm in cluster: 0
===========================================
[ Entry 669 ]
===========================================
[ Entry 670 ]
===========================================
[ Entry 671 ]
I'm in cluster: 0
===========================================
[ Entry 672 ]
I'm in cluster: 0
===========================================
[ Entry 673 ]
I'm in cluster: 0
===========================================
[ Entry 674 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 675 ]
I'm in cluster: 0
===========================================
[ Entry 676 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 677 ]
===========================================
[ Entry 678 ]
I'm in cluster: 0
===========================================
[ Entry 679 ]
I'm in cluster: 0
===========================================
[ Entry 680 ]
I'm in cluster: 0
===========================================
[ Entry 681 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 682 ]
===========================================
[ Entry 683 ]
I'm in cluster: 0
===========================================
[ Entry 684 ]
I'm in cluster: 0
===========================================
[ Entry 685 ]
===========================================
[ Entry 686 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 687 ]
I'm in cluster: 0
===========================================
[ Entry 688 ]
I'm in cluster: 0
===========================================
[ Entry 689 ]
===========================================
[ Entry 690 ]
===========================================
[ Entry 691 ]
I'm in cluster: 0
===========================================
[ Entry 692 ]
I'm in cluster: 0
===========================================
[ Entry 693 ]
I'm in cluster: 0
===========================================
[ Entry 694 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 695 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 696 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 697 ]
I'm in cluster: 0
===========================================
[ Entry 698 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 699 ]
I'm in cluster: 0
===========================================
[ Entry 700 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 701 ]
===========================================
[ Entry 702 ]
===========================================
[ Entry 703 ]
===========================================
[ Entry 704 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 705 ]
I'm in cluster: 0
===========================================
[ Entry 706 ]
I'm in cluster: 0
===========================================
[ Entry 707 ]
I'm in cluster: 0
===========================================
[ Entry 708 ]
===========================================
[ Entry 709 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 710 ]
I'm in cluster: 0
===========================================
[ Entry 711 ]
I'm in cluster: 0
===========================================
[ Entry 712 ]
I'm in cluster: 0
===========================================
[ Entry 713 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 714 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 715 ]
===========================================
[ Entry 716 ]
I'm in cluster: 0
===========================================
[ Entry 717 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 718 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 719 ]
I'm in cluster: 0
===========================================
[ Entry 720 ]
===========================================
[ Entry 721 ]
I'm in cluster: 0
===========================================
[ Entry 722 ]
I'm in cluster: 0
===========================================
[ Entry 723 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 724 ]
I'm in cluster: 0
===========================================
[ Entry 725 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 726 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 727 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 728 ]
I'm in cluster: 0
===========================================
[ Entry 729 ]
I'm in cluster: 0
===========================================
[ Entry 730 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 731 ]
===========================================
[ Entry 732 ]
I'm in cluster: 0
===========================================
[ Entry 733 ]
===========================================
[ Entry 734 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 735 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 736 ]
===========================================
[ Entry 737 ]
I'm in cluster: 0
===========================================
[ Entry 738 ]
I'm in cluster: 0
===========================================
[ Entry 739 ]
===========================================
[ Entry 740 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 741 ]
I'm in cluster: 0
===========================================
[ Entry 742 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 743 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 744 ]
I'm in cluster: 0
===========================================
[ Entry 745 ]
I'm in cluster: 0
===========================================
[ Entry 746 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 747 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 748 ]
I'm in cluster: 0
===========================================
[ Entry 749 ]
I'm in cluster: 0
===========================================
[ Entry 750 ]
I'm in cluster: 0
===========================================
[ Entry 751 ]
I'm in cluster: 0
===========================================
[ Entry 752 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 753 ]
I'm in cluster: 0
===========================================
[ Entry 754 ]
I'm in cluster: 0
===========================================
[ Entry 755 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 756 ]
I'm in cluster: 0
===========================================
[ Entry 757 ]
===========================================
[ Entry 758 ]
===========================================
[ Entry 759 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 760 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 761 ]
I'm in cluster: 0
===========================================
[ Entry 762 ]
===========================================
[ Entry 763 ]
I'm in cluster: 0
===========================================
[ Entry 764 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 765 ]
I'm in cluster: 0
===========================================
[ Entry 766 ]
I'm in cluster: 0
===========================================
[ Entry 767 ]
I'm in cluster: 0
===========================================
[ Entry 768 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 769 ]
I'm in cluster: 0
===========================================
[ Entry 770 ]
I'm in cluster: 0
===========================================
[ Entry 771 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 772 ]
===========================================
[ Entry 773 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 774 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 775 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 776 ]
===========================================
[ Entry 777 ]
I'm in cluster: 0
===========================================
[ Entry 778 ]
I'm in cluster: 0
===========================================
[ Entry 779 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 780 ]
I'm in cluster: 0
===========================================
[ Entry 781 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 782 ]
I'm in cluster: 0
===========================================
[ Entry 783 ]
I'm in cluster: 0
===========================================
[ Entry 784 ]
I'm in cluster: 0
===========================================
[ Entry 785 ]
I'm in cluster: 0
===========================================
[ Entry 786 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 787 ]
I'm in cluster: 0
===========================================
[ Entry 788 ]
I'm in cluster: 0
===========================================
[ Entry 789 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 790 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
I'm in cluster: 5
===========================================
[ Entry 791 ]
===========================================
[ Entry 792 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 793 ]
I'm in cluster: 0
===========================================
[ Entry 794 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 795 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 796 ]
I'm in cluster: 0
===========================================
[ Entry 797 ]
===========================================
[ Entry 798 ]
I'm in cluster: 0
===========================================
[ Entry 799 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 800 ]
===========================================
[ Entry 801 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 802 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 803 ]
I'm in cluster: 0
===========================================
[ Entry 804 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 805 ]
I'm in cluster: 0
===========================================
[ Entry 806 ]
I'm in cluster: 0
===========================================
[ Entry 807 ]
===========================================
[ Entry 808 ]
I'm in cluster: 0
===========================================
[ Entry 809 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 810 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 811 ]
===========================================
[ Entry 812 ]
===========================================
[ Entry 813 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 814 ]
===========================================
[ Entry 815 ]
I'm in cluster: 0
===========================================
[ Entry 816 ]
===========================================
[ Entry 817 ]
I'm in cluster: 0
===========================================
[ Entry 818 ]
I'm in cluster: 0
===========================================
[ Entry 819 ]
I'm in cluster: 0
===========================================
[ Entry 820 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 821 ]
I'm in cluster: 0
===========================================
[ Entry 822 ]
I'm in cluster: 0
===========================================
[ Entry 823 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 824 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 825 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 826 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 827 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 828 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 829 ]
I'm in cluster: 0
===========================================
[ Entry 830 ]
I'm in cluster: 0
===========================================
[ Entry 831 ]
I'm in cluster: 0
===========================================
[ Entry 832 ]
I'm in cluster: 0
===========================================
[ Entry 833 ]
===========================================
[ Entry 834 ]
===========================================
[ Entry 835 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 836 ]
I'm in cluster: 0
===========================================
[ Entry 837 ]
I'm in cluster: 0
===========================================
[ Entry 838 ]
===========================================
[ Entry 839 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 840 ]
===========================================
[ Entry 841 ]
I'm in cluster: 0
===========================================
[ Entry 842 ]
I'm in cluster: 0
===========================================
[ Entry 843 ]
I'm in cluster: 0
===========================================
[ Entry 844 ]
===========================================
[ Entry 845 ]
===========================================
[ Entry 846 ]
I'm in cluster: 0
===========================================
[ Entry 847 ]
===========================================
[ Entry 848 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 849 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 850 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 851 ]
I'm in cluster: 0
===========================================
[ Entry 852 ]
I'm in cluster: 0
===========================================
[ Entry 853 ]
I'm in cluster: 0
===========================================
[ Entry 854 ]
===========================================
[ Entry 855 ]
===========================================
[ Entry 856 ]
I'm in cluster: 0
===========================================
[ Entry 857 ]
I'm in cluster: 0
===========================================
[ Entry 858 ]
I'm in cluster: 0
===========================================
[ Entry 859 ]
I'm in cluster: 0
===========================================
[ Entry 860 ]
===========================================
[ Entry 861 ]
I'm in cluster: 0
===========================================
[ Entry 862 ]
===========================================
[ Entry 863 ]
===========================================
[ Entry 864 ]
I'm in cluster: 0
===========================================
[ Entry 865 ]
I'm in cluster: 0
===========================================
[ Entry 866 ]
I'm in cluster: 0
===========================================
[ Entry 867 ]
===========================================
[ Entry 868 ]
I'm in cluster: 0
===========================================
[ Entry 869 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 870 ]
===========================================
[ Entry 871 ]
I'm in cluster: 0
===========================================
[ Entry 872 ]
I'm in cluster: 0
===========================================
[ Entry 873 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 874 ]
I'm in cluster: 0
===========================================
[ Entry 875 ]
===========================================
[ Entry 876 ]
I'm in cluster: 0
===========================================
[ Entry 877 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 878 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 879 ]
I'm in cluster: 0
===========================================
[ Entry 880 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 881 ]
I'm in cluster: 0
===========================================
[ Entry 882 ]
I'm in cluster: 0
===========================================
[ Entry 883 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 884 ]
I'm in cluster: 0
===========================================
[ Entry 885 ]
===========================================
[ Entry 886 ]
I'm in cluster: 0
===========================================
[ Entry 887 ]
===========================================
[ Entry 888 ]
I'm in cluster: 0
===========================================
[ Entry 889 ]
===========================================
[ Entry 890 ]
I'm in cluster: 0
===========================================
[ Entry 891 ]
I'm in cluster: 0
===========================================
[ Entry 892 ]
===========================================
[ Entry 893 ]
===========================================
[ Entry 894 ]
I'm in cluster: 0
===========================================
[ Entry 895 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 896 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 897 ]
===========================================
[ Entry 898 ]
I'm in cluster: 0
===========================================
[ Entry 899 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 900 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 901 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 902 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 903 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 904 ]
I'm in cluster: 0
===========================================
[ Entry 905 ]
===========================================
[ Entry 906 ]
===========================================
[ Entry 907 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 908 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 909 ]
I'm in cluster: 0
===========================================
[ Entry 910 ]
===========================================
[ Entry 911 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 912 ]
I'm in cluster: 0
===========================================
[ Entry 913 ]
===========================================
[ Entry 914 ]
I'm in cluster: 0
===========================================
[ Entry 915 ]
I'm in cluster: 0
===========================================
[ Entry 916 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 917 ]
I'm in cluster: 0
===========================================
[ Entry 918 ]
I'm in cluster: 0
===========================================
[ Entry 919 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 920 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 921 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 922 ]
===========================================
[ Entry 923 ]
===========================================
[ Entry 924 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 925 ]
I'm in cluster: 0
===========================================
[ Entry 926 ]
===========================================
[ Entry 927 ]
I'm in cluster: 0
===========================================
[ Entry 928 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 929 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 930 ]
===========================================
[ Entry 931 ]
I'm in cluster: 0
===========================================
[ Entry 932 ]
I'm in cluster: 0
===========================================
[ Entry 933 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 934 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 935 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 936 ]
I'm in cluster: 0
===========================================
[ Entry 937 ]
===========================================
[ Entry 938 ]
I'm in cluster: 0
===========================================
[ Entry 939 ]
===========================================
[ Entry 940 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 941 ]
===========================================
[ Entry 942 ]
===========================================
[ Entry 943 ]
===========================================
[ Entry 944 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 945 ]
===========================================
[ Entry 946 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 947 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 948 ]
===========================================
[ Entry 949 ]
I'm in cluster: 0
===========================================
[ Entry 950 ]
I'm in cluster: 0
===========================================
[ Entry 951 ]
===========================================
[ Entry 952 ]
===========================================
[ Entry 953 ]
===========================================
[ Entry 954 ]
I'm in cluster: 0
===========================================
[ Entry 955 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 956 ]
I'm in cluster: 0
===========================================
[ Entry 957 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 958 ]
I'm in cluster: 0
===========================================
[ Entry 959 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 960 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 961 ]
I'm in cluster: 0
===========================================
[ Entry 962 ]
I'm in cluster: 0
===========================================
[ Entry 963 ]
I'm in cluster: 0
===========================================
[ Entry 964 ]
===========================================
[ Entry 965 ]
I'm in cluster: 0
===========================================
[ Entry 966 ]
===========================================
[ Entry 967 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 968 ]
===========================================
[ Entry 969 ]
I'm in cluster: 0
===========================================
[ Entry 970 ]
I'm in cluster: 0
===========================================
[ Entry 971 ]
I'm in cluster: 0
===========================================
[ Entry 972 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 973 ]
I'm in cluster: 0
===========================================
[ Entry 974 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 975 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 976 ]
I'm in cluster: 0
===========================================
[ Entry 977 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 978 ]
I'm in cluster: 0
===========================================
[ Entry 979 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 980 ]
===========================================
[ Entry 981 ]
I'm in cluster: 0
===========================================
[ Entry 982 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 983 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 984 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 985 ]
I'm in cluster: 0
===========================================
[ Entry 986 ]
I'm in cluster: 0
===========================================
[ Entry 987 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 988 ]
I'm in cluster: 0
===========================================
[ Entry 989 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 990 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 991 ]
I'm in cluster: 0
===========================================
[ Entry 992 ]
I'm in cluster: 0
===========================================
[ Entry 993 ]
I'm in cluster: 0
===========================================
[ Entry 994 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 995 ]
I'm in cluster: 0
===========================================
[ Entry 996 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 997 ]
===========================================
[ Entry 998 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 999 ]
I'm in cluster: 0
===========================================
[ Entry 1000 ]
I'm in cluster: 0
===========================================
[ Entry 1001 ]
I'm in cluster: 0
===========================================
[ Entry 1002 ]
===========================================
[ Entry 1003 ]
===========================================
[ Entry 1004 ]
I'm in cluster: 0
===========================================
[ Entry 1005 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1006 ]
===========================================
[ Entry 1007 ]
===========================================
[ Entry 1008 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1009 ]
I'm in cluster: 0
===========================================
[ Entry 1010 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1011 ]
===========================================
[ Entry 1012 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1013 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1014 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
I'm in cluster: 5
===========================================
[ Entry 1015 ]
I'm in cluster: 0
===========================================
[ Entry 1016 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1017 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1018 ]
===========================================
[ Entry 1019 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1020 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1021 ]
I'm in cluster: 0
===========================================
[ Entry 1022 ]
===========================================
[ Entry 1023 ]
===========================================
[ Entry 1024 ]
===========================================
[ Entry 1025 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1026 ]
===========================================
[ Entry 1027 ]
I'm in cluster: 0
===========================================
[ Entry 1028 ]
I'm in cluster: 0
===========================================
[ Entry 1029 ]
I'm in cluster: 0
===========================================
[ Entry 1030 ]
===========================================
[ Entry 1031 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1032 ]
I'm in cluster: 0
===========================================
[ Entry 1033 ]
===========================================
[ Entry 1034 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1035 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1036 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1037 ]
I'm in cluster: 0
===========================================
[ Entry 1038 ]
I'm in cluster: 0
===========================================
[ Entry 1039 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1040 ]
I'm in cluster: 0
===========================================
[ Entry 1041 ]
===========================================
[ Entry 1042 ]
===========================================
[ Entry 1043 ]
===========================================
[ Entry 1044 ]
===========================================
[ Entry 1045 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1046 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1047 ]
===========================================
[ Entry 1048 ]
===========================================
[ Entry 1049 ]
===========================================
[ Entry 1050 ]
===========================================
[ Entry 1051 ]
===========================================
[ Entry 1052 ]
===========================================
[ Entry 1053 ]
===========================================
[ Entry 1054 ]
I'm in cluster: 0
===========================================
[ Entry 1055 ]
===========================================
[ Entry 1056 ]
===========================================
[ Entry 1057 ]
===========================================
[ Entry 1058 ]
===========================================
[ Entry 1059 ]
===========================================
[ Entry 1060 ]
===========================================
[ Entry 1061 ]
===========================================
[ Entry 1062 ]
===========================================
[ Entry 1063 ]
===========================================
[ Entry 1064 ]
===========================================
[ Entry 1065 ]
I'm in cluster: 0
===========================================
[ Entry 1066 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1067 ]
I'm in cluster: 0
===========================================
[ Entry 1068 ]
I'm in cluster: 0
===========================================
[ Entry 1069 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1070 ]
I'm in cluster: 0
===========================================
[ Entry 1071 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1072 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1073 ]
===========================================
[ Entry 1074 ]
I'm in cluster: 0
===========================================
[ Entry 1075 ]
===========================================
[ Entry 1076 ]
I'm in cluster: 0
===========================================
[ Entry 1077 ]
===========================================
[ Entry 1078 ]
===========================================
[ Entry 1079 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1080 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1081 ]
===========================================
[ Entry 1082 ]
===========================================
[ Entry 1083 ]
I'm in cluster: 0
===========================================
[ Entry 1084 ]
===========================================
[ Entry 1085 ]
I'm in cluster: 0
===========================================
[ Entry 1086 ]
===========================================
[ Entry 1087 ]
I'm in cluster: 0
===========================================
[ Entry 1088 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1089 ]
===========================================
[ Entry 1090 ]
===========================================
[ Entry 1091 ]
I'm in cluster: 0
===========================================
[ Entry 1092 ]
===========================================
[ Entry 1093 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1094 ]
I'm in cluster: 0
===========================================
[ Entry 1095 ]
I'm in cluster: 0
===========================================
[ Entry 1096 ]
===========================================
[ Entry 1097 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1098 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1099 ]
===========================================
[ Entry 1100 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1101 ]
===========================================
[ Entry 1102 ]
I'm in cluster: 0
===========================================
[ Entry 1103 ]
I'm in cluster: 0
===========================================
[ Entry 1104 ]
===========================================
[ Entry 1105 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1106 ]
I'm in cluster: 0
===========================================
[ Entry 1107 ]
I'm in cluster: 0
===========================================
[ Entry 1108 ]
I'm in cluster: 0
===========================================
[ Entry 1109 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1110 ]
I'm in cluster: 0
===========================================
[ Entry 1111 ]
I'm in cluster: 0
===========================================
[ Entry 1112 ]
===========================================
[ Entry 1113 ]
I'm in cluster: 0
===========================================
[ Entry 1114 ]
===========================================
[ Entry 1115 ]
I'm in cluster: 0
===========================================
[ Entry 1116 ]
===========================================
[ Entry 1117 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1118 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1119 ]
I'm in cluster: 0
===========================================
[ Entry 1120 ]
I'm in cluster: 0
===========================================
[ Entry 1121 ]
===========================================
[ Entry 1122 ]
===========================================
[ Entry 1123 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1124 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1125 ]
===========================================
[ Entry 1126 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1127 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1128 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1129 ]
I'm in cluster: 0
===========================================
[ Entry 1130 ]
===========================================
[ Entry 1131 ]
I'm in cluster: 0
===========================================
[ Entry 1132 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1133 ]
===========================================
[ Entry 1134 ]
I'm in cluster: 0
===========================================
[ Entry 1135 ]
===========================================
[ Entry 1136 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1137 ]
===========================================
[ Entry 1138 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1139 ]
===========================================
[ Entry 1140 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1141 ]
I'm in cluster: 0
===========================================
[ Entry 1142 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1143 ]
I'm in cluster: 0
===========================================
[ Entry 1144 ]
I'm in cluster: 0
===========================================
[ Entry 1145 ]
I'm in cluster: 0
===========================================
[ Entry 1146 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1147 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1148 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1149 ]
===========================================
[ Entry 1150 ]
I'm in cluster: 0
===========================================
[ Entry 1151 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1152 ]
I'm in cluster: 0
===========================================
[ Entry 1153 ]
===========================================
[ Entry 1154 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1155 ]
===========================================
[ Entry 1156 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1157 ]
===========================================
[ Entry 1158 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1159 ]
I'm in cluster: 0
===========================================
[ Entry 1160 ]
I'm in cluster: 0
===========================================
[ Entry 1161 ]
===========================================
[ Entry 1162 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1163 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1164 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1165 ]
===========================================
[ Entry 1166 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1167 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1168 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1169 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1170 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1171 ]
I'm in cluster: 0
===========================================
[ Entry 1172 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1173 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1174 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1175 ]
===========================================
[ Entry 1176 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1177 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1178 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1179 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1180 ]
===========================================
[ Entry 1181 ]
===========================================
[ Entry 1182 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1183 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1184 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1185 ]
I'm in cluster: 0
===========================================
[ Entry 1186 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1187 ]
I'm in cluster: 0
===========================================
[ Entry 1188 ]
I'm in cluster: 0
===========================================
[ Entry 1189 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1190 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1191 ]
===========================================
[ Entry 1192 ]
I'm in cluster: 0
===========================================
[ Entry 1193 ]
===========================================
[ Entry 1194 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1195 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1196 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1197 ]
===========================================
[ Entry 1198 ]
I'm in cluster: 0
===========================================
[ Entry 1199 ]
===========================================
[ Entry 1200 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1201 ]
I'm in cluster: 0
===========================================
[ Entry 1202 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
I'm in cluster: 5
===========================================
[ Entry 1203 ]
I'm in cluster: 0
===========================================
[ Entry 1204 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1205 ]
===========================================
[ Entry 1206 ]
I'm in cluster: 0
===========================================
[ Entry 1207 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1208 ]
I'm in cluster: 0
===========================================
[ Entry 1209 ]
===========================================
[ Entry 1210 ]
===========================================
[ Entry 1211 ]
I'm in cluster: 0
===========================================
[ Entry 1212 ]
I'm in cluster: 0
===========================================
[ Entry 1213 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1214 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1215 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1216 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1217 ]
===========================================
[ Entry 1218 ]
===========================================
[ Entry 1219 ]
I'm in cluster: 0
===========================================
[ Entry 1220 ]
===========================================
[ Entry 1221 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1222 ]
I'm in cluster: 0
===========================================
[ Entry 1223 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1224 ]
I'm in cluster: 0
===========================================
[ Entry 1225 ]
===========================================
[ Entry 1226 ]
I'm in cluster: 0
===========================================
[ Entry 1227 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1228 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1229 ]
===========================================
[ Entry 1230 ]
I'm in cluster: 0
===========================================
[ Entry 1231 ]
I'm in cluster: 0
===========================================
[ Entry 1232 ]
===========================================
[ Entry 1233 ]
===========================================
[ Entry 1234 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1235 ]
===========================================
[ Entry 1236 ]
===========================================
[ Entry 1237 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1238 ]
===========================================
[ Entry 1239 ]
I'm in cluster: 0
===========================================
[ Entry 1240 ]
===========================================
[ Entry 1241 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1242 ]
===========================================
[ Entry 1243 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1244 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1245 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1246 ]
===========================================
[ Entry 1247 ]
===========================================
[ Entry 1248 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 1249 ]
I'm in cluster: 0
===========================================
[ Entry 1250 ]
I'm in cluster: 0
===========================================
[ Entry 1251 ]
I'm in cluster: 0
===========================================
[ Entry 1252 ]
===========================================
[ Entry 1253 ]
===========================================
[ Entry 1254 ]
I'm in cluster: 0
===========================================
[ Entry 1255 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1256 ]
===========================================
[ Entry 1257 ]
===========================================
[ Entry 1258 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1259 ]
I'm in cluster: 0
===========================================
[ Entry 1260 ]
===========================================
[ Entry 1261 ]
I'm in cluster: 0
===========================================
[ Entry 1262 ]
I'm in cluster: 0
===========================================
[ Entry 1263 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1264 ]
===========================================
[ Entry 1265 ]
===========================================
[ Entry 1266 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1267 ]
===========================================
[ Entry 1268 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1269 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1270 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1271 ]
I'm in cluster: 0
===========================================
[ Entry 1272 ]
I'm in cluster: 0
===========================================
[ Entry 1273 ]
I'm in cluster: 0
===========================================
[ Entry 1274 ]
I'm in cluster: 0
===========================================
[ Entry 1275 ]
I'm in cluster: 0
===========================================
[ Entry 1276 ]
===========================================
[ Entry 1277 ]
===========================================
[ Entry 1278 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1279 ]
I'm in cluster: 0
===========================================
[ Entry 1280 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1281 ]
I'm in cluster: 0
===========================================
[ Entry 1282 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1283 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1284 ]
===========================================
[ Entry 1285 ]
===========================================
[ Entry 1286 ]
===========================================
[ Entry 1287 ]
===========================================
[ Entry 1288 ]
===========================================
[ Entry 1289 ]
I'm in cluster: 0
===========================================
[ Entry 1290 ]
===========================================
[ Entry 1291 ]
===========================================
[ Entry 1292 ]
===========================================
[ Entry 1293 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 1294 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1295 ]
I'm in cluster: 0
===========================================
[ Entry 1296 ]
===========================================
[ Entry 1297 ]
I'm in cluster: 0
===========================================
[ Entry 1298 ]
I'm in cluster: 0
===========================================
[ Entry 1299 ]
I'm in cluster: 0
===========================================
[ Entry 1300 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1301 ]
I'm in cluster: 0
===========================================
[ Entry 1302 ]
I'm in cluster: 0
===========================================
[ Entry 1303 ]
I'm in cluster: 0
===========================================
[ Entry 1304 ]
I'm in cluster: 0
===========================================
[ Entry 1305 ]
===========================================
[ Entry 1306 ]
I'm in cluster: 0
===========================================
[ Entry 1307 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1308 ]
I'm in cluster: 0
===========================================
[ Entry 1309 ]
===========================================
[ Entry 1310 ]
===========================================
[ Entry 1311 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1312 ]
I'm in cluster: 0
===========================================
[ Entry 1313 ]
===========================================
[ Entry 1314 ]
===========================================
[ Entry 1315 ]
===========================================
[ Entry 1316 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1317 ]
===========================================
[ Entry 1318 ]
I'm in cluster: 0
abratenko@trex:~/ubdl/larflow/larflow/Ana$ ls -lrt
total 31808
-rw-rw-r-- 1 abratenko abratenko     217 Aug  4 17:45 README.md
-rw-rw-r-- 1 abratenko abratenko   13937 Aug  4 17:45 kpsreco_vertexana.cxx
-rw-rw-r-- 1 abratenko abratenko    3533 Aug  4 17:45 keypoint_truthana.cxx
-rw-rw-r-- 1 abratenko abratenko   12714 Aug  4 17:45 keypoint_recoana.cxx
-rwxrwxr-x 1 abratenko abratenko 1476240 Aug  7 13:55 kpsreco_vertexana
-rw-r--r-- 1 abratenko abratenko     411 Aug  7 14:37 crt_0-0.root
-rw-r--r-- 1 abratenko abratenko     411 Aug 11 12:48 crt_0-2.root
-rw-r--r-- 1 abratenko abratenko  598080 Aug 13 13:10 crt_0-29.root
-rw-rw-r-- 1 abratenko abratenko    5089 Aug 17 15:27 CRTana.cxx~
-rw-rw-r-- 1 abratenko abratenko     766 Aug 17 15:49 GNUmakefile
-rw-rw-r-- 1 abratenko abratenko    1125 Aug 17 17:09 CRTvoxelHits.py~
-rw-rw-r-- 1 abratenko abratenko    1125 Aug 17 17:10 CRTvoxelHits.py
-rw-r--r-- 1 abratenko abratenko    5308 Aug 17 17:10 hitsPerVoxel.root
-rw-r--r-- 1 abratenko abratenko 6025901 Aug 17 17:16 crt_0-9.root
-rw-r--r-- 1 abratenko abratenko 7115474 Aug 21 10:55 crt_0-1318_10cmVoxels.root
drwxrwxr-x 2 abratenko abratenko    4096 Aug 21 11:09 oldBadPlots
-rw-r--r-- 1 abratenko abratenko 7655417 Aug 21 11:19 crt_0-1318_5cmVoxels.root
-rw-rw-r-- 1 abratenko abratenko    5260 Aug 21 11:23 CRTana.cxx
-rwxrwxr-x 1 abratenko abratenko  924328 Aug 21 11:23 CRTana
-rw-r--r-- 1 abratenko abratenko 8668749 Aug 21 11:24 crt_0-1318.root
abratenko@trex:~/ubdl/larflow/larflow/Ana$ mv crt_0-1318.root crt_0-1318_3cmVoxels.root
abratenko@trex:~/ubdl/larflow/larflow/Ana$ root -l crt_0-1318_3cmVoxels.root
root [0] 
Attaching file crt_0-1318_3cmVoxels.root as _file0...
(TFile *) 0x5603bf4381f0
root [1] .ls
TFile**		crt_0-1318_3cmVoxels.root	
 TFile*		crt_0-1318_3cmVoxels.root	
  KEY: TTree	tree;1	tree of hits per voxel
  KEY: TH1D	hitcount_wire_hist_U;1	wire #
  KEY: TH1D	hitcount_wire_hist_V;1	wire #
  KEY: TH1D	hitcount_wire_hist_Y;1	wire #
  KEY: TH2D	hitcount_wire_th2d_U;1	wire ; tick
  KEY: TH2D	hitcount_wire_th2d_V;1	wire ; tick
  KEY: TH2D	hitcount_wire_th2d_Y;1	wire ; tick
  KEY: TH1D	hitcount_xyz_hist_x;1	
  KEY: TH1D	hitcount_xyz_hist_y;1	
  KEY: TH1D	hitcount_xyz_hist_z;1	
  KEY: TH2D	hitcount_xyz_th2d_xy;1	
  KEY: TH2D	hitcount_xyz_th2d_zy;1	
  KEY: TH2D	hitcount_xyz_th2d_zx;1	
  KEY: TH3D	hitcount_xyz_th3d;1	
root [2] tree->Draw("hitsPerVoxel")
Info in <TCanvas::MakeDefCanvas>:  created default TCanvas with name c1
root [3] tree->Draw("hitsPerVoxel>>htemp(100, 0, 100)")
Info in <TCanvas::MakeDefCanvas>:  created default TCanvas with name c1
root [4] .q
abratenko@trex:~/ubdl/larflow/larflow/Ana$ ls -lrt
total 31808
-rw-rw-r-- 1 abratenko abratenko     217 Aug  4 17:45 README.md
-rw-rw-r-- 1 abratenko abratenko   13937 Aug  4 17:45 kpsreco_vertexana.cxx
-rw-rw-r-- 1 abratenko abratenko    3533 Aug  4 17:45 keypoint_truthana.cxx
-rw-rw-r-- 1 abratenko abratenko   12714 Aug  4 17:45 keypoint_recoana.cxx
-rwxrwxr-x 1 abratenko abratenko 1476240 Aug  7 13:55 kpsreco_vertexana
-rw-r--r-- 1 abratenko abratenko     411 Aug  7 14:37 crt_0-0.root
-rw-r--r-- 1 abratenko abratenko     411 Aug 11 12:48 crt_0-2.root
-rw-r--r-- 1 abratenko abratenko  598080 Aug 13 13:10 crt_0-29.root
-rw-rw-r-- 1 abratenko abratenko    5089 Aug 17 15:27 CRTana.cxx~
-rw-rw-r-- 1 abratenko abratenko     766 Aug 17 15:49 GNUmakefile
-rw-rw-r-- 1 abratenko abratenko    1125 Aug 17 17:09 CRTvoxelHits.py~
-rw-rw-r-- 1 abratenko abratenko    1125 Aug 17 17:10 CRTvoxelHits.py
-rw-r--r-- 1 abratenko abratenko    5308 Aug 17 17:10 hitsPerVoxel.root
-rw-r--r-- 1 abratenko abratenko 6025901 Aug 17 17:16 crt_0-9.root
-rw-r--r-- 1 abratenko abratenko 7115474 Aug 21 10:55 crt_0-1318_10cmVoxels.root
drwxrwxr-x 2 abratenko abratenko    4096 Aug 21 11:09 oldBadPlots
-rw-r--r-- 1 abratenko abratenko 7655417 Aug 21 11:19 crt_0-1318_5cmVoxels.root
-rw-rw-r-- 1 abratenko abratenko    5260 Aug 21 11:23 CRTana.cxx
-rwxrwxr-x 1 abratenko abratenko  924328 Aug 21 11:23 CRTana
-rw-r--r-- 1 abratenko abratenko 8668749 Aug 21 11:24 crt_0-1318_3cmVoxels.root
abratenko@trex:~/ubdl/larflow/larflow/Ana$ e CRTana.cxx
abratenko@trex:~/ubdl/larflow/larflow/Ana$ make
<< compile CRTana >>
g++ -g -fPIC `root-config --cflags` `larlite-config --includes` -I/home/abratenko/ubdl/larlite/../ `larcv-config --includes` `ublarcvapp-config --includes` -I/home/abratenko/ubdl/larflow/build/include  CRTana.cxx -o CRTana -L/home/abratenko/ubdl/larflow/build/lib -lLArFlow_LArFlowConstants -lLArFlow_PrepFlowMatchData -lLArFlow_KeyPoints `ublarcvapp-config --libs` -lLArCVApp_MCTools -lLArCVApp_ubdllee -lLArCVApp_UBWireTool -lLArCVApp_LArliteHandler `larcv-config --libs` -lLArCVCorePyUtil `larlite-config --libs` `root-config --libs`
abratenko@trex:~/ubdl/larflow/larflow/Ana$ ./CRTana ~/data/hadded_crtFiles.root 0 1319
    [NORMAL]  <open> Opening a file in READ mode: /home/abratenko/data/hadded_crtFiles.root
===========================================
[ Entry 0 ]
I'm in cluster: 0
===========================================
[ Entry 1 ]
===========================================
[ Entry 2 ]
I'm in cluster: 0
===========================================
[ Entry 3 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 4 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 5 ]
===========================================
[ Entry 6 ]
I'm in cluster: 0
===========================================
[ Entry 7 ]
I'm in cluster: 0
===========================================
[ Entry 8 ]
===========================================
[ Entry 9 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 10 ]
I'm in cluster: 0
===========================================
[ Entry 11 ]
===========================================
[ Entry 12 ]
===========================================
[ Entry 13 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 14 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 15 ]
===========================================
[ Entry 16 ]
I'm in cluster: 0
===========================================
[ Entry 17 ]
I'm in cluster: 0
===========================================
[ Entry 18 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 19 ]
===========================================
[ Entry 20 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 21 ]
I'm in cluster: 0
===========================================
[ Entry 22 ]
===========================================
[ Entry 23 ]
===========================================
[ Entry 24 ]
I'm in cluster: 0
===========================================
[ Entry 25 ]
===========================================
[ Entry 26 ]
I'm in cluster: 0
===========================================
[ Entry 27 ]
I'm in cluster: 0
===========================================
[ Entry 28 ]
I'm in cluster: 0
===========================================
[ Entry 29 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 30 ]
===========================================
[ Entry 31 ]
===========================================
[ Entry 32 ]
I'm in cluster: 0
===========================================
[ Entry 33 ]
===========================================
[ Entry 34 ]
I'm in cluster: 0
===========================================
[ Entry 35 ]
I'm in cluster: 0
===========================================
[ Entry 36 ]
I'm in cluster: 0
===========================================
[ Entry 37 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 38 ]
===========================================
[ Entry 39 ]
===========================================
[ Entry 40 ]
I'm in cluster: 0
===========================================
[ Entry 41 ]
===========================================
[ Entry 42 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 43 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 44 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 45 ]
===========================================
[ Entry 46 ]
===========================================
[ Entry 47 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 48 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 49 ]
I'm in cluster: 0
===========================================
[ Entry 50 ]
===========================================
[ Entry 51 ]
===========================================
[ Entry 52 ]
===========================================
[ Entry 53 ]
===========================================
[ Entry 54 ]
I'm in cluster: 0
===========================================
[ Entry 55 ]
===========================================
[ Entry 56 ]
===========================================
[ Entry 57 ]
I'm in cluster: 0
===========================================
[ Entry 58 ]
I'm in cluster: 0
===========================================
[ Entry 59 ]
I'm in cluster: 0
===========================================
[ Entry 60 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 61 ]
===========================================
[ Entry 62 ]
===========================================
[ Entry 63 ]
===========================================
[ Entry 64 ]
===========================================
[ Entry 65 ]
I'm in cluster: 0
===========================================
[ Entry 66 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 67 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 68 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 69 ]
===========================================
[ Entry 70 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 71 ]
I'm in cluster: 0
===========================================
[ Entry 72 ]
===========================================
[ Entry 73 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 74 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 75 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 76 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 77 ]
===========================================
[ Entry 78 ]
I'm in cluster: 0
===========================================
[ Entry 79 ]
I'm in cluster: 0
===========================================
[ Entry 80 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 81 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 82 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 83 ]
===========================================
[ Entry 84 ]
I'm in cluster: 0
===========================================
[ Entry 85 ]
I'm in cluster: 0
===========================================
[ Entry 86 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 87 ]
===========================================
[ Entry 88 ]
===========================================
[ Entry 89 ]
I'm in cluster: 0
===========================================
[ Entry 90 ]
I'm in cluster: 0
===========================================
[ Entry 91 ]
===========================================
[ Entry 92 ]
I'm in cluster: 0
===========================================
[ Entry 93 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 94 ]
I'm in cluster: 0
===========================================
[ Entry 95 ]
I'm in cluster: 0
===========================================
[ Entry 96 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 97 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 98 ]
I'm in cluster: 0
===========================================
[ Entry 99 ]
===========================================
[ Entry 100 ]
I'm in cluster: 0
===========================================
[ Entry 101 ]
===========================================
[ Entry 102 ]
I'm in cluster: 0
===========================================
[ Entry 103 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 104 ]
I'm in cluster: 0
===========================================
[ Entry 105 ]
I'm in cluster: 0
===========================================
[ Entry 106 ]
===========================================
[ Entry 107 ]
I'm in cluster: 0
===========================================
[ Entry 108 ]
I'm in cluster: 0
===========================================
[ Entry 109 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 110 ]
I'm in cluster: 0
===========================================
[ Entry 111 ]
I'm in cluster: 0
===========================================
[ Entry 112 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 113 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 114 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 115 ]
===========================================
[ Entry 116 ]
I'm in cluster: 0
===========================================
[ Entry 117 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 118 ]
===========================================
[ Entry 119 ]
I'm in cluster: 0
===========================================
[ Entry 120 ]
===========================================
[ Entry 121 ]
===========================================
[ Entry 122 ]
I'm in cluster: 0
===========================================
[ Entry 123 ]
===========================================
[ Entry 124 ]
I'm in cluster: 0
===========================================
[ Entry 125 ]
===========================================
[ Entry 126 ]
===========================================
[ Entry 127 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 128 ]
===========================================
[ Entry 129 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 130 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 131 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 132 ]
I'm in cluster: 0
===========================================
[ Entry 133 ]
I'm in cluster: 0
===========================================
[ Entry 134 ]
I'm in cluster: 0
===========================================
[ Entry 135 ]
===========================================
[ Entry 136 ]
I'm in cluster: 0
===========================================
[ Entry 137 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 138 ]
===========================================
[ Entry 139 ]
I'm in cluster: 0
===========================================
[ Entry 140 ]
===========================================
[ Entry 141 ]
===========================================
[ Entry 142 ]
I'm in cluster: 0
===========================================
[ Entry 143 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 144 ]
I'm in cluster: 0
===========================================
[ Entry 145 ]
I'm in cluster: 0
===========================================
[ Entry 146 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 147 ]
I'm in cluster: 0
===========================================
[ Entry 148 ]
I'm in cluster: 0
===========================================
[ Entry 149 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 150 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 151 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 152 ]
I'm in cluster: 0
===========================================
[ Entry 153 ]
===========================================
[ Entry 154 ]
===========================================
[ Entry 155 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 156 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 157 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 158 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 159 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 160 ]
I'm in cluster: 0
===========================================
[ Entry 161 ]
I'm in cluster: 0
===========================================
[ Entry 162 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 163 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 164 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 165 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
I'm in cluster: 5
===========================================
[ Entry 166 ]
I'm in cluster: 0
===========================================
[ Entry 167 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 168 ]
I'm in cluster: 0
===========================================
[ Entry 169 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 170 ]
===========================================
[ Entry 171 ]
===========================================
[ Entry 172 ]
I'm in cluster: 0
===========================================
[ Entry 173 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 174 ]
I'm in cluster: 0
===========================================
[ Entry 175 ]
I'm in cluster: 0
===========================================
[ Entry 176 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 177 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 178 ]
===========================================
[ Entry 179 ]
===========================================
[ Entry 180 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 181 ]
I'm in cluster: 0
===========================================
[ Entry 182 ]
===========================================
[ Entry 183 ]
I'm in cluster: 0
===========================================
[ Entry 184 ]
I'm in cluster: 0
===========================================
[ Entry 185 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 186 ]
===========================================
[ Entry 187 ]
===========================================
[ Entry 188 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 189 ]
I'm in cluster: 0
===========================================
[ Entry 190 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 191 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 192 ]
===========================================
[ Entry 193 ]
===========================================
[ Entry 194 ]
===========================================
[ Entry 195 ]
I'm in cluster: 0
===========================================
[ Entry 196 ]
I'm in cluster: 0
===========================================
[ Entry 197 ]
I'm in cluster: 0
===========================================
[ Entry 198 ]
I'm in cluster: 0
===========================================
[ Entry 199 ]
I'm in cluster: 0
===========================================
[ Entry 200 ]
I'm in cluster: 0
===========================================
[ Entry 201 ]
I'm in cluster: 0
===========================================
[ Entry 202 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 203 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 204 ]
I'm in cluster: 0
===========================================
[ Entry 205 ]
===========================================
[ Entry 206 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 207 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 208 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 209 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 210 ]
I'm in cluster: 0
===========================================
[ Entry 211 ]
I'm in cluster: 0
===========================================
[ Entry 212 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 213 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 214 ]
===========================================
[ Entry 215 ]
I'm in cluster: 0
===========================================
[ Entry 216 ]
I'm in cluster: 0
===========================================
[ Entry 217 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 218 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 219 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 220 ]
I'm in cluster: 0
===========================================
[ Entry 221 ]
===========================================
[ Entry 222 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 223 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 224 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 225 ]
===========================================
[ Entry 226 ]
I'm in cluster: 0
===========================================
[ Entry 227 ]
I'm in cluster: 0
===========================================
[ Entry 228 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 229 ]
I'm in cluster: 0
===========================================
[ Entry 230 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 231 ]
===========================================
[ Entry 232 ]
I'm in cluster: 0
===========================================
[ Entry 233 ]
===========================================
[ Entry 234 ]
===========================================
[ Entry 235 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 236 ]
===========================================
[ Entry 237 ]
I'm in cluster: 0
===========================================
[ Entry 238 ]
===========================================
[ Entry 239 ]
I'm in cluster: 0
===========================================
[ Entry 240 ]
I'm in cluster: 0
===========================================
[ Entry 241 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 242 ]
===========================================
[ Entry 243 ]
===========================================
[ Entry 244 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 245 ]
===========================================
[ Entry 246 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 247 ]
I'm in cluster: 0
===========================================
[ Entry 248 ]
===========================================
[ Entry 249 ]
===========================================
[ Entry 250 ]
===========================================
[ Entry 251 ]
===========================================
[ Entry 252 ]
===========================================
[ Entry 253 ]
===========================================
[ Entry 254 ]
===========================================
[ Entry 255 ]
===========================================
[ Entry 256 ]
===========================================
[ Entry 257 ]
===========================================
[ Entry 258 ]
===========================================
[ Entry 259 ]
===========================================
[ Entry 260 ]
===========================================
[ Entry 261 ]
===========================================
[ Entry 262 ]
===========================================
[ Entry 263 ]
===========================================
[ Entry 264 ]
===========================================
[ Entry 265 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 266 ]
I'm in cluster: 0
===========================================
[ Entry 267 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 268 ]
===========================================
[ Entry 269 ]
I'm in cluster: 0
===========================================
[ Entry 270 ]
===========================================
[ Entry 271 ]
===========================================
[ Entry 272 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 273 ]
I'm in cluster: 0
===========================================
[ Entry 274 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 275 ]
===========================================
[ Entry 276 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 277 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 278 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 279 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 280 ]
I'm in cluster: 0
===========================================
[ Entry 281 ]
I'm in cluster: 0
===========================================
[ Entry 282 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 283 ]
===========================================
[ Entry 284 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 285 ]
I'm in cluster: 0
===========================================
[ Entry 286 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 287 ]
===========================================
[ Entry 288 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 289 ]
===========================================
[ Entry 290 ]
I'm in cluster: 0
===========================================
[ Entry 291 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 292 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 293 ]
===========================================
[ Entry 294 ]
===========================================
[ Entry 295 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 296 ]
===========================================
[ Entry 297 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 298 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 299 ]
===========================================
[ Entry 300 ]
===========================================
[ Entry 301 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 302 ]
I'm in cluster: 0
===========================================
[ Entry 303 ]
===========================================
[ Entry 304 ]
===========================================
[ Entry 305 ]
I'm in cluster: 0
===========================================
[ Entry 306 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 307 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 308 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 309 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 310 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 311 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 312 ]
===========================================
[ Entry 313 ]
===========================================
[ Entry 314 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 315 ]
===========================================
[ Entry 316 ]
I'm in cluster: 0
===========================================
[ Entry 317 ]
===========================================
[ Entry 318 ]
===========================================
[ Entry 319 ]
I'm in cluster: 0
===========================================
[ Entry 320 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 321 ]
I'm in cluster: 0
===========================================
[ Entry 322 ]
===========================================
[ Entry 323 ]
I'm in cluster: 0
===========================================
[ Entry 324 ]
I'm in cluster: 0
===========================================
[ Entry 325 ]
I'm in cluster: 0
===========================================
[ Entry 326 ]
===========================================
[ Entry 327 ]
I'm in cluster: 0
===========================================
[ Entry 328 ]
I'm in cluster: 0
===========================================
[ Entry 329 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 330 ]
===========================================
[ Entry 331 ]
===========================================
[ Entry 332 ]
===========================================
[ Entry 333 ]
I'm in cluster: 0
===========================================
[ Entry 334 ]
I'm in cluster: 0
===========================================
[ Entry 335 ]
I'm in cluster: 0
===========================================
[ Entry 336 ]
===========================================
[ Entry 337 ]
I'm in cluster: 0
===========================================
[ Entry 338 ]
===========================================
[ Entry 339 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 340 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 341 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 342 ]
I'm in cluster: 0
===========================================
[ Entry 343 ]
I'm in cluster: 0
===========================================
[ Entry 344 ]
I'm in cluster: 0
===========================================
[ Entry 345 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 346 ]
I'm in cluster: 0
===========================================
[ Entry 347 ]
I'm in cluster: 0
===========================================
[ Entry 348 ]
I'm in cluster: 0
===========================================
[ Entry 349 ]
I'm in cluster: 0
===========================================
[ Entry 350 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 351 ]
I'm in cluster: 0
===========================================
[ Entry 352 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 353 ]
I'm in cluster: 0
===========================================
[ Entry 354 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 355 ]
===========================================
[ Entry 356 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 357 ]
I'm in cluster: 0
===========================================
[ Entry 358 ]
===========================================
[ Entry 359 ]
===========================================
[ Entry 360 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 361 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 362 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 363 ]
I'm in cluster: 0
===========================================
[ Entry 364 ]
I'm in cluster: 0
===========================================
[ Entry 365 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 366 ]
===========================================
[ Entry 367 ]
===========================================
[ Entry 368 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 369 ]
I'm in cluster: 0
===========================================
[ Entry 370 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 371 ]
===========================================
[ Entry 372 ]
I'm in cluster: 0
===========================================
[ Entry 373 ]
I'm in cluster: 0
===========================================
[ Entry 374 ]
I'm in cluster: 0
===========================================
[ Entry 375 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 376 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 377 ]
===========================================
[ Entry 378 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 379 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 380 ]
===========================================
[ Entry 381 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 382 ]
I'm in cluster: 0
===========================================
[ Entry 383 ]
I'm in cluster: 0
===========================================
[ Entry 384 ]
I'm in cluster: 0
===========================================
[ Entry 385 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 386 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 387 ]
I'm in cluster: 0
===========================================
[ Entry 388 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 389 ]
I'm in cluster: 0
===========================================
[ Entry 390 ]
I'm in cluster: 0
===========================================
[ Entry 391 ]
I'm in cluster: 0
===========================================
[ Entry 392 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 393 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 394 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 395 ]
I'm in cluster: 0
===========================================
[ Entry 396 ]
I'm in cluster: 0
===========================================
[ Entry 397 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 398 ]
I'm in cluster: 0
===========================================
[ Entry 399 ]
===========================================
[ Entry 400 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 401 ]
===========================================
[ Entry 402 ]
I'm in cluster: 0
===========================================
[ Entry 403 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 404 ]
I'm in cluster: 0
===========================================
[ Entry 405 ]
===========================================
[ Entry 406 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 407 ]
I'm in cluster: 0
===========================================
[ Entry 408 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 409 ]
I'm in cluster: 0
===========================================
[ Entry 410 ]
I'm in cluster: 0
===========================================
[ Entry 411 ]
===========================================
[ Entry 412 ]
===========================================
[ Entry 413 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 414 ]
===========================================
[ Entry 415 ]
===========================================
[ Entry 416 ]
I'm in cluster: 0
===========================================
[ Entry 417 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 418 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 419 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 420 ]
===========================================
[ Entry 421 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 422 ]
===========================================
[ Entry 423 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 424 ]
===========================================
[ Entry 425 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 426 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 427 ]
I'm in cluster: 0
===========================================
[ Entry 428 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 429 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 430 ]
===========================================
[ Entry 431 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 432 ]
I'm in cluster: 0
===========================================
[ Entry 433 ]
I'm in cluster: 0
===========================================
[ Entry 434 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 435 ]
===========================================
[ Entry 436 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 437 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 438 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 439 ]
===========================================
[ Entry 440 ]
===========================================
[ Entry 441 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 442 ]
===========================================
[ Entry 443 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 444 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 445 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 446 ]
===========================================
[ Entry 447 ]
===========================================
[ Entry 448 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 449 ]
===========================================
[ Entry 450 ]
===========================================
[ Entry 451 ]
===========================================
[ Entry 452 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 453 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 454 ]
===========================================
[ Entry 455 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 456 ]
===========================================
[ Entry 457 ]
===========================================
[ Entry 458 ]
I'm in cluster: 0
===========================================
[ Entry 459 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 460 ]
I'm in cluster: 0
===========================================
[ Entry 461 ]
I'm in cluster: 0
===========================================
[ Entry 462 ]
===========================================
[ Entry 463 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 464 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 465 ]
I'm in cluster: 0
===========================================
[ Entry 466 ]
I'm in cluster: 0
===========================================
[ Entry 467 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 468 ]
I'm in cluster: 0
===========================================
[ Entry 469 ]
===========================================
[ Entry 470 ]
I'm in cluster: 0
===========================================
[ Entry 471 ]
I'm in cluster: 0
===========================================
[ Entry 472 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 473 ]
===========================================
[ Entry 474 ]
I'm in cluster: 0
===========================================
[ Entry 475 ]
===========================================
[ Entry 476 ]
I'm in cluster: 0
===========================================
[ Entry 477 ]
I'm in cluster: 0
===========================================
[ Entry 478 ]
===========================================
[ Entry 479 ]
===========================================
[ Entry 480 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 481 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 482 ]
I'm in cluster: 0
===========================================
[ Entry 483 ]
===========================================
[ Entry 484 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 485 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 486 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 487 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 488 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 489 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 490 ]
I'm in cluster: 0
===========================================
[ Entry 491 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 492 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 493 ]
===========================================
[ Entry 494 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 495 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 496 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 497 ]
I'm in cluster: 0
===========================================
[ Entry 498 ]
I'm in cluster: 0
===========================================
[ Entry 499 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 500 ]
===========================================
[ Entry 501 ]
===========================================
[ Entry 502 ]
I'm in cluster: 0
===========================================
[ Entry 503 ]
===========================================
[ Entry 504 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 505 ]
I'm in cluster: 0
===========================================
[ Entry 506 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 507 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 508 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 509 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 510 ]
===========================================
[ Entry 511 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 512 ]
I'm in cluster: 0
===========================================
[ Entry 513 ]
===========================================
[ Entry 514 ]
I'm in cluster: 0
===========================================
[ Entry 515 ]
I'm in cluster: 0
===========================================
[ Entry 516 ]
I'm in cluster: 0
===========================================
[ Entry 517 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 518 ]
===========================================
[ Entry 519 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 520 ]
I'm in cluster: 0
===========================================
[ Entry 521 ]
I'm in cluster: 0
===========================================
[ Entry 522 ]
===========================================
[ Entry 523 ]
I'm in cluster: 0
===========================================
[ Entry 524 ]
I'm in cluster: 0
===========================================
[ Entry 525 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 526 ]
===========================================
[ Entry 527 ]
===========================================
[ Entry 528 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 529 ]
I'm in cluster: 0
===========================================
[ Entry 530 ]
I'm in cluster: 0
===========================================
[ Entry 531 ]
===========================================
[ Entry 532 ]
I'm in cluster: 0
===========================================
[ Entry 533 ]
I'm in cluster: 0
===========================================
[ Entry 534 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 535 ]
===========================================
[ Entry 536 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 537 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 538 ]
I'm in cluster: 0
===========================================
[ Entry 539 ]
===========================================
[ Entry 540 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 541 ]
I'm in cluster: 0
===========================================
[ Entry 542 ]
I'm in cluster: 0
===========================================
[ Entry 543 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 544 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 545 ]
I'm in cluster: 0
===========================================
[ Entry 546 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 547 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 548 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 549 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 550 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 551 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 552 ]
I'm in cluster: 0
===========================================
[ Entry 553 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 554 ]
===========================================
[ Entry 555 ]
===========================================
[ Entry 556 ]
I'm in cluster: 0
===========================================
[ Entry 557 ]
I'm in cluster: 0
===========================================
[ Entry 558 ]
I'm in cluster: 0
===========================================
[ Entry 559 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 560 ]
I'm in cluster: 0
===========================================
[ Entry 561 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 562 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 563 ]
===========================================
[ Entry 564 ]
I'm in cluster: 0
===========================================
[ Entry 565 ]
I'm in cluster: 0
===========================================
[ Entry 566 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 567 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 568 ]
I'm in cluster: 0
===========================================
[ Entry 569 ]
I'm in cluster: 0
===========================================
[ Entry 570 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 571 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 572 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 573 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 574 ]
I'm in cluster: 0
===========================================
[ Entry 575 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 576 ]
===========================================
[ Entry 577 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 578 ]
I'm in cluster: 0
===========================================
[ Entry 579 ]
===========================================
[ Entry 580 ]
===========================================
[ Entry 581 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 582 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 583 ]
I'm in cluster: 0
===========================================
[ Entry 584 ]
I'm in cluster: 0
===========================================
[ Entry 585 ]
I'm in cluster: 0
===========================================
[ Entry 586 ]
===========================================
[ Entry 587 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 588 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 589 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 590 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 591 ]
===========================================
[ Entry 592 ]
I'm in cluster: 0
===========================================
[ Entry 593 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 594 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 595 ]
===========================================
[ Entry 596 ]
I'm in cluster: 0
===========================================
[ Entry 597 ]
I'm in cluster: 0
===========================================
[ Entry 598 ]
I'm in cluster: 0
===========================================
[ Entry 599 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 600 ]
===========================================
[ Entry 601 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 602 ]
I'm in cluster: 0
===========================================
[ Entry 603 ]
===========================================
[ Entry 604 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 605 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 606 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 607 ]
I'm in cluster: 0
===========================================
[ Entry 608 ]
I'm in cluster: 0
===========================================
[ Entry 609 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 610 ]
I'm in cluster: 0
===========================================
[ Entry 611 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 612 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 613 ]
I'm in cluster: 0
===========================================
[ Entry 614 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 615 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 616 ]
===========================================
[ Entry 617 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 618 ]
===========================================
[ Entry 619 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 620 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 621 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 622 ]
I'm in cluster: 0
===========================================
[ Entry 623 ]
===========================================
[ Entry 624 ]
===========================================
[ Entry 625 ]
I'm in cluster: 0
===========================================
[ Entry 626 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 627 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 628 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 629 ]
===========================================
[ Entry 630 ]
===========================================
[ Entry 631 ]
===========================================
[ Entry 632 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 633 ]
I'm in cluster: 0
===========================================
[ Entry 634 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 635 ]
I'm in cluster: 0
===========================================
[ Entry 636 ]
===========================================
[ Entry 637 ]
I'm in cluster: 0
===========================================
[ Entry 638 ]
I'm in cluster: 0
===========================================
[ Entry 639 ]
===========================================
[ Entry 640 ]
I'm in cluster: 0
===========================================
[ Entry 641 ]
===========================================
[ Entry 642 ]
I'm in cluster: 0
===========================================
[ Entry 643 ]
===========================================
[ Entry 644 ]
===========================================
[ Entry 645 ]
===========================================
[ Entry 646 ]
===========================================
[ Entry 647 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 648 ]
I'm in cluster: 0
===========================================
[ Entry 649 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 650 ]
===========================================
[ Entry 651 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 652 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 653 ]
I'm in cluster: 0
===========================================
[ Entry 654 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 655 ]
===========================================
[ Entry 656 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 657 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 658 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 659 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 660 ]
===========================================
[ Entry 661 ]
I'm in cluster: 0
===========================================
[ Entry 662 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 663 ]
I'm in cluster: 0
===========================================
[ Entry 664 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 665 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 666 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 667 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 668 ]
I'm in cluster: 0
===========================================
[ Entry 669 ]
===========================================
[ Entry 670 ]
===========================================
[ Entry 671 ]
I'm in cluster: 0
===========================================
[ Entry 672 ]
I'm in cluster: 0
===========================================
[ Entry 673 ]
I'm in cluster: 0
===========================================
[ Entry 674 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 675 ]
I'm in cluster: 0
===========================================
[ Entry 676 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 677 ]
===========================================
[ Entry 678 ]
I'm in cluster: 0
===========================================
[ Entry 679 ]
I'm in cluster: 0
===========================================
[ Entry 680 ]
I'm in cluster: 0
===========================================
[ Entry 681 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 682 ]
===========================================
[ Entry 683 ]
I'm in cluster: 0
===========================================
[ Entry 684 ]
I'm in cluster: 0
===========================================
[ Entry 685 ]
===========================================
[ Entry 686 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 687 ]
I'm in cluster: 0
===========================================
[ Entry 688 ]
I'm in cluster: 0
===========================================
[ Entry 689 ]
===========================================
[ Entry 690 ]
===========================================
[ Entry 691 ]
I'm in cluster: 0
===========================================
[ Entry 692 ]
I'm in cluster: 0
===========================================
[ Entry 693 ]
I'm in cluster: 0
===========================================
[ Entry 694 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 695 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 696 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 697 ]
I'm in cluster: 0
===========================================
[ Entry 698 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 699 ]
I'm in cluster: 0
===========================================
[ Entry 700 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 701 ]
===========================================
[ Entry 702 ]
===========================================
[ Entry 703 ]
===========================================
[ Entry 704 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 705 ]
I'm in cluster: 0
===========================================
[ Entry 706 ]
I'm in cluster: 0
===========================================
[ Entry 707 ]
I'm in cluster: 0
===========================================
[ Entry 708 ]
===========================================
[ Entry 709 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 710 ]
I'm in cluster: 0
===========================================
[ Entry 711 ]
I'm in cluster: 0
===========================================
[ Entry 712 ]
I'm in cluster: 0
===========================================
[ Entry 713 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 714 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 715 ]
===========================================
[ Entry 716 ]
I'm in cluster: 0
===========================================
[ Entry 717 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 718 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 719 ]
I'm in cluster: 0
===========================================
[ Entry 720 ]
===========================================
[ Entry 721 ]
I'm in cluster: 0
===========================================
[ Entry 722 ]
I'm in cluster: 0
===========================================
[ Entry 723 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 724 ]
I'm in cluster: 0
===========================================
[ Entry 725 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 726 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 727 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 728 ]
I'm in cluster: 0
===========================================
[ Entry 729 ]
I'm in cluster: 0
===========================================
[ Entry 730 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 731 ]
===========================================
[ Entry 732 ]
I'm in cluster: 0
===========================================
[ Entry 733 ]
===========================================
[ Entry 734 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 735 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 736 ]
===========================================
[ Entry 737 ]
I'm in cluster: 0
===========================================
[ Entry 738 ]
I'm in cluster: 0
===========================================
[ Entry 739 ]
===========================================
[ Entry 740 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 741 ]
I'm in cluster: 0
===========================================
[ Entry 742 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 743 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 744 ]
I'm in cluster: 0
===========================================
[ Entry 745 ]
I'm in cluster: 0
===========================================
[ Entry 746 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 747 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 748 ]
I'm in cluster: 0
===========================================
[ Entry 749 ]
I'm in cluster: 0
===========================================
[ Entry 750 ]
I'm in cluster: 0
===========================================
[ Entry 751 ]
I'm in cluster: 0
===========================================
[ Entry 752 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 753 ]
I'm in cluster: 0
===========================================
[ Entry 754 ]
I'm in cluster: 0
===========================================
[ Entry 755 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 756 ]
I'm in cluster: 0
===========================================
[ Entry 757 ]
===========================================
[ Entry 758 ]
===========================================
[ Entry 759 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 760 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 761 ]
I'm in cluster: 0
===========================================
[ Entry 762 ]
===========================================
[ Entry 763 ]
I'm in cluster: 0
===========================================
[ Entry 764 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 765 ]
I'm in cluster: 0
===========================================
[ Entry 766 ]
I'm in cluster: 0
===========================================
[ Entry 767 ]
I'm in cluster: 0
===========================================
[ Entry 768 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 769 ]
I'm in cluster: 0
===========================================
[ Entry 770 ]
I'm in cluster: 0
===========================================
[ Entry 771 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 772 ]
===========================================
[ Entry 773 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 774 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 775 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 776 ]
===========================================
[ Entry 777 ]
I'm in cluster: 0
===========================================
[ Entry 778 ]
I'm in cluster: 0
===========================================
[ Entry 779 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 780 ]
I'm in cluster: 0
===========================================
[ Entry 781 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 782 ]
I'm in cluster: 0
===========================================
[ Entry 783 ]
I'm in cluster: 0
===========================================
[ Entry 784 ]
I'm in cluster: 0
===========================================
[ Entry 785 ]
I'm in cluster: 0
===========================================
[ Entry 786 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 787 ]
I'm in cluster: 0
===========================================
[ Entry 788 ]
I'm in cluster: 0
===========================================
[ Entry 789 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 790 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
I'm in cluster: 5
===========================================
[ Entry 791 ]
===========================================
[ Entry 792 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 793 ]
I'm in cluster: 0
===========================================
[ Entry 794 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 795 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 796 ]
I'm in cluster: 0
===========================================
[ Entry 797 ]
===========================================
[ Entry 798 ]
I'm in cluster: 0
===========================================
[ Entry 799 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 800 ]
===========================================
[ Entry 801 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 802 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 803 ]
I'm in cluster: 0
===========================================
[ Entry 804 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 805 ]
I'm in cluster: 0
===========================================
[ Entry 806 ]
I'm in cluster: 0
===========================================
[ Entry 807 ]
===========================================
[ Entry 808 ]
I'm in cluster: 0
===========================================
[ Entry 809 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 810 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 811 ]
===========================================
[ Entry 812 ]
===========================================
[ Entry 813 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 814 ]
===========================================
[ Entry 815 ]
I'm in cluster: 0
===========================================
[ Entry 816 ]
===========================================
[ Entry 817 ]
I'm in cluster: 0
===========================================
[ Entry 818 ]
I'm in cluster: 0
===========================================
[ Entry 819 ]
I'm in cluster: 0
===========================================
[ Entry 820 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 821 ]
I'm in cluster: 0
===========================================
[ Entry 822 ]
I'm in cluster: 0
===========================================
[ Entry 823 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 824 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 825 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 826 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 827 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 828 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 829 ]
I'm in cluster: 0
===========================================
[ Entry 830 ]
I'm in cluster: 0
===========================================
[ Entry 831 ]
I'm in cluster: 0
===========================================
[ Entry 832 ]
I'm in cluster: 0
===========================================
[ Entry 833 ]
===========================================
[ Entry 834 ]
===========================================
[ Entry 835 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 836 ]
I'm in cluster: 0
===========================================
[ Entry 837 ]
I'm in cluster: 0
===========================================
[ Entry 838 ]
===========================================
[ Entry 839 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 840 ]
===========================================
[ Entry 841 ]
I'm in cluster: 0
===========================================
[ Entry 842 ]
I'm in cluster: 0
===========================================
[ Entry 843 ]
I'm in cluster: 0
===========================================
[ Entry 844 ]
===========================================
[ Entry 845 ]
===========================================
[ Entry 846 ]
I'm in cluster: 0
===========================================
[ Entry 847 ]
===========================================
[ Entry 848 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 849 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 850 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 851 ]
I'm in cluster: 0
===========================================
[ Entry 852 ]
I'm in cluster: 0
===========================================
[ Entry 853 ]
I'm in cluster: 0
===========================================
[ Entry 854 ]
===========================================
[ Entry 855 ]
===========================================
[ Entry 856 ]
I'm in cluster: 0
===========================================
[ Entry 857 ]
I'm in cluster: 0
===========================================
[ Entry 858 ]
I'm in cluster: 0
===========================================
[ Entry 859 ]
I'm in cluster: 0
===========================================
[ Entry 860 ]
===========================================
[ Entry 861 ]
I'm in cluster: 0
===========================================
[ Entry 862 ]
===========================================
[ Entry 863 ]
===========================================
[ Entry 864 ]
I'm in cluster: 0
===========================================
[ Entry 865 ]
I'm in cluster: 0
===========================================
[ Entry 866 ]
I'm in cluster: 0
===========================================
[ Entry 867 ]
===========================================
[ Entry 868 ]
I'm in cluster: 0
===========================================
[ Entry 869 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 870 ]
===========================================
[ Entry 871 ]
I'm in cluster: 0
===========================================
[ Entry 872 ]
I'm in cluster: 0
===========================================
[ Entry 873 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 874 ]
I'm in cluster: 0
===========================================
[ Entry 875 ]
===========================================
[ Entry 876 ]
I'm in cluster: 0
===========================================
[ Entry 877 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 878 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 879 ]
I'm in cluster: 0
===========================================
[ Entry 880 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 881 ]
I'm in cluster: 0
===========================================
[ Entry 882 ]
I'm in cluster: 0
===========================================
[ Entry 883 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 884 ]
I'm in cluster: 0
===========================================
[ Entry 885 ]
===========================================
[ Entry 886 ]
I'm in cluster: 0
===========================================
[ Entry 887 ]
===========================================
[ Entry 888 ]
I'm in cluster: 0
===========================================
[ Entry 889 ]
===========================================
[ Entry 890 ]
I'm in cluster: 0
===========================================
[ Entry 891 ]
I'm in cluster: 0
===========================================
[ Entry 892 ]
===========================================
[ Entry 893 ]
===========================================
[ Entry 894 ]
I'm in cluster: 0
===========================================
[ Entry 895 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 896 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 897 ]
===========================================
[ Entry 898 ]
I'm in cluster: 0
===========================================
[ Entry 899 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 900 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 901 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 902 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 903 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 904 ]
I'm in cluster: 0
===========================================
[ Entry 905 ]
===========================================
[ Entry 906 ]
===========================================
[ Entry 907 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 908 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 909 ]
I'm in cluster: 0
===========================================
[ Entry 910 ]
===========================================
[ Entry 911 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 912 ]
I'm in cluster: 0
===========================================
[ Entry 913 ]
===========================================
[ Entry 914 ]
I'm in cluster: 0
===========================================
[ Entry 915 ]
I'm in cluster: 0
===========================================
[ Entry 916 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 917 ]
I'm in cluster: 0
===========================================
[ Entry 918 ]
I'm in cluster: 0
===========================================
[ Entry 919 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 920 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 921 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 922 ]
===========================================
[ Entry 923 ]
===========================================
[ Entry 924 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 925 ]
I'm in cluster: 0
===========================================
[ Entry 926 ]
===========================================
[ Entry 927 ]
I'm in cluster: 0
===========================================
[ Entry 928 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 929 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 930 ]
===========================================
[ Entry 931 ]
I'm in cluster: 0
===========================================
[ Entry 932 ]
I'm in cluster: 0
===========================================
[ Entry 933 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 934 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 935 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 936 ]
I'm in cluster: 0
===========================================
[ Entry 937 ]
===========================================
[ Entry 938 ]
I'm in cluster: 0
===========================================
[ Entry 939 ]
===========================================
[ Entry 940 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 941 ]
===========================================
[ Entry 942 ]
===========================================
[ Entry 943 ]
===========================================
[ Entry 944 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 945 ]
===========================================
[ Entry 946 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 947 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 948 ]
===========================================
[ Entry 949 ]
I'm in cluster: 0
===========================================
[ Entry 950 ]
I'm in cluster: 0
===========================================
[ Entry 951 ]
===========================================
[ Entry 952 ]
===========================================
[ Entry 953 ]
===========================================
[ Entry 954 ]
I'm in cluster: 0
===========================================
[ Entry 955 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 956 ]
I'm in cluster: 0
===========================================
[ Entry 957 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 958 ]
I'm in cluster: 0
===========================================
[ Entry 959 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 960 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 961 ]
I'm in cluster: 0
===========================================
[ Entry 962 ]
I'm in cluster: 0
===========================================
[ Entry 963 ]
I'm in cluster: 0
===========================================
[ Entry 964 ]
===========================================
[ Entry 965 ]
I'm in cluster: 0
===========================================
[ Entry 966 ]
===========================================
[ Entry 967 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 968 ]
===========================================
[ Entry 969 ]
I'm in cluster: 0
===========================================
[ Entry 970 ]
I'm in cluster: 0
===========================================
[ Entry 971 ]
I'm in cluster: 0
===========================================
[ Entry 972 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 973 ]
I'm in cluster: 0
===========================================
[ Entry 974 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 975 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 976 ]
I'm in cluster: 0
===========================================
[ Entry 977 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 978 ]
I'm in cluster: 0
===========================================
[ Entry 979 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 980 ]
===========================================
[ Entry 981 ]
I'm in cluster: 0
===========================================
[ Entry 982 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 983 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 984 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 985 ]
I'm in cluster: 0
===========================================
[ Entry 986 ]
I'm in cluster: 0
===========================================
[ Entry 987 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 988 ]
I'm in cluster: 0
===========================================
[ Entry 989 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 990 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 991 ]
I'm in cluster: 0
===========================================
[ Entry 992 ]
I'm in cluster: 0
===========================================
[ Entry 993 ]
I'm in cluster: 0
===========================================
[ Entry 994 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 995 ]
I'm in cluster: 0
===========================================
[ Entry 996 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 997 ]
===========================================
[ Entry 998 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 999 ]
I'm in cluster: 0
===========================================
[ Entry 1000 ]
I'm in cluster: 0
===========================================
[ Entry 1001 ]
I'm in cluster: 0
===========================================
[ Entry 1002 ]
===========================================
[ Entry 1003 ]
===========================================
[ Entry 1004 ]
I'm in cluster: 0
===========================================
[ Entry 1005 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1006 ]
===========================================
[ Entry 1007 ]
===========================================
[ Entry 1008 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1009 ]
I'm in cluster: 0
===========================================
[ Entry 1010 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1011 ]
===========================================
[ Entry 1012 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1013 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1014 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
I'm in cluster: 5
===========================================
[ Entry 1015 ]
I'm in cluster: 0
===========================================
[ Entry 1016 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1017 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1018 ]
===========================================
[ Entry 1019 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1020 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1021 ]
I'm in cluster: 0
===========================================
[ Entry 1022 ]
===========================================
[ Entry 1023 ]
===========================================
[ Entry 1024 ]
===========================================
[ Entry 1025 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1026 ]
===========================================
[ Entry 1027 ]
I'm in cluster: 0
===========================================
[ Entry 1028 ]
I'm in cluster: 0
===========================================
[ Entry 1029 ]
I'm in cluster: 0
===========================================
[ Entry 1030 ]
===========================================
[ Entry 1031 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1032 ]
I'm in cluster: 0
===========================================
[ Entry 1033 ]
===========================================
[ Entry 1034 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1035 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1036 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1037 ]
I'm in cluster: 0
===========================================
[ Entry 1038 ]
I'm in cluster: 0
===========================================
[ Entry 1039 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1040 ]
I'm in cluster: 0
===========================================
[ Entry 1041 ]
===========================================
[ Entry 1042 ]
===========================================
[ Entry 1043 ]
===========================================
[ Entry 1044 ]
===========================================
[ Entry 1045 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1046 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1047 ]
===========================================
[ Entry 1048 ]
===========================================
[ Entry 1049 ]
===========================================
[ Entry 1050 ]
===========================================
[ Entry 1051 ]
===========================================
[ Entry 1052 ]
===========================================
[ Entry 1053 ]
===========================================
[ Entry 1054 ]
I'm in cluster: 0
===========================================
[ Entry 1055 ]
===========================================
[ Entry 1056 ]
===========================================
[ Entry 1057 ]
===========================================
[ Entry 1058 ]
===========================================
[ Entry 1059 ]
===========================================
[ Entry 1060 ]
===========================================
[ Entry 1061 ]
===========================================
[ Entry 1062 ]
===========================================
[ Entry 1063 ]
===========================================
[ Entry 1064 ]
===========================================
[ Entry 1065 ]
I'm in cluster: 0
===========================================
[ Entry 1066 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1067 ]
I'm in cluster: 0
===========================================
[ Entry 1068 ]
I'm in cluster: 0
===========================================
[ Entry 1069 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1070 ]
I'm in cluster: 0
===========================================
[ Entry 1071 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1072 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1073 ]
===========================================
[ Entry 1074 ]
I'm in cluster: 0
===========================================
[ Entry 1075 ]
===========================================
[ Entry 1076 ]
I'm in cluster: 0
===========================================
[ Entry 1077 ]
===========================================
[ Entry 1078 ]
===========================================
[ Entry 1079 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1080 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1081 ]
===========================================
[ Entry 1082 ]
===========================================
[ Entry 1083 ]
I'm in cluster: 0
===========================================
[ Entry 1084 ]
===========================================
[ Entry 1085 ]
I'm in cluster: 0
===========================================
[ Entry 1086 ]
===========================================
[ Entry 1087 ]
I'm in cluster: 0
===========================================
[ Entry 1088 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1089 ]
===========================================
[ Entry 1090 ]
===========================================
[ Entry 1091 ]
I'm in cluster: 0
===========================================
[ Entry 1092 ]
===========================================
[ Entry 1093 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1094 ]
I'm in cluster: 0
===========================================
[ Entry 1095 ]
I'm in cluster: 0
===========================================
[ Entry 1096 ]
===========================================
[ Entry 1097 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1098 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1099 ]
===========================================
[ Entry 1100 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1101 ]
===========================================
[ Entry 1102 ]
I'm in cluster: 0
===========================================
[ Entry 1103 ]
I'm in cluster: 0
===========================================
[ Entry 1104 ]
===========================================
[ Entry 1105 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1106 ]
I'm in cluster: 0
===========================================
[ Entry 1107 ]
I'm in cluster: 0
===========================================
[ Entry 1108 ]
I'm in cluster: 0
===========================================
[ Entry 1109 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1110 ]
I'm in cluster: 0
===========================================
[ Entry 1111 ]
I'm in cluster: 0
===========================================
[ Entry 1112 ]
===========================================
[ Entry 1113 ]
I'm in cluster: 0
===========================================
[ Entry 1114 ]
===========================================
[ Entry 1115 ]
I'm in cluster: 0
===========================================
[ Entry 1116 ]
===========================================
[ Entry 1117 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1118 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1119 ]
I'm in cluster: 0
===========================================
[ Entry 1120 ]
I'm in cluster: 0
===========================================
[ Entry 1121 ]
===========================================
[ Entry 1122 ]
===========================================
[ Entry 1123 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1124 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1125 ]
===========================================
[ Entry 1126 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1127 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1128 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1129 ]
I'm in cluster: 0
===========================================
[ Entry 1130 ]
===========================================
[ Entry 1131 ]
I'm in cluster: 0
===========================================
[ Entry 1132 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1133 ]
===========================================
[ Entry 1134 ]
I'm in cluster: 0
===========================================
[ Entry 1135 ]
===========================================
[ Entry 1136 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1137 ]
===========================================
[ Entry 1138 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1139 ]
===========================================
[ Entry 1140 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1141 ]
I'm in cluster: 0
===========================================
[ Entry 1142 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1143 ]
I'm in cluster: 0
===========================================
[ Entry 1144 ]
I'm in cluster: 0
===========================================
[ Entry 1145 ]
I'm in cluster: 0
===========================================
[ Entry 1146 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1147 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1148 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1149 ]
===========================================
[ Entry 1150 ]
I'm in cluster: 0
===========================================
[ Entry 1151 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1152 ]
I'm in cluster: 0
===========================================
[ Entry 1153 ]
===========================================
[ Entry 1154 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1155 ]
===========================================
[ Entry 1156 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1157 ]
===========================================
[ Entry 1158 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1159 ]
I'm in cluster: 0
===========================================
[ Entry 1160 ]
I'm in cluster: 0
===========================================
[ Entry 1161 ]
===========================================
[ Entry 1162 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1163 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1164 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1165 ]
===========================================
[ Entry 1166 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1167 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1168 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1169 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1170 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1171 ]
I'm in cluster: 0
===========================================
[ Entry 1172 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1173 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1174 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1175 ]
===========================================
[ Entry 1176 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1177 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1178 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1179 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1180 ]
===========================================
[ Entry 1181 ]
===========================================
[ Entry 1182 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1183 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1184 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1185 ]
I'm in cluster: 0
===========================================
[ Entry 1186 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1187 ]
I'm in cluster: 0
===========================================
[ Entry 1188 ]
I'm in cluster: 0
===========================================
[ Entry 1189 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1190 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1191 ]
===========================================
[ Entry 1192 ]
I'm in cluster: 0
===========================================
[ Entry 1193 ]
===========================================
[ Entry 1194 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1195 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1196 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1197 ]
===========================================
[ Entry 1198 ]
I'm in cluster: 0
===========================================
[ Entry 1199 ]
===========================================
[ Entry 1200 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1201 ]
I'm in cluster: 0
===========================================
[ Entry 1202 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
I'm in cluster: 5
===========================================
[ Entry 1203 ]
I'm in cluster: 0
===========================================
[ Entry 1204 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1205 ]
===========================================
[ Entry 1206 ]
I'm in cluster: 0
===========================================
[ Entry 1207 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1208 ]
I'm in cluster: 0
===========================================
[ Entry 1209 ]
===========================================
[ Entry 1210 ]
===========================================
[ Entry 1211 ]
I'm in cluster: 0
===========================================
[ Entry 1212 ]
I'm in cluster: 0
===========================================
[ Entry 1213 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1214 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1215 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1216 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
===========================================
[ Entry 1217 ]
===========================================
[ Entry 1218 ]
===========================================
[ Entry 1219 ]
I'm in cluster: 0
===========================================
[ Entry 1220 ]
===========================================
[ Entry 1221 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1222 ]
I'm in cluster: 0
===========================================
[ Entry 1223 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1224 ]
I'm in cluster: 0
===========================================
[ Entry 1225 ]
===========================================
[ Entry 1226 ]
I'm in cluster: 0
===========================================
[ Entry 1227 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1228 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1229 ]
===========================================
[ Entry 1230 ]
I'm in cluster: 0
===========================================
[ Entry 1231 ]
I'm in cluster: 0
===========================================
[ Entry 1232 ]
===========================================
[ Entry 1233 ]
===========================================
[ Entry 1234 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1235 ]
===========================================
[ Entry 1236 ]
===========================================
[ Entry 1237 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1238 ]
===========================================
[ Entry 1239 ]
I'm in cluster: 0
===========================================
[ Entry 1240 ]
===========================================
[ Entry 1241 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1242 ]
===========================================
[ Entry 1243 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1244 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1245 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1246 ]
===========================================
[ Entry 1247 ]
===========================================
[ Entry 1248 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 1249 ]
I'm in cluster: 0
===========================================
[ Entry 1250 ]
I'm in cluster: 0
===========================================
[ Entry 1251 ]
I'm in cluster: 0
===========================================
[ Entry 1252 ]
===========================================
[ Entry 1253 ]
===========================================
[ Entry 1254 ]
I'm in cluster: 0
===========================================
[ Entry 1255 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1256 ]
===========================================
[ Entry 1257 ]
===========================================
[ Entry 1258 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1259 ]
I'm in cluster: 0
===========================================
[ Entry 1260 ]
===========================================
[ Entry 1261 ]
I'm in cluster: 0
===========================================
[ Entry 1262 ]
I'm in cluster: 0
===========================================
[ Entry 1263 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1264 ]
===========================================
[ Entry 1265 ]
===========================================
[ Entry 1266 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1267 ]
===========================================
[ Entry 1268 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1269 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1270 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1271 ]
I'm in cluster: 0
===========================================
[ Entry 1272 ]
I'm in cluster: 0
===========================================
[ Entry 1273 ]
I'm in cluster: 0
===========================================
[ Entry 1274 ]
I'm in cluster: 0
===========================================
[ Entry 1275 ]
I'm in cluster: 0
===========================================
[ Entry 1276 ]
===========================================
[ Entry 1277 ]
===========================================
[ Entry 1278 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1279 ]
I'm in cluster: 0
===========================================
[ Entry 1280 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1281 ]
I'm in cluster: 0
===========================================
[ Entry 1282 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1283 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1284 ]
===========================================
[ Entry 1285 ]
===========================================
[ Entry 1286 ]
===========================================
[ Entry 1287 ]
===========================================
[ Entry 1288 ]
===========================================
[ Entry 1289 ]
I'm in cluster: 0
===========================================
[ Entry 1290 ]
===========================================
[ Entry 1291 ]
===========================================
[ Entry 1292 ]
===========================================
[ Entry 1293 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
I'm in cluster: 3
I'm in cluster: 4
===========================================
[ Entry 1294 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1295 ]
I'm in cluster: 0
===========================================
[ Entry 1296 ]
===========================================
[ Entry 1297 ]
I'm in cluster: 0
===========================================
[ Entry 1298 ]
I'm in cluster: 0
===========================================
[ Entry 1299 ]
I'm in cluster: 0
===========================================
[ Entry 1300 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1301 ]
I'm in cluster: 0
===========================================
[ Entry 1302 ]
I'm in cluster: 0
===========================================
[ Entry 1303 ]
I'm in cluster: 0
===========================================
[ Entry 1304 ]
I'm in cluster: 0
===========================================
[ Entry 1305 ]
===========================================
[ Entry 1306 ]
I'm in cluster: 0
===========================================
[ Entry 1307 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1308 ]
I'm in cluster: 0
===========================================
[ Entry 1309 ]
===========================================
[ Entry 1310 ]
===========================================
[ Entry 1311 ]
I'm in cluster: 0
I'm in cluster: 1
===========================================
[ Entry 1312 ]
I'm in cluster: 0
===========================================
[ Entry 1313 ]
===========================================
[ Entry 1314 ]
===========================================
[ Entry 1315 ]
===========================================
[ Entry 1316 ]
I'm in cluster: 0
I'm in cluster: 1
I'm in cluster: 2
===========================================
[ Entry 1317 ]
===========================================
[ Entry 1318 ]
I'm in cluster: 0
abratenko@trex:~/ubdl/larflow/larflow/Ana$ ls
crt_0-0.root                crt_0-1318_5cmVoxels.root  crt_0-2.root  CRTana.cxx       CRTvoxelHits.py~   keypoint_recoana.cxx   kpsreco_vertexana.cxx
crt_0-1318_10cmVoxels.root  crt_0-1318.root            crt_0-9.root  CRTana.cxx~      GNUmakefile        keypoint_truthana.cxx  oldBadPlots
crt_0-1318_3cmVoxels.root   crt_0-29.root              CRTana        CRTvoxelHits.py  hitsPerVoxel.root  kpsreco_vertexana      README.md
abratenko@trex:~/ubdl/larflow/larflow/Ana$ mv crt_0-1318.root crt_0-1318_1cmVoxels.root 
abratenko@trex:~/ubdl/larflow/larflow/Ana$ root -l crt_0-1318_1cmVoxels.root 
root [0] 
Attaching file crt_0-1318_1cmVoxels.root as _file0...
(TFile *) 0x565419973550
root [1] .ls
TFile**		crt_0-1318_1cmVoxels.root	
 TFile*		crt_0-1318_1cmVoxels.root	
  KEY: TTree	tree;1	tree of hits per voxel
  KEY: TH1D	hitcount_wire_hist_U;1	wire #
  KEY: TH1D	hitcount_wire_hist_V;1	wire #
  KEY: TH1D	hitcount_wire_hist_Y;1	wire #
  KEY: TH2D	hitcount_wire_th2d_U;1	wire ; tick
  KEY: TH2D	hitcount_wire_th2d_V;1	wire ; tick
  KEY: TH2D	hitcount_wire_th2d_Y;1	wire ; tick
  KEY: TH1D	hitcount_xyz_hist_x;1	
  KEY: TH1D	hitcount_xyz_hist_y;1	
  KEY: TH1D	hitcount_xyz_hist_z;1	
  KEY: TH2D	hitcount_xyz_th2d_xy;1	
  KEY: TH2D	hitcount_xyz_th2d_zy;1	
  KEY: TH2D	hitcount_xyz_th2d_zx;1	
  KEY: TH3D	hitcount_xyz_th3d;1	
root [2] tree->Draw("hitsPerVoxel")
Info in <TCanvas::MakeDefCanvas>:  created default TCanvas with name c1
root [3] tree->Draw("hitsPerVoxel>>htemp(100, 0, 100)")
Info in <TCanvas::MakeDefCanvas>:  created default TCanvas with name c1
root [4] .q
abratenko@trex:~/ubdl/larflow/larflow/Ana$ root -l crt_0-1318_1cmVoxels.root 
root [0] 
Attaching file crt_0-1318_1cmVoxels.root as _file0...
(TFile *) 0x55abf3f9d0c0
root [1] tree->Draw("hitsPerVoxel>>htemp(100, 0, 100)")
Info in <TCanvas::MakeDefCanvas>:  created default TCanvas with name c1
root [2] .q
abratenko@trex:~/ubdl/larflow/larflow/Ana$ git status
On branch polarflow
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   CRTana.cxx
	modified:   ../../larmatchnet/ana/3plane_truthana_larmatch.cxx
	modified:   ../../larmatchnet/ana/GNUmakefile

Untracked files:
  (use "git add <file>..." to include in what will be committed)

	../../cilantro/
	CRTana
	CRTvoxelHits.py
	../KeyPoints/#LoaderKeypointData.cxx#
	../PrepFlowMatchData/test/readimg.cxx
	../../larmatchnet/ana/GNUmakefile_newer_masterVer
	../../larmatchnet/runs/
	../../postprocessor/
	../../sparse_larflow/

no changes added to commit (use "git add" and/or "git commit -a")
abratenko@trex:~/ubdl/larflow/larflow/Ana$ git add CRTana.cxx 
abratenko@trex:~/ubdl/larflow/larflow/Ana$ git status
On branch polarflow
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

	modified:   CRTana.cxx

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   ../../larmatchnet/ana/3plane_truthana_larmatch.cxx
	modified:   ../../larmatchnet/ana/GNUmakefile

Untracked files:
  (use "git add <file>..." to include in what will be committed)

	../../cilantro/
	CRTana
	CRTvoxelHits.py
	../KeyPoints/#LoaderKeypointData.cxx#
	../PrepFlowMatchData/test/readimg.cxx
	../../larmatchnet/ana/GNUmakefile_newer_masterVer
	../../larmatchnet/runs/
	../../postprocessor/
	../../sparse_larflow/

abratenko@trex:~/ubdl/larflow/larflow/Ana$ git commit -m "Fixed: now grabs hits from clusters only (1x1x1 cm^3)"
[polarflow e657c91] Fixed: now grabs hits from clusters only (1x1x1 cm^3)
 1 file changed, 7 insertions(+), 7 deletions(-)
abratenko@trex:~/ubdl/larflow/larflow/Ana$ git push origin polarflow
Username for 'https://github.com': polabr
Password for 'https://polabr@github.com': 
Counting objects: 5, done.
Delta compression using up to 20 threads.
Compressing objects: 100% (5/5), done.
Writing objects: 100% (5/5), 553 bytes | 553.00 KiB/s, done.
Total 5 (delta 4), reused 0 (delta 0)
remote: Resolving deltas: 100% (4/4), completed with 4 local objects.
remote: This repository moved. Please use the new location:
remote:   https://github.com/NuTufts/larflow.git
To https://github.com/nutufts/larflow
   c00a327..e657c91  polarflow -> polarflow
abratenko@trex:~/ubdl/larflow/larflow/Ana$ root -l crt_0-1318_3cmVoxels.root 
root [0] 
Attaching file crt_0-1318_3cmVoxels.root as _file0...
(TFile *) 0x55e4d91f44b0
root [1] .ls
TFile**		crt_0-1318_3cmVoxels.root	
 TFile*		crt_0-1318_3cmVoxels.root	
  KEY: TTree	tree;1	tree of hits per voxel
  KEY: TH1D	hitcount_wire_hist_U;1	wire #
  KEY: TH1D	hitcount_wire_hist_V;1	wire #
  KEY: TH1D	hitcount_wire_hist_Y;1	wire #
  KEY: TH2D	hitcount_wire_th2d_U;1	wire ; tick
  KEY: TH2D	hitcount_wire_th2d_V;1	wire ; tick
  KEY: TH2D	hitcount_wire_th2d_Y;1	wire ; tick
  KEY: TH1D	hitcount_xyz_hist_x;1	
  KEY: TH1D	hitcount_xyz_hist_y;1	
  KEY: TH1D	hitcount_xyz_hist_z;1	
  KEY: TH2D	hitcount_xyz_th2d_xy;1	
  KEY: TH2D	hitcount_xyz_th2d_zy;1	
  KEY: TH2D	hitcount_xyz_th2d_zx;1	
  KEY: TH3D	hitcount_xyz_th3d;1	
root [2] tree->Draw("hitsPerVoxel>>htemp(100, 0, 100)")
Info in <TCanvas::MakeDefCanvas>:  created default TCanvas with name c1
root [3] tree->Draw("hitsPerVoxel>>htemp(20, 0, 20)")
root [4] tree->Draw("hitsPerVoxel>>htemp(12, 0, 12)")
root [5] .q
abratenko@trex:~/ubdl/larflow/larflow/Ana$ ls
crt_0-0.root                crt_0-1318_3cmVoxels.root  crt_0-2.root  CRTana.cxx       CRTvoxelHits.py~   keypoint_recoana.cxx   kpsreco_vertexana.cxx
crt_0-1318_10cmVoxels.root  crt_0-1318_5cmVoxels.root  crt_0-9.root  CRTana.cxx~      GNUmakefile        keypoint_truthana.cxx  oldBadPlots
crt_0-1318_1cmVoxels.root   crt_0-29.root              CRTana        CRTvoxelHits.py  hitsPerVoxel.root  kpsreco_vertexana      README.md
abratenko@trex:~/ubdl/larflow/larflow/Ana$ e crt_0-1318_1cmVoxels.root 
abratenko@trex:~/ubdl/larflow/larflow/Ana$ e CRTana.cxx
abratenko@trex:~/ubdl/larflow/larflow/Ana$ root -l crt_0-1318_1cmVoxels.root 
root [0] 
Attaching file crt_0-1318_1cmVoxels.root as _file0...
(TFile *) 0x5622ec076980
root [1] .ls
TFile**		crt_0-1318_1cmVoxels.root	
 TFile*		crt_0-1318_1cmVoxels.root	
  KEY: TTree	tree;1	tree of hits per voxel
  KEY: TH1D	hitcount_wire_hist_U;1	wire #
  KEY: TH1D	hitcount_wire_hist_V;1	wire #
  KEY: TH1D	hitcount_wire_hist_Y;1	wire #
  KEY: TH2D	hitcount_wire_th2d_U;1	wire ; tick
  KEY: TH2D	hitcount_wire_th2d_V;1	wire ; tick
  KEY: TH2D	hitcount_wire_th2d_Y;1	wire ; tick
  KEY: TH1D	hitcount_xyz_hist_x;1	
  KEY: TH1D	hitcount_xyz_hist_y;1	
  KEY: TH1D	hitcount_xyz_hist_z;1	
  KEY: TH2D	hitcount_xyz_th2d_xy;1	
  KEY: TH2D	hitcount_xyz_th2d_zy;1	
  KEY: TH2D	hitcount_xyz_th2d_zx;1	
  KEY: TH3D	hitcount_xyz_th3d;1	
root [2] .q
abratenko@trex:~/ubdl/larflow/larflow/Ana$ ls
crt_0-0.root                crt_0-1318_3cmVoxels.root  crt_0-2.root  CRTana.cxx       CRTvoxelHits.py~   keypoint_recoana.cxx   kpsreco_vertexana.cxx
crt_0-1318_10cmVoxels.root  crt_0-1318_5cmVoxels.root  crt_0-9.root  CRTana.cxx~      GNUmakefile        keypoint_truthana.cxx  oldBadPlots
crt_0-1318_1cmVoxels.root   crt_0-29.root              CRTana        CRTvoxelHits.py  hitsPerVoxel.root  kpsreco_vertexana      README.md
abratenko@trex:~/ubdl/larflow/larflow/Ana$ e CRTana.cxx
abratenko@trex:~/ubdl/larflow/larflow/Ana$ e CRTana.cxx

File Edit Options Buffers Tools C++ Help                                                                                                                       
      // loop thru hits in this cluster                                                                                                                        
      for ( size_t iHit = 0; iHit < cluster.size(); iHit++ ) {

        const larlite::larflow3dhit& lfhit = cluster.at( iHit );

        hit_U = lfhit.targetwire[0];
        hit_V = lfhit.targetwire[1];
        hit_Y = lfhit.targetwire[2];
        tick = lfhit.tick;
        hit_x = lfhit[0];
        hit_y = lfhit[1];
        hit_z = lfhit[2];

        // fill wire hists                                                                                                                                     
        hitcount_wire_hist[0]->Fill(hit_U);
        hitcount_wire_hist[1]->Fill(hit_V);
        hitcount_wire_hist[2]->Fill(hit_Y);

        hitcount_wire_th2d[0]->Fill(hit_U, tick);
        hitcount_wire_th2d[1]->Fill(hit_V, tick);
        hitcount_wire_th2d[2]->Fill(hit_Y, tick);

        // fill xyz hists                                                                                                                                      
        hitcount_xyz_hist[0]->Fill(hit_x);
        hitcount_xyz_hist[1]->Fill(hit_y);
        hitcount_xyz_hist[2]->Fill(hit_z);

        hitcount_xyz_th2d[0]->Fill(hit_x, hit_y);
        hitcount_xyz_th2d[1]->Fill(hit_z, hit_y);
        hitcount_xyz_th2d[2]->Fill(hit_z, hit_x);

	hitcount_xyz_th2d[1]->Fill(hit_z, hit_y);
	hitcount_xyz_th2d[2]->Fill(hit_z, hit_x);

	hitcount_xyz_th3d->Fill(hit_x, hit_y, hit_z);

      }

    }
           
  }

  // Outside event loop
  for (int i = 1; i <= (xyzBins[0]); i++) { // here use i = 1, i <= max, NOT i = 0, i < max (bc bin 0 is underflow in ROOT histograms)
    for (int j = 1; j <= (xyzBins[1]); j++) {
      for (int k = 1; k <= (xyzBins[2]); k++) {
	
	//	std::cout << hitcount_xyz_th3d->GetBinContent(i, j, k) << std::endl;
	hitsPerVoxel = hitcount_xyz_th3d->GetBinContent(i, j, k);
	tree->Fill();

      }
    }
  }
  
  outfile->Write();
  outfile->Close();
  
  llio.close();

  return 0;
}
