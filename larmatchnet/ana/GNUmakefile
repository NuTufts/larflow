CXX=g++

CXXFLAGS=-g -fPIC
CXXFLAGS+= `root-config --cflags`
CXXFLAGS+= `larlite-config --includes`
CXXFLAGS+= -I$(LARLITE_BASEDIR)/../
CXXFLAGS+= `larcv-config --includes`
CXXFLAGS+= `ublarcvapp-config --includes`
CXXFLAGS+= -I$(LARFLOW_INCDIR)

LDFLAGS=-L$(LARFLOW_LIBDIR) -lLArFlow_LArFlowConstants -lLArFlow_PrepFlowMatchData -lLArFlow_KeyPoints
LDFLAGS+=`ublarcvapp-config --libs` -lLArCVApp_MCTools -lLArCVApp_ubdllee -lLArCVApp_UBWireTool -lLArCVApp_LArliteHandler
LDFLAGS+=`larcv-config --libs` -lLArCVCorePyUtil
LDFLAGS+=`larlite-config --libs`
LDFLAGS+=`root-config --libs`


all: 3plane_truthana_larmatch keypoint_truthana keypoint_recoana

%: %.cxx
	@echo "<< compile $@ >>"
	$(CXX) $(CXXFLAGS)  $< -o $@ $(LDFLAGS)

clean:
	@rm truthana_larmatch keypoint_truthana keypoint_recoana
