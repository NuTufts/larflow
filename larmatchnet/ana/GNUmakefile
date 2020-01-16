CXX=g++

CXXFLAGS=-g -fPIC
CXXFLAGS+= `root-config --cflags`
CXXFLAGS+= `larlite-config --includes`
CXXFLAGS+= -I$(LARLITE_BASEDIR)/../
CXXFLAGS+= `larcv-config --includes`
CXXFLAGS+= `ublarcvapp-config --includes`
CXXFLAGS+= -I$(LARFLOW_INCDIR)

LDFLAGS= -L$(LARFLOW_LIBDIR) -lLArFlow_PrepFlowMatchData
LDFLAGS+=`larlite-config --libs`
LDFLAGS+=`larcv-config --libs` -lLArCVCorePyUtil
LDFLAGS+=`ublarcvapp-config --libs`
LDFLAGS+=`root-config --libs`


all: truthana_larmatch

%: %.cxx
	@echo "<< compile $@ >>"
	$(CXX) $(CXXFLAGS)  $< -o $@ $(LDFLAGS)

clean:
	@rm truthana_larmatch
