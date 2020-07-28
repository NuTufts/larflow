CXX=g++
CFLAGS=-g -fPIC
INCLUDES =  `root-config --cflags`
INCLUDES += `larcv-config --includes`
INCLUDES += `larlite-config --includes`
INCLUDES += `ublarcvapp-config --includes`
INCLUDES += -I$(LARFLOW_BASEDIR)/build/include

LDLIBS =
LDLIBS += -L$(LARFLOW_BASEDIR)/build/lib -lLArFlow_Reco
LDLIBS += `ublarcvapp-config --libs` -lLArCVApp_LArOpenCVHandle -lLArCVApp_ContourTools -lLArCVApp_UBWireTool
LDLIBS += `larcv-config --libs`  -lLArCVCoreCVUtil
LDLIBS += `laropencv-config --libs`
LDLIBS += `geo2d-config --libs`
LDLIBS += `larlite-config --libs`
LDLIBS +=  `root-config --libs`
LDLIBS += -lstdc++

all: bin/ana_showercluster

bin/ana_showercluster: ana_showercluster.cxx
	$(CXX) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDLIBS)

clean:
	rm bin/ana_showercluster
