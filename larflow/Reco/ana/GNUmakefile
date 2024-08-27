CXX=g++
CFLAGS=-g -fPIC
INCLUDES =  `root-config --cflags`
INCLUDES += `larlite-config --includes`
INCLUDES += `larcv-config --includes`
INCLUDES += `ublarcvapp-config --includes`
INCLUDES += -I/usr/local/xgboost/include/
INCLUDES += -I$(LARFLOW_BASEDIR)/build/include

LDLIBS =
LDLIBS += -L$(LARFLOW_BASEDIR)/build/lib -lLArFlow_Reco -lLArFlow_SCBoundary 
LDLIBS += `ublarcvapp-config --libs` -lLArCVApp_LArOpenCVHandle -lLArCVApp_ContourTools -lLArCVApp_UBWireTool
LDLIBS += `larcv-config --libs`  -lLArCVCoreCVUtil
LDLIBS += `laropencv-config --libs`
LDLIBS += `geo2d-config --libs`
LDLIBS += `larlite-config --libs`
LDLOBS += /usr/local/xgboost/lib -lxgboost
LDLIBS +=  `root-config --libs`
LDLIBS += -lstdc++

all: bin/debug_kpsreco

bin/ana_showercluster: ana_showercluster.cxx
	$(CXX) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDLIBS)

bin/debug_kpsreco: debug_kpsreco.cxx
	$(CXX) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDLIBS)

clean:
	rm bin/ana_showercluster bin/debug_kpsreco
