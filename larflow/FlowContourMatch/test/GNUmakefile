CXX=g++
CFLAGS=-g -fPIC
INCLUDES =  `root-config --cflags`
INCLUDES += `larcv-config --includes`
INCLUDES += `larlite-config --includes`
INCLUDES += `ublarcvapp-config --includes`
INCLUDES += -I$(LARFLOW_BASEDIR)/build/include

LDLIBS += -L$(LARFLOW_BASEDIR)/build/lib -lLArFlow_FlowContourMatch
LDLIBS = `ublarcvapp-config --libs`
LDLIBS += `larcv-config --libs`
LDLIBS += `larlite-config --libs`
LDLIBS +=  `root-config --libs`
LDLIBS += -lstdc++

all: bin/sparseflow2hits


bin/sparseflow2hits: sparseflow2hits.cxx
	$(CXX) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDLIBS)
