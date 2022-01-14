
FMWK_HEADERS = LinkDef.h
#HEADERS = $(filter-out $(FMWK_HEADERS), $(wildcard *.h))

PYTHON = python${LARCV_PYTHON}
INCFLAGS  = -I. #Include itself
INCFLAGS += $(shell larlite-config --includes)
INCFLAGS += $(shell larcv-config --includes)
INCFLAGS += $(shell ${PYTHON}-config --includes)
INCFLAGS += -I$(shell ${PYTHON} -c "import numpy; print(numpy.get_include())")
INCFLAGS += -I${LARFLOW_INCDIR}
#INCFLAGS += -I$(LARCV_COREDIR)

# platform-specific options
LDFLAGS = `root-config --ldflags --libs`
LDFLAGS += `larlite-config --libs`
LDFLAGS += `larcv-config --libs`
LDFLAGS += `ublarcvapp-config --libs`
LDFLAGS = -L${LARFLOW_LIBDIR} -lLArFlow_Reco
LDFLAGS += -L$(shell ${PYTHON}-config --prefix)/lib/
LDFLAGS += $(shell ${PYTHON}-config --ldflags)


#-v2 -f G__LArFlow_LArFlowConstants.cxx -s /home/twongjirad/working/larbys/ubdl/larflow/build/larflow/LArFlowConstants/libLArFlow_LArFlowConstants.so -rml libLArFlow_LArFlowConstants.so -rmf /home/twongjirad/working/larbys/ubdl/larflow/build/larflow/LArFlowConstants/libLArFlow_LArFlowConstants.rootmap -I/home/twongjirad/software/root/build_gcc11/include -I/home/twongjirad/working/larbys/ubdl/larflow -I/home/twongjirad/working/larbys/ubdl/larflow/larflow/LArFlowConstants -I/home/twongjirad/working/larbys/ubdl/larcv/build/installed/include LArFlowConstants.h LArFlowConstants.h /home/twongjirad/working/larbys/ubdl/larflow/larflow/LArFlowConstants/LinkDef.h


all: libSmallClusterRemoval.so

clean:
	rm -f SmallClusterRemoval.o libSmallClusterRemoval.so G__libSmallClusterRemoval*

libSmallClusterRemoval.so: SmallClusterRemoval.o G__libSmallClusterRemoval.o
	g++ -shared -o $@ $^ ${LDFLAGS}

SmallClusterRemoval.o: SmallClusterRemoval.cxx SmallClusterRemoval.h 
	g++ -c SmallClusterRemoval.cxx -I${INCFLAGS} -fPIC -g `root-config --cflags` -o $@

G__libSmallClusterRemoval.o: LinkDef.h SmallClusterRemoval.cxx SmallClusterRemoval.h
	rootcling -v2 -f G__libSmallClusterRemoval.cxx -c $(INCFLAGS) SmallClusterRemoval.h LinkDef.h
	g++ -c G__libSmallClusterRemoval.cxx -I${INCFLAGS} -fPIC -g `root-config --cflags` -o $@


