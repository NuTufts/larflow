CXX      =  g++

CFLAGS   =   -g -fPIC
CFLAGS   += `root-config --cflags` -DUSE_OPENCV=1

INCLUDES = -I$(OPENCV_INCDIR)
INCLUDES += `larlite-config --includes`
INCLUDES += -I$(LARLITE_USERDEVDIR)
INCLUDES += `geo2d-config --includes`
INCLUDES += `laropencv-config --includes`
INCLUDES += `larcv-config --includes`
INCLUDES += `larlitecv-config --includes`
INCLUDES += -I$(LARFLOW_BASEDIR)/postprocessor
INCLUDES += -I$(CILANTRO_INC_DIR)
INCLUDES += -I$(EIGEN_INC_DIR)

LDFLAGS  =  `root-config --ldflags` -lstdc++
LIBS     =  `root-config --libs`
LIBS     += -L$(OPENCV_LIBDIR) -lopencv_core
LIBS     += `larlite-config --libs`
LIBS     += `basictool-config --libs`
LIBS     += `geo2d-config --libs`
LIBS     += `laropencv-config --libs`
LIBS     += `larcv-config --libs`
LIBS     += `larlitecv-config --libs`
LIBS     += -L$(CILANTRO_LIB_DIR) -lcilantro
LIBS     += -L$(LARFLOW_POST_LIBDIR) -lContourTools

# FOR PROGRAMS THAT USE PANGOLIN
INCLUDES_PANGOLIN = -I$(LARFLOW_BASEDIR)/Pangolin/include -I$(LARFLOW_BASEDIR)/Pangolin/build/src/include
LDFLAGS_PANGOLIN  = -L$(LARFLOW_BASEDIR)/Pangolin/build/src -lpangolin

# SPECIFY LIB
LIB = ../lib/libLArFlowCluster3D.so

# SPECIFY SOURCES FOR BINARIES SO THEY ARE OMITTED FROM LIB
BINSRCS = dev_truthcluster.cxx reco_cluster.cxx dev_dbscan.cxx ana_truthcluster.cxx convert2ply.cxx

BINS = $(basename $(BINSRCS))
SRCS = $(filter-out $(addsuffix .cxx,$(BINS)), $(wildcard *.cxx))
OBJS = $(SRCS:.cxx=.o)

all: $(LIB) $(BINS)

../lib/libLArFlowCluster3D.so: $(OBJS)
	$(CXX) -shared $(LDFLAGS) -o $@ $^ $(LIBS)
%.o: %.cxx %.h
	@echo "<< compiling source "$<" >>"
	$(CXX) $(CFLAGS) $(INCLUDES) -c $< -o $@

%: $(LIB) %.cxx
	@echo "<< compiling+linking binary "$@" >>"
	$(CXX) $(CFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $@.cxx -L$(LARFLOW_POST_LIBDIR) -lLArFlowCluster3D $(LIBS)

# dev_truthcluster: $(LIB) dev_truthcluster.cxx
# 	$(CXX) $(CFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $@.cxx -L$(LARFLOW_POST_LIBDIR) -lLArFlowCluster3D $(LIBS)

convert2ply: convert2ply.cxx
	$(CXX) $(CFLAGS) $(INCLUDES) $(INCLUDES_PANGOLIN) $(LDFLAGS) $(LDFLAGS_PANGOLIN) -o $@ $@.cxx $(LIBS)

clean:
	@echo "<< cleaning >>"
	@rm $(LIB) *.o $(BINS)
