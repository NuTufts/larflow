CXX      =  g++

CFLAGS   =   -g -fPIC
CFLAGS   += `root-config --cflags` -DUSE_OPENCV=1

INCLUDES = -I$(OPENCV_INCDIR)
INCLUDES += `larlite-config --includes`
INCLUDES += -I$(LARLITE_USERDEVDIR)
INCLUDES += `geo2d-config --includes`
INCLUDES += `laropencv-config --includes`
INCLUDES += `larcv-config --includes`
INCLUDES += -I$(LARFLOW_BASEDIR)/postprocessor
INCLUDES += -I$(CILANTRO_INC_DIR)
INCLUDES += -I$(EIGEN_INC_DIR)

LDFLAGS  =  `root-config --ldflags` -lstdc++
LIBS     =  `root-config --libs`
LIBS     += -L$(LARFLOW_POST_LIBDIR) -lContourTools
LIBS     += -L$(OPENCV_LIBDIR) -lopencv_core
LIBS     += `larlite-config --libs`
LIBS     += `basictool-config --libs`
LIBS     += `geo2d-config --libs`
LIBS     += `laropencv-config --libs`
LIBS     += `larcv-config --libs` -L$(LARCV_LIBDIR) -lLArCVCoreCVUtil
LIBS     += -L$(CILANTRO_LIB_DIR) -lcilantro


# FOR PROGRAMS THAT USE PANGOLIN
INCLUDES_PANGOLIN = -I$(LARFLOW_BASEDIR)/Pangolin/include -I$(LARFLOW_BASEDIR)/Pangolin/build/src/include
LDFLAGS_PANGOLIN  = -L$(LARFLOW_BASEDIR)/Pangolin/build/src -lpangolin

# SPECIFY LIB
LIB = ../lib/libLArFlowCluster3D.so

# SPECIFY SOURCES FOR BINARIES SO THEY ARE OMITTED FROM LIB
BINSRCS =
BINSRCS += dev_truthcluster.cxx
BINSRCS += dev_ancestorcluster.cxx
BINSRCS += reco_cluster.cxx
BINSRCS += dev_dbscan.cxx
BINSRCS += ana_truthcluster.cxx
BINSRCS += convert2ply.cxx
BINSRCS += dev_fillcluster.cxx
BINSRCS += dev_mrcnn_cluster.cxx

# SPECIFY BINS TO BUILD
BINS = dev_fillcluster dev_truthcluster dev_ancestorcluster dev_mrcnn_cluster

SRCS = $(filter-out $(BINSRCS), $(wildcard *.cxx))
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


convert2ply: convert2ply.cxx
	$(CXX) $(CFLAGS) $(INCLUDES) $(INCLUDES_PANGOLIN) $(LDFLAGS) $(LDFLAGS_PANGOLIN) -o $@ $@.cxx $(LIBS)

clean:
	@echo "<< cleaning cluster module >>"
	@rm -f $(LIB) *.o $(BINS)
