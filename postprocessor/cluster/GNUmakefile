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

LIB = ../lib/libLArFlowCluster3D.so


all: $(LIB) dev_truthcluster

../lib/libLArFlowCluster3D.so: TruthCluster.o CilantroPCA.o
	$(CXX) -shared $(LDFLAGS) -o $@ $^ $(LIBS)
%.o: %.cxx %.h
	@echo "<< compiling $< >>"
	$(CXX) $(CFLAGS) $(INCLUDES) -c $< -o $@

dev_truthcluster: $(LIB) dev_truthcluster.cxx
	$(CXX) $(CFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $@.cxx $(LIBS) -L$(LARFLOW_POST_LIBDIR) -lLArFlowCluster3D

