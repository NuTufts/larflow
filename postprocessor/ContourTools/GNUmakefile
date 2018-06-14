CXX      =  g++

CFLAGS   =   -g -fPIC
CFLAGS   += `root-config --cflags` -DUSE_OPENCV=1

INCLUDES =  `root-config --incdir`
INCLUDES += -I$(OPENCV_INCDIR)
INCLUDES += `larlite-config --includes`
INCLUDES += `larcv-config --includes`
INCLUDES += `larlitecv-config --includes`

LDFLAGS  =  `root-config --ldflags` -lstdc++
LIBS     =  `root-config --libs`
LIBS     += -L$(OPENCV_LIBDIR) -lopencv_core
LIBS     += `larlite-config --libs`
LIBS     += `larcv-config --libs`
LIBS     += `larlitecv-config --libs`

LIB = libContourTools.so


all: $(LIB)

libContourTools.so: ContourShapeMeta.o ContourCluster.o
	$(CXX) -shared $(LDFLAGS) -o $@ $^ $(LIBS)
%.o: %.cxx %.h
	@echo "<< compiling $< >>"
	$(CXX) $(CFLAGS) $(INCLUDES) -c $< -o $@

