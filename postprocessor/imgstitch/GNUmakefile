CXX      =  g++

CFLAGS   =   -g -fPIC
CFLAGS   += `root-config --cflags` -DUSE_OPENCV=1

INCLUDES = `larlite-config --includes`
INCLUDES += -I$(LARLITE_USERDEVDIR)
INCLUDES += `larcv-config --includes`

LDFLAGS  =  `root-config --ldflags` -lstdc++
LIBS     =  `root-config --libs`
LIBS     += `larlite-config --libs`
LIBS     += `basictool-config --libs`

# SPECIFY LIB
LIB = ../lib/libLArFlowImageStitchers.so

# SPECIFY SOURCES FOR BINARIES SO THEY ARE OMITTED FROM LIB
BINSRCS =
BINSRCS += stitch_dlcosmic_images.cxx

# SPECIFY BINS TO BUILD
BINS = $(basename $(BINSRCS)) # for all
#BINS = 

SRCS = $(filter-out $(BINSRCS), $(wildcard *.cxx))
OBJS = $(SRCS:.cxx=.o)

all: $(LIB) $(BINS)

$(LIB): $(OBJS)
	@echo "<< linking library "$@" >>"
	$(CXX) -shared $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.cxx %.h
	@echo "<< compiling source "$<" >>"
	$(CXX) $(CFLAGS) $(INCLUDES) -c $< -o $@

%: $(LIB) %.cxx
	@echo "<< compiling+linking binary "$@" >>"
	$(CXX) $(CFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $@.cxx -L$(LARFLOW_POST_LIBDIR) -lLArFlowCluster3D $(LIBS)

clean:
	@echo "<< cleaning >>"
	@rm $(LIB) *.o $(BINS)
