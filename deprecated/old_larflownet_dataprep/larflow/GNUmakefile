
# compiler
CXX = g++

CXXFLAGS = -g -fPIC

# include options for this package
INCFLAGS  = -I.
INCFLAGS += $(shell root-config --cflags)
INCFLAGS += $(shell larlite-config --includes)
INCFLAGS += -I$(LARLITE_USERDEVDIR)
INCFLAGS += $(shell larcv-config --includes)
INCFLAGS += $(shell larlitecv-config --includes)

LDFLAGS += $(shell root-config --ldflags --libs)
LDFLAGS += $(shell larlite-config --libs)
LDFLAGS += $(shell larcv-config --libs)
LDFLAGS += $(shell larlitecv-config --libs)

CXXFLAGS += $(INCFLAGS)

# note: llcvprocessor headers and libraries are in larlitecv/build/include and lib
#LDFLAGS += -l

# platform-specific options
OSNAME          = $(shell uname -s)
HOST            = $(shell uname -n)
OSNAMEMODE      = $(OSNAME)

PROGRAMS = crop_wlarcv2
PROGSRC = $(addsuffix .cxx, $(PROGRAMS) )
SRCS = $(filter-out $(PROGSRC),$(wildcard *.cxx))
OBJS = $(addprefix obj/,$(patsubst %.cxx, %.o, $(SRCS)))
BINS  = $(addprefix bin/,$(PROGRAMS))

all: $(OBJS) $(BINS)


clean:
	rm $(BINS)

bin/%: %.cxx $(OBJS)
	$(CXX) $(CXXFLAGS) $*.cxx -o $@ $(OBJS) $(LDFLAGS)

obj/%.o: %.cxx
	$(CXX) $(CXXFLAGS) -c $*.cxx -o $@
