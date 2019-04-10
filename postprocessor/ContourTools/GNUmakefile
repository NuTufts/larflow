CXX      =  g++

CFLAGS   =   -g -fPIC
CFLAGS   +=  -DUSE_OPENCV=1

INCLUDES = `root-config --cflags`
INCLUDES += -I$(OPENCV_INCDIR)
INCLUDES += `larlite-config --includes`
INCLUDES += -I$(LARLITE_USERDEVDIR)
INCLUDES += `geo2d-config --includes`
INCLUDES += `laropencv-config --includes`
INCLUDES += `larcv-config --includes`

LDFLAGS  =  `root-config --ldflags` -lstdc++
LIBS     =  `root-config --libs`
LIBS     += -L$(OPENCV_LIBDIR) -lopencv_core
LIBS     += `larlite-config --libs`
LIBS     += `basictool-config --libs`
LIBS     += `geo2d-config --libs`
LIBS     += `laropencv-config --libs`
LIBS     += `larcv-config --libs`

LIB = ../lib/libContourTools.so

SRCS=$(wildcard *.cxx)
OBJS = $(SRCS:.cxx=.o)

all: $(LIB)

$(LIB): $(OBJS)
	@echo "<< linking library "$@" >>"
	$(CXX) -shared $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.cxx %.h
	@echo "<< compiling $< >>"
	$(CXX) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	@echo "<< cleaning>>"
	rm -f $(OBJS) $(LIB)
