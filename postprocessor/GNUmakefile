CXX      =  g++

CFLAGS   =   -g -fPIC
CFLAGS   += `root-config --cflags`

INCLUDES =  `root-config --incdir`
INCLUDES += `larlite-config --includes`
INCLUDES += `larcv-config --includes`
INCLUDES += `larlite-config --includes`

LDFLAGS  =  `root-config --ldflags` -lstdc++
LIBS     =  `root-config --libs`
LIBS     += `larlite-config --libs`
LIBS     += `larcv-config --libs`
LIBS     += `larlitecv-config --libs`

EXEC = dev


all: $(EXEC)

%: %.cxx
	$(CXX) $(CFLAGS) $(INCLUDES) -c $< -o $@.o
	$(CXX) $(LDFLAGS) -o $@ $@.o $(LIBS)
	rm $@.o
