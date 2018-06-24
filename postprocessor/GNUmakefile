CXX      =  g++

CFLAGS   =   -g -fPIC
CFLAGS   += `root-config --cflags` -DUSE_OPENCV=1

INCLUDES =  `root-config --incdir`
INCLUDES += `larlite-config --includes`
INCLUDES += `larcv-config --includes`
INCLUDES += `larlitecv-config --includes`

LDFLAGS  =  `root-config --ldflags` -lstdc++
LIBS     =  `root-config --libs`
LIBS     += `larlite-config --libs`
LIBS     += `larcv-config --libs`
LIBS     += `larlitecv-config --libs`
LIBS     += -Llib -lContourTools

SUBDIRS = ContourTools

EXEC = dev


all: $(EXEC) obj

%: %.cxx
	$(CXX) $(CFLAGS) $(INCLUDES) -c $< -o $@.o
	$(CXX) $(LDFLAGS) -o $@ $@.o $(LIBS)
	rm $@.o

obj:
	@for i in $(SUBDIRS); do ( echo "" && echo "Compiling $$i..." && cd $$i && $(MAKE) ) || exit $$?; done
