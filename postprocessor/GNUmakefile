CXX      =  g++

CFLAGS   =   -g -fPIC
CFLAGS   += `root-config --cflags` -DUSE_OPENCV=1

INCLUDES =  `root-config --incdir`
INCLUDES += `larlite-config --includes`
INCLUDES += `larcv-config --includes`
INCLUDES += `larlitecv-config --includes`

LDFLAGS  =  `root-config --ldflags` -lstdc++
LDLIBS   =  `root-config --libs`
LDLIBS   += `larlite-config --libs`
LDLIBS   += `larcv-config --libs`
LDLIBS   += `larlitecv-config --libs`

LIBS = ContourTools FlowContourMatching
LDLIBS += -Llib $(addprefix -l,$(LIBS))

SUBDIRS = ContourTools FlowContourMatching
INCLUDES += $(addprefix -I, $(SUBDIRS))
EXEC = dev vis


all: obj $(EXEC) 

%: %.cxx
	$(CXX) $(CFLAGS) $(INCLUDES) -c $< -o $@.o
	$(CXX) $(LDFLAGS) -o $@ $@.o $(LDLIBS)
	rm $@.o

obj:
	@for i in $(SUBDIRS); do ( echo "" && echo "Compiling $$i..." && cd $$i && $(MAKE) ) || exit $$?; done
