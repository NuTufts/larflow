CXX      =  g++

CFLAGS   =   -g -fPIC
CFLAGS   += `root-config --cflags` -DUSE_OPENCV=1

INCLUDES =  -I`root-config --incdir`
INCLUDES += `larlite-config --includes`
INCLUDES += `larcv-config --includes`
INCLUDES += -I${UBLARCVAPP_INCDIR}

LDFLAGS  =  `root-config --ldflags` -lstdc++

#LIBS = ContourTools FlowContourMatching LArFlowCluster3D
LIBS = ContourTools FlowContourMatching
LDLIBS = -Llib $(addprefix -l,$(LIBS))
LDLIBS   +=  `root-config --libs`
LDLIBS   += `larlite-config --libs`
LDLIBS   += `larcv-config --libs` -lLArCVCoreCVUtil -lLArCVCoreProcessor
LDLIBS   += -L${UBLARCVAPP_LIBDIR} -lLArCVApp_UBImageMod


#SUBDIRS = ContourTools FlowContourMatching cluster flashmatch imgstitch
SUBDIRS = ContourTools FlowContourMatching
INCLUDES += $(addprefix -I, $(SUBDIRS))
EXEC = dev
#EXEC += dump2json
#EXEC += ana_flowhits

all: obj $(EXEC) 

%: %.cxx
	$(CXX) $(CFLAGS) $(INCLUDES) -c $< -o $@.o
	$(CXX) $(LDFLAGS) -o $@ $@.o $(LDLIBS)
	rm $@.o

obj:
	@for i in $(SUBDIRS); do ( echo "" && echo "Compiling $$i..." && cd $$i && $(MAKE) ) || exit $$?; done

clean:
	@for i in $(SUBDIRS); do ( echo "" && echo "cleaning $$i..." && cd $$i && $(MAKE) clean ) || exit $$?; done
	@rm -f lib/*.so dev
