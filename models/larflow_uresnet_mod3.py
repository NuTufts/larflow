import torch.nn as nn
import torch as torch
import torch.utils.checkpoint as torchcheck
import math

###########################################################
#
# U-ResNet
# U-net witih ResNet modules
#
# Semantic segmentation network used by MicroBooNE
# to label track/shower pixels
#
# resnet implementation from pytorch.torchvision module
# U-net from (cite)
#
# meant to be copy of caffe version
# 
###########################################################

from common_layers import *
    
class LArFlowUResNet(nn.Module):

    def __init__(self, num_classes=3, input_channels=1,
                 layer_channels=[16,32,64,128,512,1024], layer_strides=[2,2,2,2,2,2],
                 num_final_features=256,
                 use_deconvtranspose=False,
                 use_visi=True,                 
                 onlyone_res=False,
                 showsizes=False,
                 use_grad_checkpoints=False,
                 gpuid1=0, gpuid2=0):

        super(LArFlowUResNet, self).__init__()

        self._showsizes = showsizes # print size at each layer
        self.use_visi = use_visi
        self.gpuid1=gpuid1
        self.gpuid2=gpuid2
        self.multi_gpu = False
        if self.gpuid1!=self.gpuid2:
            self.multi_gpu = True
        self.use_deconv = use_deconvtranspose
        self.num_final_features = num_final_features
        self.redistributed = False
        self.use_grad_checkpoints = use_grad_checkpoints

        # layer specification
        if len(layer_strides)!=len(layer_channels):
            raise ValueError("layer_strides entries has to equal layer_channels entries")
        self.encoder_nchannels = layer_channels        
        self.encoder_strides   = layer_strides
        self.decoder_nchannels = [] # will be specified as we build encoder
        self.decoder_strides   = []
        self.nlayers = len(self.encoder_nchannels)
            
        # Encoder

        # stem
        # one big stem
        self.stem = LArFlowStem( input_channels, layer_channels[0], stride=1 )
        self.stem_noutchannels = layer_channels[0]

        # encoding layers
        self.encoder_layers = []
        self.decoder_nchannels = []
        for ilayer,nchannels in enumerate(self.encoder_nchannels):
            outputchannels = nchannels
            if ilayer==0:
                inputchannels = nchannels
            else:
                inputchannels = self.encoder_nchannels[ilayer-1]
            layer_stride = self.encoder_strides[ilayer]
            #self.encoder_layers.append( DoubleResNet(BasicBlock,inputchannels,outputchannels,stride=layer_stride,onlyone=onlyone_res) )
            layername =  "enc_layer%d"%(ilayer)           
            layer = DoubleResNet(BasicBlock,inputchannels,outputchannels,stride=layer_stride,onlyone=onlyone_res)
            self.encoder_layers.append(layername)
            self.decoder_nchannels.append( outputchannels )
            self.decoder_strides.append(layer_stride)
            setattr(self,layername,layer)

        # decoding layers for flow
        self.decoder_nchannels.reverse()
        self.decoder_strides.reverse()
        self.decoder_layers = {"flow1":[],"flow2":[]}
        for n,name in enumerate(["flow1","flow2"]):
            for ilayer,enc_nchannels in enumerate(self.decoder_nchannels):
                if ilayer==0:
                    deconvin  = enc_nchannels*3 # we concat across all planes
                    deconvout = enc_nchannels*3 # maintain same number of channels
                    skipchs   = self.decoder_nchannels[ilayer+1]*2 # source+target enc features
                    outchs    = self.decoder_nchannels[ilayer+1]
                elif ilayer>0 and ilayer+1<len(self.decoder_nchannels):
                    deconvin  = enc_nchannels # we concat across all planes
                    deconvout = deconvin
                    skipchs   = self.decoder_nchannels[ilayer+1]*2 # source+target enc features
                    outchs    = self.decoder_nchannels[ilayer+1]
                else:
                    deconvin  = enc_nchannels # we concat across all planes
                    deconvout = deconvin      # maintain same number of channels
                    skipchs   = self.stem_noutchannels # stem features
                    outchs    = self.num_final_features

                layername = "%s_layer%d"%(name,ilayer)
                layer = LArFlowUpsampleLayer( deconvin, skipchs, deconvout, outchs, use_conv=self.use_deconv, onlyoneres=onlyone_res, stride=self.decoder_strides[ilayer] )
                #self.decoder_layers[name].append( layer )
                self.decoder_layers[name].append(layername)
                setattr(self,layername,layer)


        # # 1x1 conv for flow
        self.flow_layers = {}
        for n,name in enumerate(["flow1","flow2"]):
            layername = "%s_predict"%(name)
            layer = nn.Conv2d( self.num_final_features, 1, kernel_size=1, stride=1, padding=0, bias=True )
            self.flow_layers[name] = layername
            setattr(self,layername,layer)
            
        # 1x1 conv for mathability
        if self.use_visi:
            self.visi_conv = nn.Conv2d( self.inplanes, 2, kernel_size=1, stride=1, padding=0, bias=True ) # 2 classes, 0=not vis, 1=vis
            self.visi_softmax = nn.LogSoftmax(dim=1)
            self.visi_conv.cuda(self.gpuid2)
            self.visi_softmax.cuda(self.gpuid2)
        
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def redistribute_layers(self):
        """ move some layers to 2nd gpu"""
        print "LArFLowUResNet: redistributing model across GPUs"
        for k,layers in self.decoder_layers.items():
            for i,layer in enumerate(layers):
                layer = layer.to(device=torch.device("cuda:%d"%(self.gpuid2)))
                #print "{}-decode {}".format( (k,i), layer.device )
        for k,layer in self.flow_layers.items():
            layer = layer.to(device=torch.device("cuda:%d"%(self.gpuid2)))
            #print "{}-predict {}".format( k, layer.device )            
        self.redistibuted = True

    def forward(self,source,target1,target2):

        if self.multi_gpu and not self.redistributed:
            self.redistribute_layers()
        if self._showsizes:
            print "larflowuresnet_mod3::forward()"
        
        # ENCODER
        inputname = ("source","target1","target2")
        stem_output    = []
        encoder_output = [ [], [], [] ] # for skip connections
        for n,x in enumerate([source,target1,target2]):

            ## stem
            if not self.use_grad_checkpoints:
                x,x0  = self.stem(x)
            else:
                x,x0  = torchcheck.checkpoint(self.stem,x)
            if self._showsizes:
                print "{} stem/enc-layer0: {} {}".format(inputname[n],x0.shape,x0.device)
            if n>0:
                stem_output.append(x0) # only save target output
            else:
                stem_output.append(None) # dont need the source image

            ## encoder                
            for i,layername in enumerate(self.encoder_layers):
                layer = getattr(self,layername)
                if n!=0:
                    # for targets, we need to save output for skip-connections
                    if i>0:
                        layerin = encoder_output[n][-1]
                    else:
                        layerin = x

                    if not self.use_grad_checkpoints:
                        layerout = layer(layerin)
                    else:
                        layerout = torchcheck.checkpoint( layer, layerin )
                        
                    encoder_output[n].append( layerout )
                    if self._showsizes:
                        print "{} enc-layer{}: {} {}".format(inputname[n],i+1,encoder_output[n][-1].size(),encoder_output[n][-1].device)
                else:
                    # for source, we can release mem, except for last output
                    if i>0:
                        layerin = layerout
                    else:
                        layerin = x
                    if not self.use_grad_checkpoints:
                        layerout = layer(layerin)
                    else:
                        layerout = torchcheck.checkpoint(layer,layerin)
                    if i+1==len(self.encoder_layers):                        
                        encoder_output[n].append(layerout)
                    else:
                        #encoder_output[n].append(None) # don't concat source
                        encoder_output[n].append(layerout) # concat source
                    if self._showsizes:
                        print "{} enc-layer{}: {} {}".format(inputname[n],i+1,layerout.size(),layerout.device)

        # concat last layer
        enc_concat = torch.cat( [ out[-1] for out in encoder_output ], 1 )
        if self.multi_gpu:
            enc_concat = enc_concat.to(device=torch.device("cuda:%d"%(self.gpuid2)))
        if self._showsizes:
            print "enc-concat: ",enc_concat.shape," ",enc_concat.device

        # flow features
        flow_features = {"flow1":None,"flow2":None}
        for n,name in enumerate(["flow1","flow2"]):
            for i,layername in enumerate(self.decoder_layers[name]):
                layer = getattr(self,layername)
                if i==0:
                    xcat = torch.cat( [encoder_output[0][-2-i],encoder_output[1+n][-2-i]], 1 ) # concat source+target enc features                    
                    x = layer(enc_concat,xcat)
                elif i+1<len(self.decoder_layers[name]):
                    xcat = torch.cat( [encoder_output[0][-2-i],encoder_output[1+n][-2-i]], 1 ) # concat source+target enc features                    
                    x = layer(x,xcat)
                else:
                    x = layer(x,stem_output[n+1])
                    flow_features[name] = x
                if self._showsizes:
                    print "{}-dec-layer{}: {} {}".format(name,i,x.size(),x.device)
            if self._showsizes:
                print "{} feature layer: {} {}".format(name,flow_features[name].size(),x.device)

        # we don't need the features anymore
        #del encoder_output

        # flow prediction regression layer, move back to gpu1
        flow_predict = {"flow1":None,"flow2":None}
        for n,name in enumerate(["flow1","flow2"]):
            layer = getattr(self,self.flow_layers[name])
            flow_predict[name] = layer(flow_features[name]).to(device=torch.device("cuda:%d"%(self.gpuid1)))
            if self._showsizes:
                print "{} prediction layer: {} {}".format(name,flow_predict[name].size(),flow_predict[name].device)
            # move back to original gpu
                    
        return flow_predict["flow1"],flow_predict["flow2"]


if __name__ == "__main__":

    batchsize = 1
    ncols = 832
    nrows = 512
    dev = torch.device("cuda:0")
    savegrad = False

    model = LArFlowUResNet( num_classes=3, input_channels=1,
                            layer_channels=[16,32,64,128,256,512],
                            layer_strides=[2,2,2,2,2,2],
                            num_final_features=256,
                            showsizes=True,
                            use_visi=False,
                            use_deconvtranspose=True,
                            onlyone_res=True,
                            gpuid1=0, gpuid2=0 )

    source  = torch.rand( (batchsize,1,ncols,nrows), dtype=torch.float )
    target1 = torch.rand( (batchsize,1,ncols,nrows), dtype=torch.float )
    target2 = torch.rand( (batchsize,1,ncols,nrows), dtype=torch.float )

    model   = model.to(device=dev)
    source  = source.to(device=dev)
    target1 = target1.to(device=dev)
    target2 = target2.to(device=dev)

    #print model
    model.redistribute_layers()

    flow1,flow2 = model(source,target1,target2)
    pars = model.parameters()
    #for par in pars:
    #    print par.device
    
    raw_input()
