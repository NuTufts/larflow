import torch.nn as nn
import torch as torch
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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=False)
                     

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.stride = stride

        self.bypass = None
        if inplanes!=planes or stride>1:
            self.bypass = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.bypass is not None:
            outbp = self.bypass(x)
            out += outbp
        else:
            out += x

        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1 ):
        super(Bottleneck, self).__init__()

        # residual path
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
                               
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        # if stride >1, then we need to subsamble the input
        if stride>1:
            self.shortcut = nn.Conv2d(inplanes,planes,kernel_size=1,stride=stride,bias=False)
        else:
            self.shortcut = None
            

    def forward(self, x):

        if self.shortcut is None:
            bypass = x
        else:
            bypass = self.shortcut(x)

        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)

        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.relu(residual)

        residual = self.conv3(residual)
        residual = self.bn3(residual)

        out = bypass+residual
        out = self.relu(out)

        return out

class PreactivationBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1 ):
        super(Preactivation, self).__init__()

        # residual path
        self.bn1   = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, bias=False)

        self.bn2   = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        # if stride >1, then we need to subsamble the input
        if stride>1:
            self.shortcut = nn.Conv2d(inplanes,planes,kernel_size=1,stride=stride,bias=False)
        else:
            self.shortcut = None
            

    def forward(self, x):

        if self.shortcut is None:
            bypass = x
        else:
            bypass = self.shortcut(x)
    
    

class DoubleResNet(nn.Module):
    def __init__(self,Block,inplanes,planes,stride=1,onlyone=False):
        super(DoubleResNet,self).__init__()
        self.res1 = Block(inplanes,planes,stride)
        if not onlyone:
            self.res2 = Block(  planes,planes,     1)
        else:
            self.res2 = None

    def forward(self, x):
        out = self.res1(x)
        if self.res2:
            out = self.res2(out)
        return out
        
    
class ConvTransposeLayer(nn.Module):
    def __init__(self,deconv_inplanes,skip_inplanes,deconv_outplanes,res_outplanes,use_conv=False,onlyoneres=False,stride=2):
        """
        inputs
        ------
        deconv_inplanes: number of channels in input layer to upsample
        skip_inplnes: number of channels from skip connection
        deconv_outplanes: 
        """
        super(ConvTransposeLayer,self).__init__()
        self.use_conv = use_conv
        if use_conv:
            self.deconv = nn.ConvTranspose2d( deconv_inplanes, deconv_outplanes, kernel_size=4, stride=stride, padding=1, bias=False )
            self.res    = DoubleResNet(BasicBlock,deconv_outplanes+skip_inplanes,res_outplanes,stride=1,onlyone=onlyoneres)            
        else:
            if stride>1:
                self.deconv = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
            else:
                self.deconv = None
            self.res    = DoubleResNet(BasicBlock,deconv_inplanes+skip_inplanes,res_outplanes,stride=1,onlyone=onlyoneres)

    #def device(self):
    #    print " deconv: ",self.deconv.device
    #    print " res: ",self.res.device

    def forward(self,x,skip_x):
        if self.deconv:
            if self.use_conv:
                # we upsample using convtranspose
                out = self.deconv(x,output_size=skip_x.size())
            else:
                out = self.deconv(x)
                #out = torch.nn.functional.interpolate(x,skip_x.size(),mode="bilinear",align_corners=False) # 0.4.1
        else:
            out = x
            
        # concat skip connections
        outcat = torch.cat( (out,skip_x), 1 )
        out = self.res(outcat)
        return out

class LArFlowStem(nn.Module):
    
    def __init__(self,input_channels,output_channels,stride=2):
        super(LArFlowStem,self).__init__()
        # one big stem
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=7, stride=1, padding=3, bias=True) # initial conv layer
        self.bn1   = nn.BatchNorm2d(output_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d( 3, stride=stride, padding=1 )

    def forward(self,x):
        x  = self.conv1(x)
        x  = self.bn1(x)
        x0  = self.relu1(x)
        x  = self.pool1(x)
        return x,x0
        
    
class LArFlowUResNet(nn.Module):

    def __init__(self, num_classes=3, input_channels=1,
                 layer_channels=[16,32,64,128,512,1024], layer_strides=[2,2,2,2,2,2],
                 num_final_features=256,
                 use_deconvtranspose=False,
                 onlyone_res=False,
                 showsizes=False, use_visi=True, gpuid1=1, gpuid2=0):

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
            self.encoder_layers.append( DoubleResNet(BasicBlock,inputchannels,outputchannels,stride=layer_stride,onlyone=onlyone_res) )
            self.decoder_nchannels.append( outputchannels )
            self.decoder_strides.append(layer_stride)
            setattr(self,"enc_layer%d"%(ilayer),self.encoder_layers[-1])

        # decoding layers for flow
        self.decoder_nchannels.reverse()
        self.decoder_strides.reverse()
        self.decoder_layers = {"flow1":[],"flow2":[]}
        for n,name in enumerate(["flow1","flow2"]):
            for ilayer,enc_nchannels in enumerate(self.decoder_nchannels):
                if ilayer==0:
                    deconvin  = enc_nchannels*3 # we concat across all planes
                    deconvout = enc_nchannels*3 # maintain same number of channels
                    skipchs   = self.decoder_nchannels[ilayer+1]
                    outchs    = self.decoder_nchannels[ilayer+1]
                elif ilayer>0 and ilayer+1<len(self.decoder_nchannels):
                    deconvin  = enc_nchannels # we concat across all planes
                    deconvout = deconvin
                    skipchs   = self.decoder_nchannels[ilayer+1]
                    outchs    = self.decoder_nchannels[ilayer+1]
                else:
                    deconvin  = enc_nchannels # we concat across all planes
                    deconvout = deconvin      # maintain same number of channels
                    skipchs   = self.stem_noutchannels
                    outchs    = self.num_final_features
                
                layer = ConvTransposeLayer( deconvin, skipchs, deconvout, outchs, use_conv=self.use_deconv, onlyoneres=onlyone_res, stride=self.decoder_strides[ilayer] )
                self.decoder_layers[name].append( layer )
                setattr(self,"%s_layer%d"%(name,ilayer),self.decoder_layers[name][-1])


        # # 1x1 conv for flow
        self.flow_layers = {}
        for n,name in enumerate(["flow1","flow2"]):
            self.flow_layers[name] = nn.Conv2d( self.num_final_features, 1, kernel_size=1, stride=1, padding=0, bias=True )
            setattr(self,"%s_predict"%(name),self.flow_layers[name])
            
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
        
        # stem
        inputname = ("source","target1","target2")
        stem_output    = []
        encoder_output = [ [], [], [] ] # for skip connections
        for n,x in enumerate([source,target1,target2]):
            x,x0  = self.stem(x)
            if self._showsizes:
                print "{} stem/enc-layer0: {}".format(inputname[n],x0.shape)
            if n>0:
                stem_output.append(x0) # only save target output
            else:
                stem_output.append(None) # dont need the source image

            for i,layer in enumerate(self.encoder_layers):
                if n!=0:
                    # for targets, we need to save output for skip-connections
                    if i>0:
                        encoder_output[n].append( layer(encoder_output[n][-1]) ) # only save target output
                    else:
                        encoder_output[n].append( layer( x ) )
                    if self._showsizes:
                        print "{} enc-layer{}: {} {}".format(inputname[n],i+1,encoder_output[n][-1].size(),encoder_output[n][-1].device)
                else:
                    # for source, we can release mem, except for last output
                    if i>0:
                        out = layer( out ) # can overwrite
                    else:
                        out = layer(x)
                    if i+1==len(self.encoder_layers):                        
                        encoder_output[n].append(out)
                    else:
                        encoder_output[n].append(None)
                    if self._showsizes:
                        print "{} enc-layer{}: {} {}".format(inputname[n],i+1,out.size(),out.device)

        # concat last layer
        enc_concat = torch.cat( [ out[-1] for out in encoder_output ], 1 )
        if self.multi_gpu:
            enc_concat = enc_concat.to(device=torch.device("cuda:%d"%(self.gpuid2)))
        if self._showsizes:
            print "enc-concat: ",enc_concat.shape," ",enc_concat.device

        # flow features
        flow_features = {"flow1":None,"flow2":None}
        for n,name in enumerate(["flow1","flow2"]):
            for i,layer in enumerate(self.decoder_layers[name]):
                if i==0:
                    x = layer(enc_concat,encoder_output[n+1][-2-i])
                elif i+1<len(self.decoder_layers[name]):
                    x = layer(x,encoder_output[n+1][-2-i])
                else:
                    x = layer(x,stem_output[n+1])
                    flow_features[name] = x
                if self._showsizes:
                    print "{}-dec-layer{}: {} {}".format(name,i,x.size(),x.device)
            if self._showsizes:
                print "{} feature layer: {} {}".format(name,flow_features[name].size(),x.device)

        # we don't need the features anymore
        #del encoder_output

        # flow prediction regression layer
        flow_predict = {"flow1":None,"flow2":None}
        for n,name in enumerate(["flow1","flow2"]):
            flow_predict[name] = self.flow_layers[name](flow_features[name])
            if self._showsizes:
                print "{} prediction layer: {} {}".format(name,flow_predict[name].size(),flow_predict[name].device)
                    
        return flow_predict["flow1"],flow_predict["flow2"]


if __name__ == "__main__":

    batchsize = 2
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
                            use_deconvtranspose=False,
                            onlyone_res=True,
                            gpuid1=0, gpuid2=1 )

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
