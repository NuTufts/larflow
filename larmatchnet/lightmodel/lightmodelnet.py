import sys
sys.path = ['', '/usr/lib/python38.zip', '/usr/lib/python3.8', '/usr/lib/python3.8/lib-dynload', '/usr/local/lib/python3.8/dist-packages', '/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/larflow/python','/usr/local/lib/python3.8/dist-packages/sparseconvnet-0.2-py3.8-linux-x86_64.egg', '/usr/lib/python3/dist-packages', '/usr/local/root/lib', '/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/lardly', '/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/larflow/utils', '/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/larflow/sparse_larflow', '/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/larflow/models', '/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/ublarcvapp/python', '/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/larcv/python', '/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/Geo2D/python', '/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/larlite/python', '/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/LArOpenCV/python']
#sys.path = ['', '/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/lardly', '/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/larflow/python', '/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/larflow/utils', '/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/larflow/sparse_larflow', '/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/larflow/models', '/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/ublarcvapp/python', '/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/larcv/python', '/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/Geo2D/python', '/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/larlite/python', '/usr/local/root/lib', '/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/LArOpenCV/python', '/usr/lib/python38.zip', '/usr/lib/python3.8', '/usr/lib/python3.8/lib-dynload', '/cluster/home/pabrat01/.local/lib/python3.8/site-packages', '/usr/local/lib/python3.8/dist-packages', '/usr/local/lib/python3.8/dist-packages/sparseconvnet-0.2-py3.8-linux-x86_64.egg', '/usr/lib/python3/dist-packages']
import torch
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF

#from lm_dataloader import load_lm_data
from dummyloader import load_data

class LightModelNet(ME.MinkowskiNetwork):

    def __init__(self, in_nchannel, out_nchannel, D):
        super(LightModelNet, self).__init__(D)
        self.block1 = torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_nchannel,
                out_channels=8,
                kernel_size=3,
                stride=1,
                dimension=D),
            ME.MinkowskiBatchNorm(8))

        self.block2 = torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=2,
                dimension=D),
            ME.MinkowskiBatchNorm(16),
        )

        self.block3 = torch.nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                dimension=D),
            ME.MinkowskiBatchNorm(32))

        self.block3_tr = torch.nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,
                dimension=D),
            ME.MinkowskiBatchNorm(16))

        self.block2_tr = torch.nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,
                dimension=D),
            ME.MinkowskiBatchNorm(16))

        self.conv1_tr = ME.MinkowskiConvolution(
            in_channels=24,
            out_channels=out_nchannel,
            kernel_size=1,
            stride=1,
            dimension=D)

    def forward(self, x):

        out_s1 = self.block1(x)

        #print("self.dimension: ", self.block1(x).dimension)
        out = MF.relu(out_s1)

        out_s2 = self.block2(out)
        out = MF.relu(out_s2)

        out_s4 = self.block3(out)
        out = MF.relu(out_s4)

        out = MF.relu(self.block3_tr(out))
        out = ME.cat(out, out_s2)

        out = MF.relu(self.block2_tr(out))
        out = ME.cat(out, out_s1)

        return self.conv1_tr(out)


if __name__ == '__main__':
    # loss and network
    net = LightModelNet(1, 5, D=2)
    print(net)

    ##input_file = "100events_062323_FMDATA_coords_withErrorFlags_100Events.root"
    ##entry = 0

    # a data loader must return a tuple of coords, features, and labels.
    #coords, feat, label = data_loader()
    coords, feat, label = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = net.to(device)
    input = ME.SparseTensor(feat, coords, device=device)

    print("input.shape: ", input.shape)

    print("input.D: ", input.D)
    #print("net.dimension: ", net.dimension)

    # Forward
    output = net(input)
