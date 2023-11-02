import sys
import torch
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF

from lm_dataloader import load_lm_data
#from dummyloader import load_data

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
            kernel_size=3,
            stride=1,
            dimension=D)

    def forward(self, x):

        print("shape of x input: ", x.shape)

        out_s1 = self.block1(x)

        print("self.block1(x) shape: ", out_s1.shape)

        #print("self.dimension: ", self.block1(x).dimension)
        out = MF.relu(out_s1)

        print("MF.relu(out_s1) shape: ", out.shape)

        out_s2 = self.block2(out)
        print("self.block2(out) shape: ", out_s2.shape)
        out = MF.relu(out_s2)
        print("MF.relu(out_s2) shape: ", out.shape)

        out_s4 = self.block3(out)
        print("self.block3(out) shape: ", out_s4.shape)
        out = MF.relu(out_s4)
        print("MF.relu(out_s4) shape: ", out.shape)

        out = MF.relu(self.block3_tr(out))

        print("out = MF.relu(self.block3_tr(out)) shape (after RELU): ", out.shape)

        out = ME.cat(out, out_s2)

        print("out_s2 shape: ", out_s2.shape)
        print("ME.cat(out, out_s2) shape: ", out.shape)

        out = MF.relu(self.block2_tr(out))
        print("MF.relu(self.block2_tr(out)) shape: ", out.shape)
        out = ME.cat(out, out_s1)
        print("ME.cat(out, out_s1) ", out.shape)

        return self.conv1_tr(out)


if __name__ == '__main__':
    # loss and network
    net = LightModelNet(3, 32, D=3) # 3 ADC features go in, 1 feature out (the fudge factor)
    print(net)

    input_file = "100events_062323_FMDATA_coords_withErrorFlags_100Events.root"
    entry = 0

    # a data loader must return a tuple of coords, features, and labels.
    #coords, feat, label = data_loader()
    coords, feat, label = load_lm_data(input_file, entry)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = net.to(device)
    #input = ME.SparseTensor(feat, coords, device=device)
    coords, feats = ME.utils.sparse_collate( [coords], [feat] )
    input = ME.SparseTensor(features=feats, coordinates=coords)

    print("input.shape: ", input.shape)

    print("input.D: ", input.D)
    #print("net.dimension: ", net.dimension)

    # Forward
    output = net(input)

    print("output.shape: ", output.shape)
