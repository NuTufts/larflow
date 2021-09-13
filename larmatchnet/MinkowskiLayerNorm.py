# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import torch
import torch.nn as nn
from torch.nn import Module
from torch.autograd import Function

from MinkowskiSparseTensor import SparseTensor
from MinkowskiTensorField import TensorField

from MinkowskiPooling import MinkowskiGlobalAvgPooling
from MinkowskiBroadcast import (
    MinkowskiBroadcastAddition,
    MinkowskiBroadcastMultiplication,
)
from MinkowskiEngineBackend._C import (
    CoordinateMapKey,
    BroadcastMode,
    PoolingMode,
)
from MinkowskiCoordinateManager import CoordinateManager

from MinkowskiCommon import (
    MinkowskiModuleBase,
    get_minkowski_function,
)


class MinkowskiStableLayerNorm(MinkowskiModuleBase):
    def __init__(self, num_features, num_groups):
        Module.__init__(self)
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = 1e-6
        self.weight = nn.Parameter(torch.ones(1, num_groups))
        self.bias = nn.Parameter(torch.zeros(1, num_groups))

        self.mean_in = MinkowskiGlobalAvgPooling()
        self.glob_sum = MinkowskiBroadcastAddition()
        self.glob_sum2 = MinkowskiBroadcastAddition()
        self.glob_mean = MinkowskiGlobalAvgPooling()
        self.glob_times = MinkowskiBroadcastMultiplication()
        self.reset_parameters()

    def __repr__(self):
        s = f"(nchannels={self.num_features})"
        return self.__class__.__name__ + s

    def reset_parameters(self):
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, x):
        neg_mean_in = self.mean_in(
            SparseTensor(-x.F, coords_key=x.coords_key, coords_manager=x.coords_man)
        )
        centered_in = self.glob_sum(x, neg_mean_in)
        temp = SparseTensor(
            centered_in.F ** 2,
            coordinate_map_key=centered_in.coordinate_map_key,
            coordinate_manager=centered_in.coordinate_manager,
        )
        var_in = self.glob_mean(temp)
        instd_in = SparseTensor(
            1 / (var_in.F + self.eps).sqrt(),
            coordinate_map_key=var_in.coordinate_map_key,
            coordinate_manager=var_in.coordinate_manager,
        )

        x = self.glob_times(self.glob_sum2(x, neg_mean_in), instd_in)
        return SparseTensor(
            x.F * self.weight + self.bias,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )


