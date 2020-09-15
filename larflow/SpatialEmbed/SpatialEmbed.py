from collections import OrderedDict
import sys,os
import torch 
import torch.nn as nn
from torch.autograd import Variable
import sparseconvnet as scn
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, os.environ['LARFLOW_BASEDIR']+"/larmatchnet/")
from utils_sparselarflow import create_resnet_layer

sys.path.insert(1, os.environ['LARFLOW_BASEDIR']+"/larflow/SpatialEmbed/LovaszSoftmax/pytorch")
import lovasz_losses

class SpatialEmbed(nn.Module):
    def __init__(self, ndimensions=2,
                 inputshape=(1024, 3456),
                 input_nfeatures=1,
                 stem_nfeatures=16,
                 features_per_layer=16,
                 leakiness=0.01,
                 ninput_planes=3,
                 use_unet=False,
                 nresnet_blocks=10 ):
        
        super(SpatialEmbed, self).__init__()

        self.use_unet = use_unet

        self.ninput_planes = ninput_planes  
              
        # OFFSET
        self.source_inputlayer_offset  = scn.InputLayer(ndimensions,inputshape,mode=0)

            # STEM
        self.stem_offset = scn.Sequential() 
        self.stem_offset.add( scn.SubmanifoldConvolution(ndimensions, input_nfeatures, stem_nfeatures, 3, False ) )
        self.stem_offset.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )

        if self.use_unet:
            self.stem_offset.add( scn.SubmanifoldConvolution(ndimensions, stem_nfeatures, stem_nfeatures, 3, False ) )
            self.stem_offset.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )
            self.stem_offset.add( scn.SubmanifoldConvolution(ndimensions, stem_nfeatures, stem_nfeatures, 3, False ) )

            # UNET BLOCK    
        if self.use_unet:
            self.unet_layers_offset = scn.UNet( 2, 2,
                                         [stem_nfeatures,
                                          stem_nfeatures*2,
                                          stem_nfeatures*3,
                                          stem_nfeatures*4,
                                          stem_nfeatures*5],
                                         residual_blocks=True,
                                         downsample=[2, 2] )

            # RESNET BLOCK
        if not self.use_unet:
            self.resnet_layers_offset = create_resnet_layer(nresnet_blocks, stem_nfeatures, 4, leakiness=leakiness )
        else:
            self.resnet_layers_offset = create_resnet_layer(1,  stem_nfeatures, 4, leakiness=leakiness )
            self.resnet_layers_offset.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )

        # INPUT LAYERS: converts torch tensor into scn.SparseMatrix
        self.source_inputlayer_seed = scn.InputLayer(ndimensions,inputshape,mode=0)


        # SEED
            # STEM
        self.stem_seed = scn.Sequential() 
        self.stem_seed.add( scn.SubmanifoldConvolution(ndimensions, input_nfeatures, stem_nfeatures, 3, False ) )
        self.stem_seed.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )

        if self.use_unet:
            self.stem_seed.add( scn.SubmanifoldConvolution(ndimensions, stem_nfeatures, stem_nfeatures, 3, False ) )
            self.stem_seed.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )
            self.stem_seed.add( scn.SubmanifoldConvolution(ndimensions, stem_nfeatures, stem_nfeatures, 3, False ) )

            # UNET BLOCK
        if self.use_unet:

            self.unet_layers_seed = scn.UNet( 2, 2,
                                         [stem_nfeatures,
                                          stem_nfeatures*2,
                                          stem_nfeatures*3,
                                          stem_nfeatures*4,
                                          stem_nfeatures*5],
                                         residual_blocks=True,
                                         downsample=[2, 2] )
 
            # RESNET BLOCK
        if not self.use_unet:
            self.resnet_layers_seed = create_resnet_layer(nresnet_blocks, stem_nfeatures, features_per_layer, leakiness=leakiness )
        else:
            self.resnet_layers_seed = create_resnet_layer(1,  stem_nfeatures, features_per_layer, leakiness=leakiness )
            self.resnet_layers_seed.add( scn.BatchNormLeakyReLU(stem_nfeatures,leakiness=leakiness) )


        # OUTPUT FEATURES
        self.nfeatures = features_per_layer
        self.feature_layer_offset = scn.SubmanifoldConvolution(ndimensions, 4, 4, 1, True ) # 4 output channels: x, y, sigma_x, sigma_y
        self.feature_layer_seed = scn.SubmanifoldConvolution(ndimensions, features_per_layer, self.nfeatures, 1, True )

        # from there, we move back into a tensor
        self.source_outlayer_offset  = scn.OutputLayer(ndimensions)
        self.source_outlayer_seed = scn.OutputLayer(ndimensions)


    def forward_features( self, coord_plane0_t, plane0_feat_t,
                          batchsize, verbose=False ):

        """
        run the feature generating portion of network only. 
        get feature vector at each pixel coordinate.
        expects 1 planes
        input
        -----
        coord_plane0_t [torch tensor (N_0,3)] list of N pixel coordinates (row,col,batch)
        plane0_feat_t  [torch tensor (N_0,1)] list of pixel values for each N pixel coordinate   
        output
        ------
        tuple containing 3 feature tensors [torch tensors with shapes ((N_0,C), (N_1,C), (N_2,C))]
        """
        if verbose:
            print "[SpatialEmbed::make feature vectors] "
            print "  coord[plane0]=",coord_plane0_t.shape," feat[plane0]=",plane0_feat_t.shape   

        # Form input tuples for input layers
        # adds torch tensors to SparseConvTensor object
        input_plane0 = ( coord_plane0_t, plane0_feat_t, batchsize )

        # generate offset map features
        offset_map = self.source_inputlayer_offset(input_plane0)
        offset_map = self.stem_offset( offset_map )         
        if verbose: print "stem_offset=   ",offset_map.features.shape
        if self.use_unet: 
            offset_map = self.unet_layers_offset(offset_map)        
            if verbose: print "unet_offset=   ",offset_map.features.shape 
        offset_map = self.resnet_layers_offset( offset_map )
        if verbose: print "resnet_offset= ",offset_map.features.shape
        offset_map = self.feature_layer_offset( offset_map )
        if verbose: print "feature_offset=",offset_map.features.shape
                    
        # generate seed map features
        seed_map = self.source_inputlayer_seed(input_plane0)
        seed_map = self.stem_seed( seed_map )         
        if verbose: print "stem_seed=   ",seed_map.features.shape
        if self.use_unet: 
            seed_map = self.unet_layers_seed(seed_map)         
            if verbose: print "unet_seed=   ",seed_map.features.shape
        seed_map = self.resnet_layers_seed( seed_map )
        if verbose: print "resnet_seed= ",seed_map.features.shape
        seed_map = self.feature_layer_seed( seed_map )
        if verbose: print "feature_seed=",seed_map.features.shape

        # extracts torch tensors from SparseConvTensor object        
        offset_map  = self.source_outlayer_offset( offset_map )
        seed_map  = self.source_outlayer_seed( seed_map )

        return offset_map, seed_map


def spatialembed_loss(coord_t, offsets, seeds, binary_maps, class_segmentation, num_instances, types, device, verbose=False, 
                      lovasz_weight=60, seed_weight=1, sigma_smooth_weight=1, iterator=0):
    ''' 
        offsets: 4 channel tensor: e_ix, e_iy, sigma_x, sigma_y
        seeds: #types channel tensor
        binary_maps: (#instances, #number of pixels), list of instances (and their pixels, expressed as 0 or 1)
        class_segmentation: (#classes, #number of pixels), class map for each class (binary) (should be same size as seeds)
        types: vector length #instances, type labels
    '''
    if verbose: 
        print "    Num pixels: ", offsets.detach().size()[0], ", Num instances: ", num_instances
        print "    Offsets size, seeds size, binary_maps size, class_segmentation size:"
        print "    ", offsets.detach().size(), seeds.detach().size(), binary_maps.detach().size(), class_segmentation.detach().size()
        print "      seeds: ",seeds.shape
        print "      binary_maps: ",binary_maps.shape
        print "      class seg: ",class_segmentation.shape

    o_x, o_y   = coord_t[:,0], coord_t[:,1]
    e_ix, e_iy = offsets[:,0], offsets[:,1]
    e_ix, e_iy = o_x + e_ix, o_y + e_iy
    sigma_x, sigma_y = offsets[:,2], offsets[:,3] 

    # Gaussian tensor

    # Average e_i for pixels in each instance
    x_centroids_binary = torch.mul(e_ix, binary_maps)
    y_centroids_binary = torch.mul(e_iy, binary_maps)

    x_centroids = x_centroids_binary.sum(dim=1) / (x_centroids_binary != 0).sum(dim=1).float()
    y_centroids = y_centroids_binary.sum(dim=1) / (y_centroids_binary != 0).sum(dim=1).float()

    # Average sigma_k per instance
    sigma_x_binary = torch.mul(sigma_x, binary_maps)
    sigma_y_binary = torch.mul(sigma_y, binary_maps)

    sigma_kx = sigma_x_binary.sum(dim=1) / (sigma_x_binary != 0).sum(dim=1).float()
    sigma_ky = sigma_y_binary.sum(dim=1) / (sigma_y_binary != 0).sum(dim=1).float()

    # print sigma_kx, sigma_ky

    # Gaussian per instance (as per paper)
    x_portion = e_ix.repeat(num_instances, 1)
    x_portion = x_portion - x_centroids.view(num_instances, 1)
    x_portion = x_portion.pow(2)
    x_portion = torch.div( x_portion, 2 * sigma_kx.view(num_instances, 1).pow(2) )

    y_portion = e_iy.repeat(num_instances, 1)
    y_portion = y_portion - y_centroids.view(num_instances, 1)
    y_portion = y_portion.pow(2)
    y_portion = torch.div( y_portion, 2 * sigma_ky.view(num_instances, 1).pow(2) )

    gaussian = - x_portion - y_portion

    gaussian = torch.exp(gaussian) # size should be (num_instances, num_pixels)

    mseloss = nn.MSELoss()
    bceloss = nn.BCELoss()
    mseloss_class = nn.MSELoss(reduction='none')
    loss_gaus = 0

    
    folded_gaussian = torch.max(gaussian, dim=0)[0]
    folded_binary_maps = torch.max(binary_maps, dim=0)[0]

    print "expected ", int((binary_maps[0].detach() > 0.2).sum().float()),int((binary_maps[1].detach() > 0.2).sum().float()),int((binary_maps[2].detach() > 0.2).sum().float())
    print "Gaussian ", int((gaussian[0].detach() > 0.2).sum().float()), int((gaussian[1].detach() > 0.2).sum().float()), int((gaussian[2].detach() > 0.2).sum().float())
    print "tot      ", int((folded_binary_maps.detach() > 0.2).sum().float()), int((folded_gaussian.detach() > 0.2).sum().float())

    for i in range(num_instances):
        loss_gaus += 20 * mseloss(gaussian[i], binary_maps[i])

    loss_gaus += 40 * mseloss(folded_gaussian, folded_binary_maps)
    # loss += 60 * lovasz_losses.lovasz_hinge_flat(folded_gaussian, folded_binary_maps)

    if verbose: print "        Lovasz: ", loss_gaus.detach()

    if iterator == -1: 
        temp_coord_t = np.array(coord_t.detach().to('cpu'))
        x, y, dummy = zip(*np.array(coord_t.detach().to('cpu')))

        plt.scatter(x, y, marker='.', c=list(np.array((folded_gaussian.detach().to('cpu') * 10).type(torch.int))), cmap=plt.cm.autumn)
        # plt.scatter(x, y, marker='.', c=list(np.array((folded_binary_maps.detach().to('cpu') * 10).type(torch.int))), cmap=plt.cm.autumn)

        plt.colorbar()
        # plt.plot(x, y, '.', markersize=5, color='black')
        # for inst_idx in xrange(entry.DataBranch.num_instances_plane(plane)):
        #     inst_xs, inst_ys = zip(*entry.DataBranch.instance(plane, inst_idx))
        #     type_inst = abs(entry.DataBranch.typeof_instance(plane, inst_idx))
        #     plt.plot(inst_xs, inst_ys, '.', markersize=7, color=colors[type_inst], label=particle_names[type_inst])
        plt.show()


    # Seed-map MSE loss
    print "folded Gaussian: ",folded_gaussian.shape
    gaussian_class_segmentation = folded_gaussian.detach() * class_segmentation
    if verbose:
        print "gaussian_class_segmentation: ",gaussian_class_segmentation.shape," requires_grad=",gaussian_class_segmentation.requires_grad
    

    class_pix_w = torch.ones( seeds.shape, requires_grad=False).to( seeds.device )

    nclasstot = torch.sum( class_segmentation, 1 )
    nbgtot = float(seeds.shape[0])-nclasstot    
    print "  nclasstot: ",nclasstot," nbgtot: ",nbgtot    
    print "Classseg ",
    for i in range(6):
        print int((gaussian_class_segmentation.detach()[i] > 0.2).sum().float()),
        class_pix_w[:,i][ class_segmentation[i,:]==1.0 ] = (nbgtot[i]+1)/seeds.shape[0]
        class_pix_w[:,i][ class_segmentation[i,:]==0.0 ] = (nclasstot[i]+1)/seeds.shape[0]
    print 

    # weight class
    if verbose: print "  class weights: ",class_pix_w.shape
    
    print "Seeds    ",
    for i in range(6):
        print int((seeds.detach().t()[i] > 0.2).sum().float()),
    print 

    if verbose: print "        Seed: ", (seed_weight * mseloss(seeds, gaussian_class_segmentation.t())).detach()

    # seeds = 10 * seeds
    # gaussian_class_segmentation = 10 * gaussian_class_segmentation

    # gaussian_class_segmentation = gaussian_class_segmentation.t()
    # loss += bceloss(seeds, gaussian_class_segmentation)

    sigmoid = torch.nn.Sigmoid()
    sig_seeds = sigmoid(seeds)
    loss_seed = mseloss_class(seeds, gaussian_class_segmentation.t())
    class_pix_w *= 200
    loss_seed *= class_pix_w
    loss_seed = loss_seed.sum()
    # loss_seed *= 140
    if verbose: print "loss_seed: ",loss_seed


    # seeds = seeds / 10
    # gaussian_class_segmentation = gaussian_class_segmentation / 10


    # Sigma-smoothing loss 
    #    (take sigmas @instance pixel locations and consolidate into one vector)
    avg_x_sigma_flattened = binary_maps * sigma_kx.view(num_instances, 1).detach()
    avg_y_sigma_flattened = binary_maps * sigma_ky.view(num_instances, 1).detach()

    avg_x_sigma_flattened = torch.max(avg_x_sigma_flattened.t(), 1)[0]
    avg_y_sigma_flattened = torch.max(avg_y_sigma_flattened.t(), 1)[0]

    #only compare it to the places that we care about
    seeds_x_sigma = binary_maps * sigma_x.detach()
    seeds_y_sigma = binary_maps * sigma_y.detach()

    seeds_x_sigma = torch.max(seeds_x_sigma.t(), 1)[0]
    seeds_y_sigma = torch.max(seeds_y_sigma.t(), 1)[0]

    #if verbose: print "        Sigma: ", (sigma_smooth_weight * mseloss(sigma_x, avg_x_sigma_flattened) + sigma_smooth_weight * mseloss(sigma_y, avg_y_sigma_flattened)).detach()
    loss_sigma = mseloss(seeds_x_sigma, avg_x_sigma_flattened) 
    loss_sigma += mseloss(seeds_y_sigma, avg_y_sigma_flattened) 

    print "    Loss Sigma: ", loss_sigma

    loss = loss_gaus+loss_seed+loss_sigma
    print "    Loss: ", loss.detach()
    return loss


def post_process(coord_t, offsets, seeds):

    seeds = seeds.t()
    instances = []
    for i, instance_type in enumerate(seeds):
        visited = set()

        # initialize first max pixel
        max_idx, max_val = max_with_index(instance_type, visited)
        c_kx, c_ky  = coord_t[max_idx][0], coord_t[max_idx][1]
        cent_eix, cent_eiy = c_kx + offsets[max_idx][0], c_ky + offsets[max_idx][1]
        sigma_x, sigma_y  = offsets[max_idx][2], offsets[max_idx][3]
        centroid_val = np.exp(-((cent_eix - c_kx)**2)/(2*(sigma_x**2)) - ((cent_eiy - c_ky)**2)/(2*(sigma_y**2)))

        print '--------------------------'
        print cent_eix - c_kx, cent_eiy - c_ky
        print sigma_x, sigma_y
        print '--------------------------'
        # print "    c_kx, c_ky ", c_kx, c_ky
        # print "    cent_eix, cent_eiy ", cent_eix, cent_eiy
        # print "    sigma_x, sigma_y ", sigma_x, sigma_y
        # print "    inner exp, ", ((-((cent_eix - c_kx)**2)/(2*(sigma_x**2)) - ((cent_eiy - c_ky)**2)/(2*(sigma_y**2))))
        print "Class, seed, offset ", i, max_val, centroid_val


        # ============================ debugging 
        # Gaussian per instance (as per paper)
        
        # temp_coord_t = np.array(coord_t.detach().to('cpu'))
        # x, y, dummy = zip(*np.array(coord_t.detach().to('cpu')))

        # plt.scatter(x, y, marker='.', c=list(np.array(instance_type.detach().to('cpu'))), cmap=plt.cm.autumn)
        # plt.colorbar()
        # plt.show()
        # ============================ debugging 

        instances_for_class = []
        while centroid_val > 0.5:  # while there are pixels which we think are part of instances...
            max_idx, max_val = max_with_index(instance_type, visited)
            c_kx, c_ky  = coord_t[max_idx][0], coord_t[max_idx][1]
            cent_eix, cent_eiy = c_kx + offsets[max_idx][0], c_ky + offsets[max_idx][1]
            sigma_x, sigma_y  = offsets[max_idx][2], offsets[max_idx][3]
            centroid_val = np.exp(-((cent_eix - c_kx)**2)/(2*(sigma_x**2)) - ((cent_eiy - c_ky)**2)/(2*(sigma_y**2)))

            visited.add(max_idx)

            instance_pixels = [(coord_t[max_idx][0], coord_t[max_idx][1])]
            for i, pix in enumerate(instance_type):  # collect all non-visited pixels that belong to that centroid
                if (i not in visited):
                    eix, eiy = coord_t[i][0] + offsets[i][0], coord_t[i][1] + offsets[i][1]
                    val = -((eix - c_kx)**2)/(2*(sigma_x**2)) - ((eiy - c_ky)**2)/(2*(sigma_y**2))
                    if (np.exp(val) > 0.5):
                        visited.add(i)
                        instance_pixels.append((coord_t[i][0], coord_t[i][1]))
            
            instances_for_class.append(instance_pixels)

        instances.append(instances_for_class)

    return instances

def max_with_index(ray, visited):
    initialized = False
    for i, elem in enumerate(ray):
        if (not initialized) and (i not in visited):
            idx, maximum = i, elem
            initialized = True

        if (elem > maximum) and (i not in visited):
            idx, maximum = i, elem
    
    return idx, maximum
