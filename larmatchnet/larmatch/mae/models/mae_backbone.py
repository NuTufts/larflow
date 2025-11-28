"""
MAE Backbone - Refactored Image Feature Extractor

This module wraps the LArMatchMinkowski UNet backbone to extract
image features for spacepoints, separating the image encoding from
task-specific heads.
"""

import torch
import torch.nn as nn

try:
    import MinkowskiEngine as ME
except ImportError:
    ME = None


class MAEBackbone(nn.Module):
    """
    Refactored backbone for MAE that wraps the LArMatchMinkowski encoder.

    This class separates the image encoding pipeline into distinct stages:
    1. encode_images: Run stem + UNet encoder/decoder on wireplane images
    2. extract_spacepoint_features: Get feature vectors for spacepoints from image features

    The backbone can optionally be frozen during MAE pretraining.

    Args:
        larmatch_model: Pretrained LArMatchMinkowski model (or None to create new)
        stem_nfeatures: Number of features in stem (default: 16)
        freeze_backbone: Whether to freeze the UNet backbone weights
        norm_layer: Normalization layer type ('batchnorm', 'instancenorm')
    """

    def __init__(self, larmatch_model=None, stem_nfeatures=16,
                 freeze_backbone=False, norm_layer='batchnorm'):
        super().__init__()

        self.stem_nfeatures = stem_nfeatures
        self.spacepoint_feat_dim = stem_nfeatures * 3  # Features from 3 planes concatenated

        if larmatch_model is not None:
            # Use existing model's backbone
            self.stem = larmatch_model.stem
            self.encoder = larmatch_model.encoder
            self.decoder = larmatch_model.decoder
            self._has_separate_ssnet_decoder = larmatch_model._separate_ssnet_decoder
            if self._has_separate_ssnet_decoder:
                self.ssnet_decoder = larmatch_model.ssnet_decoder
        else:
            # Create new backbone
            self._init_new_backbone(stem_nfeatures, norm_layer)

        if freeze_backbone:
            self.freeze()

    def _init_new_backbone(self, stem_nfeatures, norm_layer):
        """Initialize a new backbone from scratch."""
        if ME is None:
            raise ImportError("MinkowskiEngine is required for MAEBackbone")

        from collections import OrderedDict
        from MinkowskiEngine.modules.resnet_block import BasicBlock

        # Import from larmatch model module
        try:
            from larmatch.model.backbone_resunetme import (
                MinkEncode6LayerInstance, MinkDecode6LayerInstance,
                MinkEncode6LayerBasicBlock, MinkDecode6LayerBasicBlock
            )
            from larmatch.model.resnetinstance_block import (
                BasicBlockInstanceNorm, BasicBlockBatchNorm
            )
        except ImportError:
            raise ImportError("Could not import LArMatch backbone modules")

        input_nfeatures = 1
        stem_nlayers = 3
        stem_layers = OrderedDict()

        for istem in range(stem_nlayers):
            if istem == 0:
                respath = ME.MinkowskiConvolution(
                    input_nfeatures, stem_nfeatures,
                    kernel_size=1, stride=1, dimension=2
                )
                if norm_layer == 'instancenorm':
                    block = BasicBlockInstanceNorm(
                        input_nfeatures, stem_nfeatures,
                        dimension=2, downsample=respath
                    )
                elif norm_layer == 'batchnorm':
                    block = BasicBlockBatchNorm(
                        input_nfeatures, stem_nfeatures,
                        dimension=2, downsample=respath
                    )
            else:
                if norm_layer == 'instancenorm':
                    block = BasicBlockInstanceNorm(
                        stem_nfeatures, stem_nfeatures, dimension=2
                    )
                elif norm_layer == 'batchnorm':
                    block = BasicBlockBatchNorm(
                        stem_nfeatures, stem_nfeatures, dimension=2
                    )
            stem_layers[f"stem_layer{istem}"] = block

        self.stem = nn.Sequential(stem_layers)

        # Encoder/Decoder
        if norm_layer == "instancenorm":
            self.encoder = MinkEncode6LayerInstance(
                in_channels=stem_nfeatures, out_channels=stem_nfeatures, D=2
            )
            self.decoder = MinkDecode6LayerInstance(
                in_channels=stem_nfeatures, out_channels=stem_nfeatures, D=2
            )
        elif norm_layer == "batchnorm":
            self.encoder = MinkEncode6LayerBasicBlock(
                in_channels=stem_nfeatures, out_channels=stem_nfeatures, D=2
            )
            self.decoder = MinkDecode6LayerBasicBlock(
                in_channels=stem_nfeatures, out_channels=stem_nfeatures, D=2
            )

        self._has_separate_ssnet_decoder = False

    def freeze(self):
        """Freeze all backbone parameters."""
        for param in self.stem.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        if self._has_separate_ssnet_decoder:
            for param in self.ssnet_decoder.parameters():
                param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all backbone parameters."""
        for param in self.stem.parameters():
            param.requires_grad = True
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = True
        if self._has_separate_ssnet_decoder:
            for param in self.ssnet_decoder.parameters():
                param.requires_grad = True

    def encode_images(self, input_wireplane_sparsetensors):
        """
        Encode wireplane images through the UNet backbone.

        Args:
            input_wireplane_sparsetensors: List of 3 MinkowskiEngine SparseTensors,
                                          one for each wireplane

        Returns:
            List of 3 SparseTensors containing encoded features for each plane
        """
        x_feat_v = []
        for p, x_input in enumerate(input_wireplane_sparsetensors):
            x = self.stem(x_input)
            x_encode = self.encoder(x)
            x_decode = self.decoder(x_encode)
            x_feat_v.append(x_decode)
        return x_feat_v

    def extract_spacepoint_features(self, feat_v, query_v):
        """
        Extract feature vectors for spacepoints from encoded image features.

        Each spacepoint maps to a pixel in each of the three wireplane images.
        The feature vectors from these three pixels are concatenated to form
        the spacepoint feature vector.

        Args:
            feat_v: List of 3 MinkowskiEngine SparseTensors with encoded features
            query_v: List of 3 tensors, each of shape (N, 3) containing
                    (batch, col, row) coordinates for spacepoint pixel locations

        Returns:
            Tensor of shape (N, 3*stem_nfeatures) containing spacepoint features
        """
        # Get features at query coordinates for each plane
        spacepoint_planefeat_v = [
            feat_v[p].features_at_coordinates(query_v[p])
            for p in range(3)
        ]

        # Concatenate features from all planes and transpose
        # From (N, f) for each plane to (3*f, N) then to (N, 3*f)
        spacepoint_feat = torch.cat(spacepoint_planefeat_v, dim=1)

        return spacepoint_feat

    def forward(self, input_wireplane_sparsetensors, query_v):
        """
        Full forward pass: encode images and extract spacepoint features.

        Args:
            input_wireplane_sparsetensors: List of 3 SparseTensors for wireplane images
            query_v: List of 3 query coordinate tensors

        Returns:
            spacepoint_features: Tensor of shape (N, 3*stem_nfeatures)
            image_features: List of 3 SparseTensors with image features
        """
        # Encode images
        image_features = self.encode_images(input_wireplane_sparsetensors)

        # Extract spacepoint features
        spacepoint_features = self.extract_spacepoint_features(image_features, query_v)

        return spacepoint_features, image_features

    def get_output_dim(self):
        """Return the dimension of spacepoint features."""
        return self.spacepoint_feat_dim


class MAEBackboneWrapper(nn.Module):
    """
    Wrapper that loads a pretrained LArMatchMinkowski checkpoint and
    extracts just the backbone components.

    Args:
        checkpoint_path: Path to LArMatch checkpoint file
        freeze_backbone: Whether to freeze backbone weights
        device: Device to load model on
    """

    def __init__(self, checkpoint_path=None, freeze_backbone=False, device='cuda'):
        super().__init__()

        self.backbone = None

        if checkpoint_path is not None:
            self._load_from_checkpoint(checkpoint_path, device)

        if freeze_backbone and self.backbone is not None:
            self.backbone.freeze()

    def _load_from_checkpoint(self, checkpoint_path, device):
        """Load backbone from a LArMatch checkpoint."""
        import torch

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Get model state dict
        if 'state_model' in checkpoint:
            state_dict = checkpoint['state_model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Create LArMatchMinkowski model
        try:
            from larmatch.model.larmatchminkowski import LArMatchMinkowski
        except ImportError:
            raise ImportError("Could not import LArMatchMinkowski")

        # Determine model configuration from state dict
        # Look at stem layer shapes to determine stem_nfeatures
        stem_key = 'stem.stem_layer0.conv1.kernel'
        if stem_key in state_dict:
            stem_nfeatures = state_dict[stem_key].shape[0]
        else:
            stem_nfeatures = 16

        # Create model and load weights
        larmatch_model = LArMatchMinkowski(
            stem_nfeatures=stem_nfeatures,
            run_lm=False,
            run_ssnet=False,
            run_kp=False,
            run_paf=False
        )

        # Load only backbone weights
        backbone_keys = ['stem', 'encoder', 'decoder']
        filtered_state = {k: v for k, v in state_dict.items()
                         if any(k.startswith(bk) for bk in backbone_keys)}

        larmatch_model.load_state_dict(filtered_state, strict=False)

        # Create MAEBackbone from loaded model
        self.backbone = MAEBackbone(
            larmatch_model=larmatch_model,
            stem_nfeatures=stem_nfeatures,
            freeze_backbone=False
        )

    def forward(self, input_wireplane_sparsetensors, query_v):
        """Forward pass through backbone."""
        return self.backbone(input_wireplane_sparsetensors, query_v)

    def get_output_dim(self):
        """Return backbone output dimension."""
        if self.backbone is not None:
            return self.backbone.get_output_dim()
        return 48  # Default: 16 * 3


def create_backbone(config):
    """
    Factory function to create MAE backbone based on config.

    Args:
        config: Dictionary with backbone configuration

    Returns:
        MAEBackbone or MAEBackboneWrapper instance
    """
    checkpoint_path = config.get('BACKBONE_CHECKPOINT', None)
    freeze_backbone = config.get('FREEZE_BACKBONE', False)
    device = config.get('DEVICE', 'cuda')

    if checkpoint_path is not None:
        return MAEBackboneWrapper(
            checkpoint_path=checkpoint_path,
            freeze_backbone=freeze_backbone,
            device=device
        )
    else:
        stem_nfeatures = config.get('STEM_NFEATURES', 16)
        norm_layer = config.get('NORM_LAYER', 'batchnorm')
        return MAEBackbone(
            stem_nfeatures=stem_nfeatures,
            freeze_backbone=freeze_backbone,
            norm_layer=norm_layer
        )
