import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import numpy as np
import torch_geometric
from dlshowermodel.utils.sinusoidal_embeddings import SinusoidalPositionEmbedding
from dlshowermodel.models.SetTransformer import SetTransformer
from dlshowermodel.models.ResGATv2 import ResGATv2Block

# Define the complete model combining Transformer, Position Embedding, and GATv2
class TransformerGATv2Model(nn.Module):
    def __init__(self, spacepoint_feature_dim=48, 
                num_cluster_out_tokens=4,
                cluster_token_dim=64,
                num_cluster_hidden_heads=4, 
                pca_feature_dim=21, 
                node_pos_embedding_dim=48, 
                num_gnn_layers=2,
                gnn_hidden_dim=48, 
                num_gat_heads=4,
                dropout=0.5,
                norm_type='graph',
                edgelayer_hidden_dim=48,
                cat_pos_embed=False,
                x_range=(-520, 520), y_range=(-520, 520), z_range=(-520, 520),
                pos_origin=(0.0,0.0,1036.0/2.0),
                pos_min_freq=0.0001, pos_max_freq=1.0, pos_scale=1.0):
        super(TransformerGATv2Model, self).__init__()
        
        # larmatch vector projection into cluster transformer input space
        self.larmatch_projection = nn.Linear(spacepoint_feature_dim,spacepoint_feature_dim)

        # Transformer for processing spacepoint features within each cluster
        # to make a cluster feature vector to pass to graph
        self.transformer = SetTransformer(
            spacepoint_feature_dim, # dim_input
            num_cluster_out_tokens, # num_outputs
            cluster_token_dim,
            num_hidden_heads=num_cluster_hidden_heads,
            ln=True)

        # Positional embedding for spatial coordinates for graph clusters
        self.position_embedding = SinusoidalPositionEmbedding(
            embedding_dim=node_pos_embedding_dim,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            min_freq=pos_min_freq,
            max_freq=pos_max_freq,
            scale_factor=pos_scale
        )

        self.pos_origin = torch.tensor(pos_origin,dtype=torch.float,requires_grad=False)
        
        # Combine transformer output + PCA + charge feats
        dim_charge_feats = 3
        combined_dim = cluster_token_dim*num_cluster_out_tokens + pca_feature_dim + dim_charge_feats
        self.cat_pos_embed = cat_pos_embed
        if self.cat_pos_embed:
            combined_dim += node_pos_embedding_dim
        self.combined_projection = nn.Linear(combined_dim, node_pos_embedding_dim)
        
        # GATv2Conv layers for edge prediction
        self.num_gnn_layers = num_gnn_layers
        for ilayer in range(num_gnn_layers):
            layername = f'resgatv2conv_layer{ilayer}'
            # default input and output channels and number of heads
            ninput_dims  = gnn_hidden_dim*num_gat_heads
            noutput_dims = gnn_hidden_dim*num_gat_heads
            nheads = num_gat_heads
            # mods for first layer
            if ilayer==0:
                ninput_dims = node_pos_embedding_dim
            # mods for last layer
            if ilayer==(self.num_gnn_layers-1):
                noutput_dims = gnn_hidden_dim*2
                nheads = 1
            conv = ResGATv2Block(ninput_dims, noutput_dims, 
                        heads=nheads, dropout=dropout, norm_type=norm_type)
            setattr(self,layername,conv)
        
        # Edge prediction layers
        self.edge_pred = nn.Sequential(
            nn.Linear(gnn_hidden_dim * 4, edgelayer_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edgelayer_hidden_dim, 1)
        )

        self.init_weights()

    def init_weights(self):
        #torch_init.xavier_uniform_(self.larmatch_projection.weight)
        #self.larmatch_projection.bias.data.zero_()

        # set transformer already init upon constructor

        # set the bias of the edge predictor to reflect the average pos edge fraction (0.02)
        bias_init = -np.log( 0.02 )
        self.edge_pred[-1].bias.data.fill_( bias_init )

    
    def get_node_embeddings(self, data):
        # Process cluster features with transformer

        # project larmatch feature vector into embedding space
        # in: [num_nodes,16,48]
        # out: [num_nodes,16,48]
        N = data.cluster_features.shape[0]
        lmprojection = self.larmatch_projection( data.cluster_features ) 

        # add sampled pos embedding
        #lmprojection = lmprojection + data.sampled_pos_embed
        #print("lmprojection out: ",lmprojection.shape)

        # data.cluster_features shape: [num_nodes, 16, 48]
        transformed_features = self.transformer(lmprojection).view(N,-1)  # Shape: [num_nodes, hidden_dim]
        #print("transformed_features: ",transformed_features.shape)

        # Get position embeddings
        # data.pos shape: [num_nodes, 3]
        # first move node positions relative to origin
        node_pos = data.pos-self.pos_origin.to(data.pos.device)

        # make the position embeddings
        pos_embeddings = self.position_embedding(node_pos)  # Shape: [num_nodes, pos_embedding_dim]
        #print("pos_embeddings: ",pos_embeddings.shape)
        
        # Combine transformer output, position embeddings, and PCA features
        if not self.cat_pos_embed:
            combined = torch.cat([transformed_features, data.pca_features, data.pixsum_features], dim=1)
            node_features = self.combined_projection(combined) + pos_embeddings  # Shape: [num_nodes, gnn_hidden_dim]
        else:
            combined = torch.cat([transformed_features, data.pca_features, data.pixsum_features, pos_embeddings], dim=1)
            node_features = self.combined_projection(combined)  # Shape: [num_nodes, gnn_hidden_dim]
        
        
        return node_features
    
    def forward(self, data):
        # Get graph node embeddings [will go through transformer]
        x = self.get_node_embeddings(data)
        
        # Store node features in data for later use
        data.x = x
        
        # Apply GATv2Conv layers
        for ilayer in range(self.num_gnn_layers):
            conv = getattr(self,f'resgatv2conv_layer{ilayer}')
            x = conv(x, data.edge_index)

        # Get node pairs for edge prediction
        src, dst = data.edge_index
        
        # Get node features for each node in the edge
        src_feat = x[src]  # Shape: [num_edges, gnn_hidden_dim * 2]
        dst_feat = x[dst]  # Shape: [num_edges, gnn_hidden_dim * 2]
        
        # Combine node features for edge prediction
        edge_feat = torch.cat([src_feat, dst_feat], dim=1)  # Shape: [num_edges, gnn_hidden_dim * 4]
        
        # Predict edge probabilities
        edge_pred = self.edge_pred(edge_feat).squeeze(-1)  # Shape: [num_edges]
        
        return edge_pred

    def dump_example_config(outfilepath=None):
        example="""\
        TransformerGATv2:
            SetTransformer:
                spacepoint_feature_dim: 48 
                num_cluster_out_tokens: 4
                cluster_token_dim: 64
                num_cluster_hidden_heads: 4
            ResGATv2:
                pca_feature_dim: 21
                node_pos_embedding_dim: 48
                gnn_hidden_dim: 48
                num_gat_heads: 4
                dropout: 0.5
                edgelayer_hidden_dim:  128
                norm_type: 'graph'
            load_from_checkpoint: False
            checkpoint_file: "your_checkpoint_file.pt"
        """
        import yaml
        cfg = yaml.safe_load(example)
        if outfilepath is not None: 
            assert type(outfilepath) is str, "Please provide string to dump example yaml config."
            with open(outfilepath,'w') as outfile:
                yaml.dump(cfg,outfile,default_flow_style=False)
        return cfg

    def get_checkpoint_weights( checkpoint_filepath ):
        loc_dict = {"cuda:%d"%(gpu):"cpu" for gpu in range(10) }
        state_dict = torch.load(checkpoint_filepath, map_location=loc_dict)
        print("Checkpoint file keys: ",state_dict.keys())
        return state_dict

    def load_from_config( config ):

        if "TransformerGATv2" in config:
            cfg = config["TransformerGATv2"]
        else:
            cfg = config
        
        print(cfg)
        st_cfg = cfg["SetTransformer"]
        gnn_cfg = cfg["ResGATv2"]

        kwdict = {}
        kwdict.update(st_cfg)
        kwdict.update(gnn_cfg)

        model = TransformerGATv2Model(**kwdict)
        if 'load_from_checkpoint' in cfg and cfg['load_from_checkpoint']:
            print("Loading Model weights from Checkpoint")
            checkpoint_file = cfg['checkpoint_file']
            if not os.path.exists(checkpoint_file):
                raise ValueError(f'Could not find checkpoint at {checkpoint_file}')
            state_dict = TransformerGATv2Model.get_checkpoint_weights( checkpoint_file  )
            model.load_state_dict( state_dict )

        return model

if __name__ == "__main__":

    example_config = TransformerGATv2Model.dump_example_config("transformer_gatv2_model.cfg")
    model = TransformerGATv2Model.load_from_config( example_config )
    print(model)

        
