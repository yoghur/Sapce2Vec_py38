import torch
import logging
import config
from encoder import *
from decoder import *
from model import *

def cudify(feature_embedding):
    feature_embed_lookup = lambda pt_types: feature_embedding(
        torch.autograd.Variable(torch.LongTensor(pt_types).cuda()))
    return feature_embed_lookup

def setup_console():
    logging.getLogger('').handlers = []
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def setup_logging(log_file,console=True,filemode='w'):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file,
                        filemode=filemode)
    if console:
        #logging.getLogger('').handlers = []
        console = logging.StreamHandler()
        # optional, set the logging level
        console.setLevel(logging.INFO)
        # set a format which is the same for console use
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
    return logging

def get_encoder(feature_embed_lookup,feature_embedding,pointset,enc_agg):
    if enc_agg == "mean":
        enc = PointFeatureEncoder(feature_embed_lookup,feature_embedding,pointset,agg_func=torch.mean)
    elif enc_agg == "min":
        enc = PointFeatureEncoder(feature_embed_lookup, feature_embedding, pointset, agg_func=torch.min)
    elif enc_agg == "max":
        enc = PointFeatureEncoder(feature_embed_lookup, feature_embedding, pointset, agg_func=torch.max)
    else:
        raise Exception("Aggregation function no support!")
    return enc

def get_ffn(input_dim,f_act,context_str=""):
    if config.use_layn == "T":
        use_layn = True
    else:
        use_layn = False
    if config.skip_connection == "T":
        skip_connection = True
    else:
        skip_connection = False
    return MultiLayerFeedForwardNN(
            input_dim=input_dim,
            output_dim=config.spa_embed_dim,
            num_hidden_layers=config.num_hidden_layer,
            dropout_rate=config.dropout,
            hidden_dim=config.hidden_dim,
            activation=f_act,
            use_layernormalize=use_layn,
            skip_connection = skip_connection,
            context_str = context_str)

def get_spatial_context(model_type,max_radius):
    if model_type == "global":
        extent = (-1710000,-1690000+1000,1610000,1640000+1000)
    elif model_type == "relative":
        extent = (-max_radius-500,max_radius+500,-max_radius-500,max_radius+500)
    return extent

def get_spa_encoder(model_type,spa_enc_type,spa_embed_dim,coord_dim=2,num_rbf_anchor_pts=100,
                    rbf_kernel_size=10e2,frequency_num=16,max_radius=10000,min_radius=1,
                    f_act="sigmoid",freq_init="geometric",use_postmat="T"):
    if config.use_layn=="T":
        use_layn = True
    else:
        use_layn = False
    if use_postmat=="T":
        use_post_mat = True
    else:
        use_post_mat = False
    if spa_enc_type=="gridcell":
        ffn = get_ffn(input_dim=int(4*frequency_num),f_act=f_act,context_str="GridCellSpatialRelationEncoder")
        spa_enc = GridCellSpatialRelationEncoder(spa_embed_dim,coord_dim=coord_dim,frequency_num=frequency_num,max_radius=max_radius,min_radius=min_radius,freq_init=freq_init,ffn=ffn)
    elif spa_enc_type=="theory":
        ffn = get_ffn(input_dim=int(6*frequency_num),f_act=f_act,context_str="TheoryGridCellSpatialRelationEncoder")
        spa_enc = TheoryGridCellSpatialRelationEncoder(spa_embed_dim,coord_dim=coord_dim,frequency_num=frequency_num,max_radius=max_radius,min_radius=min_radius,freq_init=freq_init,ffn=ffn)
    elif spa_enc_type=="hexagridcell":
        spa_enc = HexagonGridCellSpatialRelationEncoder(spa_embed_dim,coord_dim=coord_dim,frequency_num=frequency_num,max_radius=max_radius,dropout=config.dropout,f_act=f_act)
    else:
        raise Exception("Space encoder function no support!")
    return spa_enc

def get_context_decoder(dec_type,query_dim,key_dim,spa_embed_dim,g_spa_embed_dim,have_query_embed=True,
                        num_attn=1,activation="leakyrule",f_activation="sigmoid",layn="T",use_postmat="T",dropout=0.5):
    if layn=="T":
        layernorm = True
    else:
        layernorm = False
    if use_postmat=="T":
        use_post_mat = True
    else:
        use_post_mat = False
    if dec_type=="concat":
        dec = IntersectConcatAttention(query_dim,key_dim,spa_embed_dim,have_query_embed=have_query_embed,num_attn=num_attn,
                                       activation=activation,f_activation=f_activation,
                                       layernorm=layernorm,use_post_mat=use_post_mat,dropout=dropout)
    elif dec_type=="g_pos_concat":
        dec = GolbalPositionIntersectConcatAttention(query_dim,key_dim,spa_embed_dim,g_spa_embed_dim,have_query_embed=have_query_embed,
                                                     num_attn=num_attn,activation=activation,f_activation=f_activation,
                                                     layernorm=layernorm,use_post_mat=use_post_mat,dropout=dropout)
    else:
        raise Exception("decoder type not support!!")
    return dec

def get_enc_dec(model_type, pointset, enc, spa_enc = None, 
                g_spa_enc = None, g_spa_dec = None, init_dec=None, dec=None, joint_dec=None, 
                activation = "sigmoid", num_context_sample = 10, num_neg_resample = 10):
    if model_type=="relative":
        enc_dec = NeighGraphEncoderDecoder(pointset=pointset,
                                           enc=enc,
                                           spa_enc=spa_enc,
                                           init_dec=init_dec,
                                           dec=dec,
                                           activation=activation,
                                           num_context_sample=num_context_sample,
                                           num_neg_resample=num_neg_resample)
    elif model_type=="global":
        enc_dec = GlobalPositionEncoderDecoder(pointset=pointset, 
                                                enc = enc, 
                                                g_spa_enc = g_spa_enc, 
                                                g_spa_dec = g_spa_dec, 
                                                activation = activation, 
                                                num_neg_resample = num_neg_resample)
    elif model_type=="join":
        enc_dec = JointRelativeGlobalEncoderDecoder(pointset=pointset,
                                                    enc=enc,
                                                    spa_enc=spa_enc,
                                                    g_spa_enc=g_spa_enc,
                                                    g_spa_dec=g_spa_dec,
                                                    init_dec=init_dec,
                                                    dec=dec,
                                                    joint_dec=joint_dec,
                                                    activation=activation,
                                                    num_context_sample=num_context_sample,
                                                    num_neg_resample=num_neg_resample)
    elif model_type=="together":
        enc_dec = GlobalPositionNeighGraphEncoderDecoer(pointset=pointset,
                                                        enc=enc,
                                                        spa_enc=spa_enc,
                                                        g_spa_enc=g_spa_enc,
                                                        init_dec=init_dec,
                                                        dec=dec,
                                                        activation=activation,
                                                        num_context_sample=num_context_sample,
                                                        num_neg_resample=num_neg_resample)
    else:
        raise Exception("Unknown Model Type!!!")
    return enc_dec