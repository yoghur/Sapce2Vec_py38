import config
from utils import *
from train_utils import *
from torch import optim
import time
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
class Trainer():
    '''
    Trainer
    '''
    def __init__(self,pointset,train_ng_list,val_ng_list,test_ng_list,feature_embedding,console=True):
        self.pointset = pointset
        self.feature_embedding = feature_embedding
        self.train_ng_list = train_ng_list
        self.val_ng_list = val_ng_list
        self.test_ng_list = test_ng_list
        model_type = config.model_type[0]
        self.model_config = config.enc_agg[0]+'+'+model_type[0]+'+'+config.freq_init+time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()) 
        self.log_file = config.log_dir+self.model_config+'.log'
        self.model_file = config.model_dir+self.model_config+'.pth'
        self.logger = setup_logging(self.log_file,console=console,filemode="a")
        
        
        #Encoder
        self.enc = get_encoder(pointset.feature_embed_lookup,feature_embedding,pointset,config.enc_agg[0])

        if model_type=="relative" or model_type=="join" or model_type=="together":
            self.spa_enc = get_spa_encoder(
                model_type=model_type,
                spa_enc_type=config.spa_enc[2],
                spa_embed_dim=config.spa_embed_dim,
                coord_dim=2,
                num_rbf_anchor_pts=config.num_rbf_anchor_pts,
                rbf_kernel_size=config.rbf_kernal_size,
                frequency_num=config.freq,
                max_radius=config.max_radius,
                min_radius=config.min_radius,
                f_act=config.spa_f_act[1],#0:sigmoid,1:relu
                freq_init=config.freq_init
            )
        else:
            self.spa_enc = None
        
        if model_type=="global" or model_type=="join" or model_type=="together":
            self.g_spa_enc = get_spa_encoder(
                model_type=model_type,
                spa_enc_type=config.g_spa_enc[2],
                spa_embed_dim=config.g_spa_embed_dim,
                coord_dim=2,
                num_rbf_anchor_pts=config.num_rbf_anchor_pts,
                rbf_kernel_size=config.rbf_kernal_size,
                frequency_num=config.g_freq,
                max_radius=config.g_max_radius,
                min_radius=config.g_min_radius,
                f_act=config.g_spa_f_act[1],#0:sigmoid,1:relu
                freq_init=config.g_freq_init,
                use_postmat=config.g_spa_enc_use_postmat
            )
        else:
            self.g_spa_enc = None
    
        #Decoder
        if model_type=="relative" or model_type=="join" or model_type=="together":
            self.init_dec = get_context_decoder(
                dec_type=config.init_decoder_atten_type[0],
                query_dim=config.embed_dim,
                key_dim=config.embed_dim,
                spa_embed_dim=config.spa_embed_dim,
                g_spa_embed_dim=config.g_spa_embed_dim,
                have_query_embed=False,
                num_attn=config.init_decoder_atten_num,
                activation=config.init_decoder_atten_act,
                f_activation=config.init_decoder_atten_f_act,
                layn=config.init_decoder_use_layn,
                use_postmat=config.init_decoder_use_postmat,
                dropout=config.dropout
            )
            if config.use_dec=="T":
                self.dec = get_context_decoder(
                    dec_type=config.decoder_atten_type[0],
                    query_dim=config.embed_dim,
                    spa_embed_dim=config.spa_embed_dim,
                    key_dim=config.embed_dim,
                    g_spa_embed_dim=config.g_spa_embed_dim,
                    have_query_embed=True,
                    num_attn=config.decoder_atten_num,
                    activation=config.decoder_atten_act,
                    f_activation=config.decoder_atten_f_act,
                    layn=config.decoder_use_layn,
                    use_postmat=config.decoder_use_postmat,
                    dropout=config.dropout
                )
            else:
                self.dec = None
            if model_type=="join":
                self.joint_dec = JointRelativeGlobalDecoder(
                    feature_embed_dim=config.embed_dim,
                    f_act=config.act[0],
                    dropout=config.dropout,
                    join_type=config.join_dec_type[0]
                )
            else:
                self.joint_dec = None
        else:
            self.init_dec = None
            self.dec = None
            self.joint_dec = None

        if model_type=="global" or model_type=="join":
            self.g_spa_dec = DirectPositionEmbeddingDecoder(
                g_spa_embed_dim=config.g_spa_embed_dim,
                feature_embed_dim=config.embed_dim,
                f_act=config.act[0],
                dropout=config.dropout
            )
        else:
            self.g_spa_dec = None
        
        #Encoder-Decoder
        self.enc_dec = get_enc_dec(
            model_type=model_type,
            pointset=pointset,
            enc=self.enc,
            spa_enc=self.spa_enc,
            g_spa_enc=self.g_spa_enc,
            g_spa_dec=self.g_spa_dec,
            init_dec=self.init_dec,
            dec=self.dec,
            joint_dec=self.joint_dec,
            activation=config.act[0],
            num_context_sample=config.num_context_sample,
            num_neg_resample=10
        )
    
        if config.cuda:
            self.enc_dec.to(device)

        opt = config.opt[0]
        if opt=="sgd":
            self.optimizer = optim.SGD(filter(lambda p : p.requires_grad,self.enc_dec.parameters()),lr=config.lr,momentum=0)
        elif opt=="adam":
            self.optimizer = optim.Adam(filter(lambda p : p.requires_grad,self.enc_dec.parameters()),lr=config.lr)

        print("Create model from {}".format(self.model_config+".pth"))
        self.logger.info("Save file at {}".format(self.model_config+".pth"))
    
    def load_model(self):
        self.logger.info("Load model from {}".format(self.model_file))
        self.enc_dec.load_state_dict(torch.load(self.model_file))
    
    def eval_model(self,flag="TEST"):
        if flag=="TEST":
            mrr_,hit1_,hit5_,hit10_ = run_eval(self.enc_dec,self.test_ng_list,0,self.logger,do_full_eval=False)
            self.logger.info("Test MRR: {:f}, 10 Neg, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(mrr_, hit1_, hit5_, hit10_))

            mrr_,hit1_,hit5_,hit10_ = run_eval(self.enc_dec,self.test_ng_list,0,self.logger,do_full_eval=True)
            self.logger.info("Test MRR: {:f}, 10 Neg, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(mrr_, hit1_, hit5_, hit10_))
        
        elif flag=="VALID":
            mrr_,hit1_,hit5_,hit10_ = run_eval(self.enc_dec,self.val_ng_list,0,self.logger,do_full_eval=False)
            self.logger.info("Valid MRR: {:f}, 10 Neg, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(mrr_, hit1_, hit5_, hit10_))

            mrr_,hit1_,hit5_,hit10_ = run_eval(self.enc_dec,self.val_ng_list,0,self.logger,do_full_eval=True)
            self.logger.info("Valid MRR: {:f}, 10 Neg, HIT@1: {:f}, HIT@5: {:f}, HIT@10: {:f}".format(mrr_, hit1_, hit5_, hit10_))
        
    def eval_model_per_type(self,typeid2root=None,flag="TEST"):
        if flag == "TEST":
            type2mrr, type2hit1, type2hit5, type2hit10 = run_eval_per_type(self.enc_dec, self.pointset, self.test_ng_list, 0, self.logger, typeid2root = typeid2root, do_full_eval = True)
        elif flag == "VALID":
            type2mrr, type2hit1, type2hit5, type2hit10 = run_eval_per_type(self.enc_dec, self.pointset, self.val_ng_list, 0, self.logger, typeid2root = typeid2root, do_full_eval = True)

        return type2mrr, type2hit1, type2hit5, type2hit10

    def train(self):
        run_train(
            model=self.enc_dec,
            optimizer=self.optimizer,
            train_ng_list=self.train_ng_list,
            val_ng_list=self.val_ng_list,
            test_ng_list=self.test_ng_list,
            logger=self.logger,
            max_iter=config.max_iter,
            batch_size=config.batch_size,
            log_every=config.log_every,
            val_every=config.val_every,
            tol=config.tol,
            model_file=self.model_file
        )
        if self.enc_dec is not None:
            torch.save(self.enc_dec.state_dict(),self.model_file)