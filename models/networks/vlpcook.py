import torch
import torch.nn as nn
import torch.nn.functional as F
from bootstrap.lib.logger import Logger

from .image_networks.networks import ImageEmbedding
from .recipe_networks.networks import HTransformerRecipeEmbedding, Transformer, \
TransformerDecoder, BERTTransformerRecipeEmbedding, CLIPTransformerRecipeEmbedding


from .xbert import BertConfig, BertModel

class VLPCooK(nn.Module):
 
    def __init__(self, opt, nb_classes, with_classif=False, opt_all=None):
        super(VLPCooK, self).__init__()
        self.dim_emb = opt['dim_emb']
        self.nb_classes = nb_classes
        self.with_classif = with_classif
        self.t_align = opt_all['model.criterion.retrieval_strategy'].get('t_align', False)

        self.image_embedding = ImageEmbedding(opt)
        if opt.get('recipe_encoder', False):
            self.bert = 'bert' in opt['recipe_encoder']
            self.clip_encoder = 'clip' in opt['recipe_encoder']
            if 'bert' in opt['recipe_encoder']:
                self.recipe_embedding =  BERTTransformerRecipeEmbedding(opt)
            elif 'clip' in opt['recipe_encoder']:
                self.recipe_embedding =  CLIPTransformerRecipeEmbedding(opt)
            else:
                self.recipe_embedding = HTransformerRecipeEmbedding(opt)
        else:
            raise NotImplementedError



        ## kw encoder
        Logger()("Load VCs encoder...")

        self.aux_kwords = opt.get('aux_kwords', False)
        self.late_cat = opt_all['model.network'].get('late_cat', False)
        self.last_cat = opt_all['model.network'].get('last_cat', False)
        self.cat_pose = opt_all['model.network'].get('cat_pose', 0)
        self.last_cat_aux = opt_all['model.network'].get('last_cat_aux', False)
        self.aux_kwords_encoder = opt_all['model.network'].get('aux_kwords_encoder', False)

        text_encoder = opt_all['model.network'].get('text_encoder', 'bert-base-uncased')
        bert_config_kw = BertConfig.from_json_file(opt_all['model.network.bert_config'])

        vision_width =opt_all['model.network'].get('vision_width', 768)


        bert_config_kw.num_hidden_layers = opt_all['model.network'].get('num_hidden_layers_kw', 2)

        bert_config_kw.fusion_layer = opt_all['model.network'].get('num_hidden_layers_kw', 2)
        
        self.kw_encoder = BertModel.from_pretrained(text_encoder, config=bert_config_kw, add_pooling_layer=False)     

        if self.aux_kwords_encoder:
            self.aux_kw_encoder = BertModel.from_pretrained(text_encoder, config=bert_config_kw, add_pooling_layer=False) 

        
        text_width = self.kw_encoder.config.hidden_size
        self.kw_proj = nn.Linear(text_width, vision_width)



        if self.with_classif:
            self.linear_classif = nn.Linear(self.dim_emb, self.nb_classes)

        ## Cross encoder
        self.n_heads_cross = opt['cross_encoder_params'].get('n_heads', 1)
        self.n_layers_cross = opt['cross_encoder_params'].get('n_layers', 1)
        self.pos_dropout_cross = opt['cross_encoder_params'].get('pos_dropout_cross', False)
        self.p_dropout_cross = opt['cross_encoder_params'].get('p_dropout', 0.1)
        self.cls_token_cross = opt['cross_encoder_params'].get('cls_token', False)
        self.get_tokens_cross = opt['cross_encoder_params'].get('get_tokens', False)
        self.class_attention_cross = opt['cross_encoder_params'].get('class_attention', False)
        self.cls_norm_cross = opt['cross_encoder_params'].get('cls_norm', False)

        self.nb_cross_attention = opt['cross_encoder_params'].get('nb_cross_attention', 1)

        self.self_att = opt['cross_encoder_params'].get('self_att', False)
        self.nb_layers_self = opt['cross_encoder_params'].get('nb_layers_self', 1)

        ## cross decoder
        self.cross_decoder = opt['cross_encoder_params'].get('cross_decoder', False)

        self.itm_loss_weight = opt_all['model.criterion'].get('itm_loss_weight', 0)

        
        if opt_all['misc']['cuda']:
            if opt_all['misc'].get("device_id", False):
                ids = opt_all['misc.device_id']
                if isinstance(ids, list):
                    self.device = torch.device('cuda:'+str(ids[0]))
                else:
                    self.device = torch.device('cuda:'+str(ids))    
            else:
                self.device = torch.device('cuda')

        self.cross_decoder_img = opt.get('cross_decoder_image', False)
        self.context_image = opt.get('context_image', None) # 0 title, 1 ingrds, 2 instrs
        self.vis = opt.get('vis', False)
        if self.itm_loss_weight > 0:
            if self.cross_decoder:
                self.cross_encoder = TransformerDecoder(dim_in=self.dim_emb, n_heads=self.n_heads_cross, 
                        n_layers=self.n_layers_cross, max_seq_tokens=100, 
                        get_tokens=self.get_tokens_cross,)
            else:
                self.cross_encoder = Transformer(dim_in=self.dim_emb, n_heads=self.n_heads_cross, 
                        n_layers=self.n_layers_cross, max_seq_tokens=100,
                        get_tokens=self.get_tokens_cross)
        
            if self.cross_decoder_img:
                self.n_heads_cross_image = opt.get('n_heads_cross_image', 4)
                self.n_layers_cross_image = opt.get('n_layers_cross_image', 2)
                self.dim_cross_image = self.dim_emb #self.image_embedding.dim_out

                self.cross_decoder_image = TransformerDecoder(dim_in=self.dim_cross_image, n_heads=self.n_heads_cross_image, 
                            n_layers=self.n_layers_cross_image, max_seq_tokens=100,
                            get_tokens=self.get_tokens_cross)
                self.proj_recipe_to_image = nn.Linear(self.recipe_embedding.hidden_size, self.dim_cross_image)

            self.proj_cross = nn.Linear(self.dim_emb, 2)
            self.proj_recipe_cross = nn.Linear(self.recipe_embedding.hidden_size, self.dim_emb)
            self.proj_image_cross = nn.Linear(self.image_embedding.dim_out, self.dim_emb)

        add_dim = 0 
        if self.last_cat:
            add_dim+=vision_width
        if self.last_cat_aux:
            add_dim+=vision_width


        self.proj_image = nn.Linear(self.image_embedding.dim_out+add_dim, self.dim_emb)

        self.proj_recipe = nn.Linear(self.recipe_embedding.dim_recipe, self.dim_emb)

        
        self.kwords_same_level = opt.get('kwords_same_level', False)
        self.late_cat_aux = opt.get('late_cat_aux', False)
        self.main_cls = opt.get('main_cls', False)

        if self.aux_kwords:
            self.aux_kw_proj = nn.Linear(text_width, vision_width)

        self.freeze_bert = opt_all['optimizer'].get('freeze_bert', False)
        self.optimizer_bert = opt_all['optimizer'].get('optimizer_bert', False)


    def get_parameters_recipe(self):
        params = []
        params.append({'params': self.recipe_embedding.parameters()})
        if self.with_classif:
            params.append({'params': self.linear_classif.parameters()})
        params.append({'params': self.image_embedding.fc.parameters()})
        params.append({'params': self.proj_image.parameters()})
        params.append({'params': self.proj_recipe.parameters()})
        params.append({'params': self.kw_proj.parameters()})
        if not self.freeze_bert :
            params.append({'params': self.kw_encoder.parameters()})
        if self.aux_kwords:
            params.append({'params': self.aux_kw_proj.parameters()})
            if self.aux_kwords_encoder and not self.freeze_bert:
                params.append({'params': self.aux_kw_encoder.parameters()})
                Logger()('training aux kw modules')

        if self.itm_loss_weight > 0:
            params.append({'params': self.cross_encoder.parameters()})
            params.append({'params': self.proj_cross.parameters()})
            params.append({'params': self.proj_recipe_cross.parameters()})
            params.append({'params': self.proj_image_cross.parameters()})
            if self.cross_decoder_img:
                params.append({'params': self.proj_recipe_to_image.parameters()})
                params.append({'params': self.cross_decoder_image.parameters()})

        return params

    def get_parameters_cross(self):
        params = []
        params.append({'params': self.proj_image.parameters()})
        params.append({'params': self.proj_recipe.parameters()})
        if self.itm_loss_weight > 0:
            params.append({'params': self.cross_encoder.parameters()})
            params.append({'params': self.proj_cross.parameters()})
            params.append({'params': self.proj_recipe_cross.parameters()})
            params.append({'params': self.proj_image_cross.parameters()})
            if self.cross_decoder_img:
                params.append({'params': self.proj_recipe_to_image.parameters()})
                params.append({'params': self.cross_decoder_image.parameters()})

        return params

    def get_parameters_image(self):
        if self.freeze_bert and not self.optimizer_bert:
            params = []
            params.append({'params': self.image_embedding.convnet.parameters()})
            params.append({'params': self.kw_encoder.parameters()})
            if self.aux_kwords_encoder:
                params.append({'params': self.aux_kw_encoder.parameters()})
            return params
        else:
            return self.image_embedding.convnet.parameters()


    def get_parameters_bert(self):
        params = []
        params.append({'params': self.kw_encoder.parameters()})
        if self.aux_kwords_encoder:
            params.append({'params': self.aux_kw_encoder.parameters()})
        return params


    def forward_image(self, image, kws, only_feat=True):

        """ forward dual encoder """
        out = {}
        device = image['data'].device
        kwords_ids = kws['kwords_ids'].to(device)
        kwords_masks = kws['kwords_masks'].to(device)

        if self.aux_kwords:
            index_main_kwords = kwords_ids.shape[1]
            aux_kwords_ids = kws['aux_kwords_ids'].to(device)
            aux_kwords_masks = kws['aux_kwords_masks'].to(device)

            if not self.aux_kwords_encoder:
                kwords_ids = torch.cat((kwords_ids, aux_kwords_ids), dim=1)
                kwords_masks = torch.cat((kwords_masks, aux_kwords_masks), dim=1)


        ### kw
        kw_output = self.kw_encoder(kwords_ids, attention_mask = kwords_masks,                      
                                        return_dict = True, mode = 'text')  
        kw_embeds_main = kw_output.last_hidden_state

        if self.aux_kwords:
            if self.aux_kwords_encoder:
                aux_kw_output = self.aux_kw_encoder(aux_kwords_ids, attention_mask = aux_kwords_masks,                      
                                                return_dict = True, mode = 'text')  
                aux_kw_embeds = aux_kw_output.last_hidden_state
                aux_kw_embeds = self.aux_kw_proj(aux_kw_embeds)
                kw_embeds = kw_embeds_main
            else:
                aux_kw_embeds = kw_embeds_main[:, index_main_kwords:, :]
                aux_kw_embeds = self.aux_kw_proj(aux_kw_embeds)
                kw_embeds = kw_embeds_main[:, :index_main_kwords, :]
        else:
            kw_embeds = kw_embeds_main


        kw_embeds = self.kw_proj(kw_embeds)

        if self.aux_kwords and self.kwords_same_level:

            kw_embeds = torch.cat((kw_embeds, aux_kw_embeds), dim=1)

        if self.late_cat or self.last_cat:
            kw_embeds_external = None
        else:
            kw_embeds_external = kw_embeds

        if self.aux_kwords and not self.kwords_same_level and not self.last_cat_aux: 
            aux_kw_embeds_external = aux_kw_embeds
        else:
            aux_kw_embeds_external = None           

        image_embedding = self.image_embedding(image, external_features=kw_embeds_external, aux_external_features=aux_kw_embeds_external)

        if self.late_cat:
            image_embedding = torch.cat((image_embedding, kw_embeds), dim=1)
        
        if self.last_cat:
            out['kw_embeds'] = kw_embeds

        if self.last_cat_aux:
            if self.main_cls:
                out['aux_kw_embeds'] = kw_embeds
            else:
                out['aux_kw_embeds'] = aux_kw_embeds

        
        if self.late_cat_aux and not self.kwords_same_level:
            image_embedding = torch.cat((image_embedding, aux_kw_embeds), dim=1)

        out['image_embedding'] = image_embedding
        out['image_feat'] = image_embedding

        if self.last_cat or self.last_cat_aux:
            image_embed = out['image_feat'][:, 0, :]
            if self.last_cat:
                kw_embeds = out['kw_embeds'][:, 0, :]
                image_embed = torch.cat((image_embed, kw_embeds), dim=-1)
            if self.last_cat_aux:
                aux_kw_embeds = out['aux_kw_embeds'][:, 0, :]
                image_embed = torch.cat((image_embed, aux_kw_embeds), dim=-1)


        if only_feat:
            return image_embed
        else:
            return self.proj_image(image_embed)



    def forward_DE(self, batch):
        """ forward dual encoder """
        out = {}

        kwords_ids = batch['image']['kwords_ids']
        kwords_masks = batch['image']['kwords_masks']

        if self.aux_kwords:
            index_main_kwords = kwords_ids.shape[1]
            aux_kwords_ids = batch['image']['aux_kwords_ids']
            aux_kwords_masks = batch['image']['aux_kwords_masks']

            if not self.aux_kwords_encoder:
                kwords_ids = torch.cat((kwords_ids, aux_kwords_ids), dim=1)
                kwords_masks = torch.cat((kwords_masks, aux_kwords_masks), dim=1)

        ### kw

        kw_output = self.kw_encoder(kwords_ids, attention_mask = kwords_masks,                      
                                        return_dict = True, mode = 'text')  
        kw_embeds_main = kw_output.last_hidden_state

        if self.aux_kwords:
            if self.aux_kwords_encoder:
                aux_kw_output = self.aux_kw_encoder(aux_kwords_ids, attention_mask = aux_kwords_masks,                      
                                                return_dict = True, mode = 'text')  
                aux_kw_embeds = aux_kw_output.last_hidden_state
                aux_kw_embeds = self.aux_kw_proj(aux_kw_embeds)
                kw_embeds = kw_embeds_main
            else:
                aux_kw_embeds = kw_embeds_main[:, index_main_kwords:, :]
                aux_kw_embeds = self.aux_kw_proj(aux_kw_embeds)
                kw_embeds = kw_embeds_main[:, :index_main_kwords, :]
        else:
            kw_embeds = kw_embeds_main


        kw_embeds = self.kw_proj(kw_embeds)

        if self.aux_kwords and self.kwords_same_level:

            kw_embeds = torch.cat((kw_embeds, aux_kw_embeds), dim=1)

        if self.late_cat or self.last_cat:
            kw_embeds_external = None
        else:
            kw_embeds_external = kw_embeds

        if self.aux_kwords and not self.kwords_same_level and not self.last_cat_aux: 
            aux_kw_embeds_external = aux_kw_embeds
        else:
            aux_kw_embeds_external = None           

        image_embedding = self.image_embedding(batch['image'], external_features=kw_embeds_external, aux_external_features=aux_kw_embeds_external)

        if self.late_cat:
            image_embedding = torch.cat((image_embedding, kw_embeds), dim=1)
        
        if self.last_cat:
            out['kw_embeds'] = kw_embeds

        if self.last_cat_aux:
            if self.main_cls:
                out['aux_kw_embeds'] = kw_embeds
            else:
                out['aux_kw_embeds'] = aux_kw_embeds

        
        if self.late_cat_aux and not self.kwords_same_level:
            image_embedding = torch.cat((image_embedding, aux_kw_embeds), dim=1)

        out['image_embedding'] = image_embedding
        out['image_feat'] = image_embedding


        recipe_embedding = self.recipe_embedding(batch['recipe'])
        out['recipe_feat'] = recipe_embedding[1]
        out['recipe_embedding'] = recipe_embedding[0]
        if self.cross_decoder_img:
            if self.bert or self.clip_encoder:
                out['recipe_feat_sep'] = recipe_embedding[1]
            elif self.context_image is not None:
                out['recipe_feat_sep'] = recipe_embedding[2][self.context_image]
            else:
                out['recipe_feat_sep'] = torch.cat((recipe_embedding[2][0], recipe_embedding[2][1], recipe_embedding[2][2]), dim=1) 
                

        if self.with_classif:
            out['image_classif'] = self.linear_classif(out['image_embedding'])
            out['recipe_classif'] = self.linear_classif(out['recipe_embedding'])


        return out

    def forward(self, batch, only_image=False):

        if only_image:
            img = {}
            img['data'] = batch['image'] 
            return self.forward_image(img, batch, only_feat=False)

            
        out = self.forward_DE(batch)

        image_feat = out['image_feat']

        recipe_embed = out['recipe_embedding']
        recipe_embed = F.normalize(F.tanh(self.proj_recipe(recipe_embed))) # avg pooling for text

        if self.last_cat or self.last_cat_aux:
            image_embed = image_feat[:, 0, :]
            if self.last_cat:
                kw_embeds = out['kw_embeds'][:, 0, :]
                image_embed = torch.cat((image_embed, kw_embeds), dim=-1)
            if self.last_cat_aux:
                aux_kw_embeds = out['aux_kw_embeds'][:, 0, :]
                image_embed = torch.cat((image_embed, aux_kw_embeds), dim=-1)
            image_embed = F.normalize(F.tanh(self.proj_image(image_embed)))
        else:
            image_embed = F.normalize(F.tanh(self.proj_image(image_feat[:, 0, :]))) # select cls token for vit


        if self.itm_loss_weight > 0:

            recipe_feat = out['recipe_feat']
            recipe_feat = self.proj_recipe_cross(recipe_feat)
            out['recipe_feat'] = recipe_feat
            image_feat = self.proj_image_cross(image_feat)
            out['image_feat'] = image_feat

            ## https://github.com/salesforce/ALBEF/blob/9e9a5e952f72374c15cea02d3c34013554c86513/models/model_retrieval.py#L26
            # performing a matrix multiplication between image_embed and the transpose of recipe_embed to compute the similarity matrix between images and texts
            # @ is the matrix multiplication operator in Python (introduced in Python 3.5)
            sim_i2t = image_embed @ recipe_embed.T # (bs1, bs2)
            sim_t2i = sim_i2t.T

            # forward the positve image-text pair
            if self.cross_decoder_img:
                recipe_feat_sep = self.proj_recipe_to_image(out['recipe_feat_sep'])
                out['recipe_feat_sep'] = recipe_feat_sep
                image_feat_pos = self.cross_decoder_image(image_feat, context=recipe_feat_sep)
            else:
                image_feat_pos = image_feat

            if self.cross_decoder:
                output_pos = self.cross_encoder(recipe_feat, context=image_feat_pos)
            else:
                cross_input_pos = torch.cat((image_feat_pos, recipe_feat), dim=1) # (bs, l1 + l2, dim)
                output_pos = self.cross_encoder(cross_input_pos)

            # select negative example
            with torch.no_grad():
                bs = image_feat.size(0)      
                weights_i2t = F.softmax(sim_i2t[:,:bs]+1e-4,dim=1)
                weights_t2i = F.softmax(sim_t2i[:,:bs]+1e-4,dim=1)

                mask = torch.eye(bs).to(weights_i2t.device) > 0
                weights_i2t.masked_fill_(mask, 0)
                weights_t2i.masked_fill_(mask, 0) 

            # select a negative image for each text
            image_feats_neg = []    
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_feats_neg.append(image_feat[neg_idx])
            image_feats_neg = torch.stack(image_feats_neg,dim=0)  # (bs, l1, dim)

            # select a negative text for each image
            recipe_feats_neg = []
            recipe_feats_sep_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                recipe_feats_neg.append(recipe_feat[neg_idx])
                if self.cross_decoder_img:
                    recipe_feats_sep_neg.append(recipe_feat_sep[neg_idx])
            recipe_feats_neg = torch.stack(recipe_feats_neg,dim=0)   # (bs, l2, dim)

            if self.cross_decoder_img:
                recipe_feats_sep_neg = torch.stack(recipe_feats_sep_neg,dim=0)
                recipe_feats_sep_all = torch.cat([recipe_feat_sep, recipe_feats_sep_neg],dim=0)    # (bs+bs, l2, dim) 
            
               


            recipe_feats_all = torch.cat([recipe_feat, recipe_feats_neg],dim=0)    # (bs+bs, l2, dim) 

            image_feats_all = torch.cat([image_feats_neg,image_feat],dim=0)

            if self.cross_decoder_img:
                image_feats_posneg = self.cross_decoder_image(image_feats_all, context=recipe_feats_sep_all)
            else:
                image_feats_posneg = image_feats_all

            if self.cross_decoder:
                output_neg = self.cross_encoder(recipe_feats_all, context=image_feats_posneg)
            else:
                cross_input_neg = torch.cat((image_feats_posneg, recipe_feats_all), dim=1) # (bs+bs, l1 + l2, dim)
                output_neg = self.cross_encoder(cross_input_neg)


            vl_embeddings = torch.cat([output_pos, output_neg],dim=0) # (bsx3, l1+l1, dim)
            vl_output = self.proj_cross(vl_embeddings)            

            if len(vl_output.size()) > 2 and (self.cross_decoder or not self.cross_encoder.get_tokens):
                vl_output = vl_output.mean(1)

            itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                                   dim=0).to(image_feat.device)


            out['cross_embedding'] = vl_output
            out['cross_embedding_labels'] = itm_labels

        out['image_embedding'] = image_embed
        out['recipe_embedding'] = recipe_embed

        return out





 