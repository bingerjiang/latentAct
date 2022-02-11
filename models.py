#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 23:33:08 2022

@author: binger
"""
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import  BertForPreTraining, BertForSequenceClassification, AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import NextSentencePredictorOutput

import pdb
from info_nce import *


class BertForForwardBackwardPrediction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # self.bert = BertModel(config)
        
        config = BertConfig.from_pretrained('bert-base-uncased')    
        self.bert = AutoModel.from_config(config)
        self.z_forward = nn.Linear(config.hidden_size, config.hidden_size)
        self.z_backward = nn.Linear(config.hidden_size, config.hidden_size)
        self.cls = nn.Linear(config.hidden_size*2, 2)

        # Initialize weights and apply final processing
        #self.post_init()

    #@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    #@replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):


        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        prev_sents_ids,curr_sents_ids, next_sents_ids = input_ids[0], input_ids[1], input_ids[2]
        prev_attention_mask, curr_attention_mask, next_attention_mask = attention_mask[0], attention_mask[1],attention_mask[2]
        prev_type_ids, curr_type_ids, next_type_ids = token_type_ids[0], token_type_ids[1], token_type_ids[2]

        # TODO: check input argument
        prev_out = self.bert(prev_sents_ids,
                                          attention_mask = prev_attention_mask,
                                          token_type_ids = prev_type_ids)
        next_out = self.bert(next_sents_ids,
                                          attention_mask = next_attention_mask,
                                          token_type_ids = next_type_ids)
        curr_out = self.bert(curr_sents_ids,
                                          attention_mask = curr_attention_mask,
                                          token_type_ids = curr_type_ids)
        
        prev_last_hid, prev_pooler = prev_out['last_hidden_state'],prev_out['pooler_output']
        next_last_hid, next_pooler = next_out['last_hidden_state'], next_out['pooler_output']
        curr_last_hid, curr_pooler = curr_out['last_hidden_state'], curr_out['pooler_output']
        
        ## get forward function and backward function
        prev_forward =self.z_forward(prev_pooler)
        curr_backward = self.z_backward(curr_pooler)
        curr_forward = self.z_forward(curr_pooler)
        next_backward = self.z_backward(next_pooler)
        
        forward_pooled_outs = torch.cat((prev_forward, curr_forward), 0)
        backward_pooled_outs = torch.cat((curr_backward, next_backward), 0)
        pooled_outs = torch.cat((forward_pooled_outs, backward_pooled_outs),1)
        
        labels = labels.repeat(2,1)

        seq_relationship_scores = self.cls(pooled_outs)
        #pdb.set_trace()
        forward_backward_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            forward_backward_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))
        #pdb.set_trace()
        return NextSentencePredictorOutput(
            loss=forward_backward_loss,
            logits=seq_relationship_scores,
            hidden_states=[prev_forward, curr_forward, curr_backward, next_backward],
            #attentions=prev_outs.attentions,
        )
    
class BertForForwardBackwardPrediction_narrow(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # self.bert = BertModel(config)
        
        config = BertConfig.from_pretrained('bert-base-uncased')    
        self.bert = AutoModel.from_config(config)
        self.z_forward = nn.Linear(config.hidden_size, 64)
        self.z_backward = nn.Linear(config.hidden_size, 64)
        self.cls = nn.Linear(64*2, 2)

        # Initialize weights and apply final processing
        #self.post_init()

    #@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    #@replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):


        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        prev_sents_ids,curr_sents_ids, next_sents_ids = input_ids[0], input_ids[1], input_ids[2]
        prev_attention_mask, curr_attention_mask, next_attention_mask = attention_mask[0], attention_mask[1],attention_mask[2]
        prev_type_ids, curr_type_ids, next_type_ids = token_type_ids[0], token_type_ids[1], token_type_ids[2]

        # TODO: check input argument
        prev_out = self.bert(prev_sents_ids,
                                          attention_mask = prev_attention_mask,
                                          token_type_ids = prev_type_ids)
        next_out = self.bert(next_sents_ids,
                                          attention_mask = next_attention_mask,
                                          token_type_ids = next_type_ids)
        curr_out = self.bert(curr_sents_ids,
                                          attention_mask = curr_attention_mask,
                                          token_type_ids = curr_type_ids)
        
        prev_last_hid, prev_pooler = prev_out['last_hidden_state'],prev_out['pooler_output']
        next_last_hid, next_pooler = next_out['last_hidden_state'], next_out['pooler_output']
        curr_last_hid, curr_pooler = curr_out['last_hidden_state'], curr_out['pooler_output']
        
        ## get forward function and backward function
        prev_forward =self.z_forward(prev_pooler)
        curr_backward = self.z_backward(curr_pooler)
        curr_forward = self.z_forward(curr_pooler)
        next_backward = self.z_backward(next_pooler)
        
        forward_pooled_outs = torch.cat((prev_forward, curr_forward), 0)
        backward_pooled_outs = torch.cat((curr_backward, next_backward), 0)
        pooled_outs = torch.cat((forward_pooled_outs, backward_pooled_outs),1)
        
        labels = labels.repeat(2,1)

        seq_relationship_scores = self.cls(pooled_outs)
        #pdb.set_trace()
        forward_backward_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            forward_backward_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        return NextSentencePredictorOutput(
            loss=forward_backward_loss,
            logits=seq_relationship_scores,
            hidden_states=[prev_forward, curr_forward, curr_backward, next_backward],
            #attentions=prev_outs.attentions,
        )
        
class BertForForwardBackward_binary_flex(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # self.bert = BertModel(config)
        
        config = BertConfig.from_pretrained('bert-base-uncased')    
        self.bert = AutoModel.from_config(config)
        self.z_forward = nn.Linear(config.hidden_size, config.FB_function_size)
        self.z_backward = nn.Linear(config.hidden_size, config.FB_function_size)
        self.cls = nn.Linear(config.FB_function_size*2, 2)

        # Initialize weights and apply final processing
        #self.post_init()

    #@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    #@replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):


        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        prev_sents_ids,curr_sents_ids, next_sents_ids = input_ids[0], input_ids[1], input_ids[2]
        prev_attention_mask, curr_attention_mask, next_attention_mask = attention_mask[0], attention_mask[1],attention_mask[2]
        prev_type_ids, curr_type_ids, next_type_ids = token_type_ids[0], token_type_ids[1], token_type_ids[2]

        # TODO: check input argument
        prev_out = self.bert(prev_sents_ids,
                                          attention_mask = prev_attention_mask,
                                          token_type_ids = prev_type_ids)
        next_out = self.bert(next_sents_ids,
                                          attention_mask = next_attention_mask,
                                          token_type_ids = next_type_ids)
        curr_out = self.bert(curr_sents_ids,
                                          attention_mask = curr_attention_mask,
                                          token_type_ids = curr_type_ids)
        
        prev_last_hid, prev_pooler = prev_out['last_hidden_state'],prev_out['pooler_output']
        next_last_hid, next_pooler = next_out['last_hidden_state'], next_out['pooler_output']
        curr_last_hid, curr_pooler = curr_out['last_hidden_state'], curr_out['pooler_output']
        
        ## get forward function and backward function
        prev_forward =self.z_forward(prev_pooler)
        curr_backward = self.z_backward(curr_pooler)
        curr_forward = self.z_forward(curr_pooler)
        next_backward = self.z_backward(next_pooler)
        
        forward_pooled_outs = torch.cat((prev_forward, curr_forward), 0)
        backward_pooled_outs = torch.cat((curr_backward, next_backward), 0)
        pooled_outs = torch.cat((forward_pooled_outs, backward_pooled_outs),1)
        
        labels = labels.repeat(2,1)

        seq_relationship_scores = self.cls(pooled_outs)
        #pdb.set_trace()
        forward_backward_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            forward_backward_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))
        #pdb.set_trace()
        return NextSentencePredictorOutput(
            loss=forward_backward_loss,
            logits=seq_relationship_scores,
            hidden_states=[prev_forward, curr_forward, curr_backward, next_backward],
            #attentions=prev_outs.attentions,
        )
    
        
class BertForForwardBackwardPrediction_cos(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # self.bert = BertModel(config)
        
        config = BertConfig.from_pretrained('bert-base-uncased')    
        self.bert = AutoModel.from_config(config)
        self.z_forward = nn.Linear(config.hidden_size, config.hidden_size)
        self.z_backward = nn.Linear(config.hidden_size, config.hidden_size)
        self.cls = nn.Linear(config.hidden_size*2, 2)

        # Initialize weights and apply final processing
        #self.post_init()

    #@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    #@replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):


        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        prev_sents_ids,curr_sents_ids, next_sents_ids = input_ids[0], input_ids[1], input_ids[2]
        prev_attention_mask, curr_attention_mask, next_attention_mask = attention_mask[0], attention_mask[1],attention_mask[2]
        prev_type_ids, curr_type_ids, next_type_ids = token_type_ids[0], token_type_ids[1], token_type_ids[2]

        # TODO: check input argument
        prev_out = self.bert(prev_sents_ids,
                                          attention_mask = prev_attention_mask,
                                          token_type_ids = prev_type_ids)
        next_out = self.bert(next_sents_ids,
                                          attention_mask = next_attention_mask,
                                          token_type_ids = next_type_ids)
        curr_out = self.bert(curr_sents_ids,
                                          attention_mask = curr_attention_mask,
                                          token_type_ids = curr_type_ids)
        
        prev_last_hid, prev_pooler = prev_out['last_hidden_state'],prev_out['pooler_output']
        next_last_hid, next_pooler = next_out['last_hidden_state'], next_out['pooler_output']
        curr_last_hid, curr_pooler = curr_out['last_hidden_state'], curr_out['pooler_output']
        
        ## get forward function and backward function
        prev_forward =self.z_forward(prev_pooler)
        curr_backward = self.z_backward(curr_pooler)
        curr_forward = self.z_forward(curr_pooler)
        next_backward = self.z_backward(next_pooler)
        
        ## [prev_f, curr_b] anchor is curr_b 
        ## [curr_f, next_b] anchor is curr_f
       
        ## anchors: (batch_size, embedding_size)
        curr_b_anchor = curr_backward[0].unsqueeze(0)
        curr_f_anchor = curr_forward[0].unsqueeze(0)
        ## positives: (batch_size, embedding_size)
        prev_f_pos = prev_forward[0].unsqueeze(0)
        next_b_pos = next_backward[0].unsqueeze(0)
        ## negatives: (batch_size, num_negative, embedding_size)
        prev_f_neg = prev_forward[1:].unsqueeze(0)
        next_b_neg = next_backward[1:].unsqueeze(0)
        
        ### info NCE ###
        infoNCEloss = InfoNCE(negative_mode='paired')
        #pdb.set_trace()

        # loss_1: calculate (prev_f, curr_b)
        # loss_2: calculate (curr_f, next_b)
        loss_1 = infoNCEloss(curr_b_anchor, prev_f_pos, prev_f_neg)
        loss_2 = infoNCEloss(curr_f_anchor, next_b_pos, next_b_neg)
        ### INFO NECT ###
        
        
        #pdb.set_trace()
        forward_backward_loss = loss_1 + loss_2
        
        
        return NextSentencePredictorOutput(
            loss=forward_backward_loss,
            logits=None,
            hidden_states=[prev_forward, curr_forward, curr_backward, next_backward],
            #attentions=prev_outs.attentions,
        )
        
class BertForForwardBackward_cos_flex(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # self.bert = BertModel(config)
        
        #config = BertConfig.from_pretrained('bert-base-uncased')    
        self.bert = AutoModel.from_config(config)
        self.z_forward = nn.Linear(config.hidden_size, config.FB_function_size)
        self.z_backward = nn.Linear(config.hidden_size, config.FB_function_size)
        #self.cls = nn.Linear(config.hidden_size*2, 2)

        # Initialize weights and apply final processing
        #self.post_init()

    #@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    #@replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):


        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        prev_sents_ids,curr_sents_ids, next_sents_ids = input_ids[0], input_ids[1], input_ids[2]
        prev_attention_mask, curr_attention_mask, next_attention_mask = attention_mask[0], attention_mask[1],attention_mask[2]
        prev_type_ids, curr_type_ids, next_type_ids = token_type_ids[0], token_type_ids[1], token_type_ids[2]

        # TODO: check input argument
        prev_out = self.bert(prev_sents_ids,
                                          attention_mask = prev_attention_mask,
                                          token_type_ids = prev_type_ids)
        next_out = self.bert(next_sents_ids,
                                          attention_mask = next_attention_mask,
                                          token_type_ids = next_type_ids)
        curr_out = self.bert(curr_sents_ids,
                                          attention_mask = curr_attention_mask,
                                          token_type_ids = curr_type_ids)
        
        prev_last_hid, prev_pooler = prev_out['last_hidden_state'],prev_out['pooler_output']
        next_last_hid, next_pooler = next_out['last_hidden_state'], next_out['pooler_output']
        curr_last_hid, curr_pooler = curr_out['last_hidden_state'], curr_out['pooler_output']
        
        ## get forward function and backward function
        prev_forward =self.z_forward(prev_pooler)
        curr_backward = self.z_backward(curr_pooler)
        curr_forward = self.z_forward(curr_pooler)
        next_backward = self.z_backward(next_pooler)
        
        ## [prev_f, curr_b] anchor is curr_b 
        ## [curr_f, next_b] anchor is curr_f
       
        ## anchors: (batch_size, embedding_size)
        curr_b_anchor = curr_backward[0].unsqueeze(0)
        curr_f_anchor = curr_forward[0].unsqueeze(0)
        ## positives: (batch_size, embedding_size)
        prev_f_pos = prev_forward[0].unsqueeze(0)
        next_b_pos = next_backward[0].unsqueeze(0)
        ## negatives: (batch_size, num_negative, embedding_size)
        prev_f_neg = prev_forward[1:].unsqueeze(0)
        next_b_neg = next_backward[1:].unsqueeze(0)
        
        ### info NCE ###
        infoNCEloss = InfoNCE(negative_mode='paired')
        #pdb.set_trace()

        # loss_1: calculate (prev_f, curr_b)
        # loss_2: calculate (curr_f, next_b)
        loss_1 = infoNCEloss(curr_b_anchor, prev_f_pos, prev_f_neg)
        loss_2 = infoNCEloss(curr_f_anchor, next_b_pos, next_b_neg)
        ### INFO NECT ###
        
        
        #pdb.set_trace()
        forward_backward_loss = loss_1 + loss_2
        
        
        return NextSentencePredictorOutput(
            loss=forward_backward_loss,
            logits=None,
            hidden_states=[prev_forward, curr_forward, curr_backward, next_backward],
            #attentions=prev_outs.attentions,
        )

class BertForForwardBackwardPrediction_cos_tlayer(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # self.bert = BertModel(config)
        
        config = BertConfig.from_pretrained('bert-base-uncased')    
        self.bert = AutoModel.from_config(config)
        self.z_forward = nn.Linear(config.hidden_size, config.hidden_size)
        self.z_backward = nn.Linear(config.hidden_size, config.hidden_size)
        self.cls = nn.Linear(config.hidden_size*2, 2)
        self.inter_rep = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)

        # Initialize weights and apply final processing
        #self.post_init()

    #@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    #@replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):


        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        prev_sents_ids,curr_sents_ids, next_sents_ids = input_ids[0], input_ids[1], input_ids[2]
        prev_attention_mask, curr_attention_mask, next_attention_mask = attention_mask[0], attention_mask[1],attention_mask[2]
        prev_type_ids, curr_type_ids, next_type_ids = token_type_ids[0], token_type_ids[1], token_type_ids[2]

        # TODO: check input argument
        prev_out = self.bert(prev_sents_ids,
                                          attention_mask = prev_attention_mask,
                                          token_type_ids = prev_type_ids)
        next_out = self.bert(next_sents_ids,
                                          attention_mask = next_attention_mask,
                                          token_type_ids = next_type_ids)
        curr_out = self.bert(curr_sents_ids,
                                          attention_mask = curr_attention_mask,
                                          token_type_ids = curr_type_ids)
        
        prev_last_hid, prev_pooler = prev_out['last_hidden_state'],prev_out['pooler_output']
        next_last_hid, next_pooler = next_out['last_hidden_state'], next_out['pooler_output']
        curr_last_hid, curr_pooler = curr_out['last_hidden_state'], curr_out['pooler_output']
        
        ## get forward function and backward function
        prev_forward =self.z_forward(prev_pooler)
        curr_backward = self.z_backward(curr_pooler)
        curr_forward = self.z_forward(curr_pooler)
        next_backward = self.z_backward(next_pooler)
        
        ## [prev_f, curr_b] anchor is curr_b 
        ## [curr_f, next_b] anchor is curr_f
       
        ## anchors: (batch_size, embedding_size)
        curr_b_anchor = curr_backward[0].unsqueeze(0)
        curr_f_anchor = curr_forward[0].unsqueeze(0)
        ## positives: (batch_size, embedding_size)
        prev_f_pos = prev_forward[0].unsqueeze(0)
        next_b_pos = next_backward[0].unsqueeze(0)
        ## negatives: (batch_size, num_negative, embedding_size)
        prev_f_neg = prev_forward[1:].unsqueeze(0)
        next_b_neg = next_backward[1:].unsqueeze(0)
        
        ### info NCE ###
        infoNCEloss = InfoNCE(negative_mode='paired')
        #pdb.set_trace()
        

        # loss_1: calculate (prev_f, curr_b)
        # loss_2: calculate (curr_f, next_b)
        loss_1 = infoNCEloss(self.inter_rep(curr_b_anchor.unsqueeze(0)).squeeze(0), self.inter_rep(prev_f_pos.unsqueeze(0)).squeeze(0), self.inter_rep(prev_f_neg))
        loss_2 = infoNCEloss(self.inter_rep(curr_f_anchor.unsqueeze(0)).squeeze(0), self.inter_rep(next_b_pos.unsqueeze(0)).squeeze(0), self.inter_rep(next_b_neg))
        ### INFO NECT ###
        
        
        #pdb.set_trace()
        forward_backward_loss = loss_1 + loss_2
        
        
        return NextSentencePredictorOutput(
            loss=forward_backward_loss,
            logits=None,
            hidden_states=[prev_forward, curr_forward, curr_backward, next_backward],
            #attentions=prev_outs.attentions,
        )
        
class BertForForwardBackward_cos_tlayer_flex(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # self.bert = BertModel(config)
        
        #config = BertConfig.from_pretrained('bert-base-uncased')    
        self.bert = AutoModel.from_config(config)
        self.z_forward = nn.Linear(config.hidden_size, config.FB_function_size)
        self.z_backward = nn.Linear(config.hidden_size, config.FB_function_size)
        #self.cls = nn.Linear(config.hidden_size*2, 2)
        self.inter_rep = nn.TransformerEncoderLayer(d_model=config.FB_function_size, nhead=config.num_attention_heads)

        # Initialize weights and apply final processing
        #self.post_init()

    #@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    #@replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):


        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        prev_sents_ids,curr_sents_ids, next_sents_ids = input_ids[0], input_ids[1], input_ids[2]
        prev_attention_mask, curr_attention_mask, next_attention_mask = attention_mask[0], attention_mask[1],attention_mask[2]
        prev_type_ids, curr_type_ids, next_type_ids = token_type_ids[0], token_type_ids[1], token_type_ids[2]

        # TODO: check input argument
        prev_out = self.bert(prev_sents_ids,
                                          attention_mask = prev_attention_mask,
                                          token_type_ids = prev_type_ids)
        next_out = self.bert(next_sents_ids,
                                          attention_mask = next_attention_mask,
                                          token_type_ids = next_type_ids)
        curr_out = self.bert(curr_sents_ids,
                                          attention_mask = curr_attention_mask,
                                          token_type_ids = curr_type_ids)
        
        prev_last_hid, prev_pooler = prev_out['last_hidden_state'],prev_out['pooler_output']
        next_last_hid, next_pooler = next_out['last_hidden_state'], next_out['pooler_output']
        curr_last_hid, curr_pooler = curr_out['last_hidden_state'], curr_out['pooler_output']
        
        ## get forward function and backward function
        prev_forward =self.z_forward(prev_pooler)
        curr_backward = self.z_backward(curr_pooler)
        curr_forward = self.z_forward(curr_pooler)
        next_backward = self.z_backward(next_pooler)
        
        ## [prev_f, curr_b] anchor is curr_b 
        ## [curr_f, next_b] anchor is curr_f
       
        ## anchors: (batch_size, embedding_size)
        curr_b_anchor = curr_backward[0].unsqueeze(0)
        curr_f_anchor = curr_forward[0].unsqueeze(0)
        ## positives: (batch_size, embedding_size)
        prev_f_pos = prev_forward[0].unsqueeze(0)
        next_b_pos = next_backward[0].unsqueeze(0)
        ## negatives: (batch_size, num_negative, embedding_size)
        prev_f_neg = prev_forward[1:].unsqueeze(0)
        next_b_neg = next_backward[1:].unsqueeze(0)
        
        ### info NCE ###
        infoNCEloss = InfoNCE(negative_mode='paired')
        #pdb.set_trace()
        

        # loss_1: calculate (prev_f, curr_b)
        # loss_2: calculate (curr_f, next_b)
        loss_1 = infoNCEloss(self.inter_rep(curr_b_anchor.unsqueeze(0)).squeeze(0), self.inter_rep(prev_f_pos.unsqueeze(0)).squeeze(0), self.inter_rep(prev_f_neg))
        loss_2 = infoNCEloss(self.inter_rep(curr_f_anchor.unsqueeze(0)).squeeze(0), self.inter_rep(next_b_pos.unsqueeze(0)).squeeze(0), self.inter_rep(next_b_neg))
        ### INFO NECT ###
        
        
        #pdb.set_trace()
        forward_backward_loss = loss_1 + loss_2
        
        
        return NextSentencePredictorOutput(
            loss=forward_backward_loss,
            logits=None,
            hidden_states=[prev_forward, curr_forward, curr_backward, next_backward],
            #attentions=prev_outs.attentions,
        )