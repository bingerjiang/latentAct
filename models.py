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
        
        #config = BertConfig.from_pretrained('bert-base-uncased')    
        self.bert = AutoModel.from_config(config)
        self.z_forward = nn.Linear(config.hidden_size, config.FB_function_size)
        self.z_backward = nn.Linear(config.hidden_size, config.FB_function_size)
        self.cls = nn.Linear(config.FB_function_size*2, 2)
        self.activation = nn.Tanh()
        self.pre_z_tanh = config.pre_z_tanh
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
        
        if self.pre_z_tanh:
            prev_pooler = self.activation(prev_pooler)
            curr_pooler = self.activation(curr_pooler)
            next_pooler = self.activation(next_pooler)
        
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
    
class BertForForwardBackward_binary_flex_tanh(BertPreTrainedModel):
    
    '''
    Same with bertforforwardbackward_binary_flex,
    added tanh before feeding to z_function
    '''
    def __init__(self, config):
        super().__init__(config)

        # self.bert = BertModel(config)
        
        #config = BertConfig.from_pretrained('bert-base-uncased')    
        self.bert = AutoModel.from_config(config)
        self.z_forward = nn.Linear(config.hidden_size, config.FB_function_size)
        self.z_backward = nn.Linear(config.hidden_size, config.FB_function_size)
        self.cls = nn.Linear(config.FB_function_size*2, 2)
        self.activation = nn.Tanh()

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
        prev_forward =self.z_forward(self.activation(prev_pooler))
        curr_backward = self.z_backward(self.activation(curr_pooler))
        curr_forward = self.z_forward(self.activation(curr_pooler))
        next_backward = self.z_backward(self.activation(next_pooler))
        
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
        self.pre_z_tanh = config.pre_z_tanh
        # Initialize weights and apply final processing
        #self.post_init()
        self.activation = nn.Tanh()

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
        
        if self.pre_z_tanh:
            prev_pooler = self.activation(prev_pooler)
            curr_pooler = self.activation(curr_pooler)
            next_pooler = self.activation(next_pooler)
        
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
        self.inter_rep = nn.TransformerEncoderLayer(d_model=config.FB_function_size, nhead=4)
        self.pre_z_tanh = config.pre_z_tanh
        self.activation = nn.Tanh()
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
        
        if self.pre_z_tanh:
            prev_pooler = self.activation(prev_pooler)
            curr_pooler = self.activation(curr_pooler)
            next_pooler = self.activation(next_pooler)
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
        
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
       
def get_sentence_embeddings(model, data, device):
   
    
    data['input_ids']=data['input_ids'].to(device)
    data['token_type_ids']=data['token_type_ids'].to(device)
    data['attention_mask']=data['attention_mask'].to(device)
    model.to(device)
    #pdb.set_trace()

    #with torch.no_grad():
    model_output = model(**data)

    #model_output.to(device)
    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, data['attention_mask'])
    #pdb.set_trace()
    return sentence_embeddings        


class CPCAR(nn.Module):

# from fb cpc

    def __init__(self,
                 dimEncoded,  #dimEncoded
                 dimGar, #dimOutput
                 keepHidden,
                 nLevelsGRU,
                 mode="GRU",
                 reverse=False):

        super(CPCAR, self).__init__()
        self.RESIDUAL_STD = 0.1

        if mode == "LSTM":
            self.baseNet = nn.LSTM(dimEncoded, dimGar,
                                   num_layers=nLevelsGRU, batch_first=True)
        elif mode == "RNN":
            self.baseNet = nn.RNN(dimEncoded, dimGar,
                                  num_layers=nLevelsGRU, batch_first=True)
        else:
            self.baseNet = nn.GRU(dimEncoded, dimGar,
                                  num_layers=nLevelsGRU, batch_first=True)

        self.hidden = None
        self.keepHidden = keepHidden
        self.reverse = reverse

    def getDimOutput(self):
        return self.baseNet.hidden_size

    def forward(self, x):

        ##if self.reverse:
        ##    x = torch.flip(x, [1])
        try:
            self.baseNet.flatten_parameters()
        except RuntimeError:
            pass
        x = x.view(-1, 11, 768)
        x, h = self.baseNet(x, self.hidden)
        if self.keepHidden:
            if isinstance(h, tuple):
                self.hidden = tuple(x.detach() for x in h)
            else:
                self.hidden = h.detach()
        
        # For better modularity, a sequence's order should be preserved
        # by each module
        ##if self.reverse:
        ##    x = torch.flip(x, [1])
        
        ## B2: x is [batch_size, seq_len, dim], as batch_first=true
        ## for 1st test, [11, 11, 768] on metawoz
        return x

class cpc_nsp(nn.Module): ## not used
    def __init__(self, config):
        super().__init__(config)

        # self.bert = BertModel(config)
        
        #config = BertConfig.from_pretrained('bert-base-uncased')    
        # self.bert = AutoModel.from_config(config)
        # self.z_forward = nn.Linear(config.hidden_size, config.FB_function_size)
        # self.z_backward = nn.Linear(config.hidden_size, config.FB_function_size)
        self.cls = nn.Linear(config.FB_function_size*2, 2)

        # Initialize weights and apply final processing
        #self.post_init()
        
    #@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    #@replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        sent_embeddings,
        cFeatures,
        args
    ):
        
        cFeatures_prediction = cFeatures[:, :-args.predict_k, :]
            ## predictions to be matched to input encodings 
            
        gar = sent_embeddings.view(-1, 11, 768)
        gar_target = gar[:, args.predict_k:, :]

        cFeatures_prediction = torch.flatten(cFeatures_prediction, start_dim=0, end_dim=1)
        gar_target = torch.flatten(gar_target, start_dim=0, end_dim=1)            

        assert(cFeatures_prediction.shape == gar_target.shape)
        
        embeddings = torch.cat((cFeatures_prediction, gar_target), dim=0)
        
        label_length = gar_target.shape[0]
        labels = torch.arange(label_length)
        labels = labels.repeat(2)
        
        loss_func = NTXentLoss()
        
        loss = loss_func(embeddings, labels)
            
            
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
        
class CPCModel_wrapped(nn.Module):
    def __init__(self,
                 encoder,
                 AR,
                 AR_type):

        super(CPCModel, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR
        self.AR_type = AR_type

    def forward(self, batchData):

        device =torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        sent_embeddings = get_sentence_embeddings(self.gEncoder, batchData, device)
        if self.AR_type == 'transformer':  
            sent_embeddings = sent_embeddings.view(-1, n_turns, dim_encoded)

        cFeatures = self.gAR(sent_embeddings)
        
        cFeatures_prediction = cFeatures[:, :-args.predict_k, :]
            ## predictions to be matched to input encodings 
            
        gar = sent_embeddings.view(-1, 11, 768)
        gar_target = gar[:, args.predict_k:, :]

        cFeatures_prediction = torch.flatten(cFeatures_prediction, start_dim=0, end_dim=1)
        gar_target = torch.flatten(gar_target, start_dim=0, end_dim=1)            

        assert(cFeatures_prediction.shape == gar_target.shape)
        
        embeddings = torch.cat((cFeatures_prediction, gar_target), dim=0)
        #pdb.set_trace()
        
        label_length = gar_target.shape[0]
        labels = torch.arange(label_length)
        labels = labels.repeat(2)
        return cFeature