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


'''
class BertOnlyNSPHead(nn.Module):
    
    #Copied from  https://huggingface.co/transformers/v2.0.0/_modules/transformers/modeling_bert.html
    
    
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score
'''
class BertForForwardBackwardPrediction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # self.bert = BertModel(config)
        
        config = BertConfig.from_pretrained('bert-base-uncased')    
        self.bert = AutoModel.from_config(config)
        self.z_forward = nn.Linear(config.hidden_size, config.hidden_size)
        self.z_backward = nn.Linear(config.hidden_size, config.hidden_size)
        self.cls = nn.Linear(config.hidden_size*2, 2)
        
        # self.forward_function = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        # self.backward_function = BertForSequenceClassification.from_pretrained('bert-base-uncased')

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

       # if "next_sentence_label" in kwargs:
       #     warnings.warn(
       #         "The `next_sentence_label` argument is deprecated and will be removed in a future version, use `labels` instead.",
       #         FutureWarning,
       #     )
       #     labels = kwargs.pop("next_sentence_label")

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
        #pdb.set_trace()
        
        ## get forward function and backward function
        prev_forward =self.z_forward(prev_pooler)
        curr_backward = self.z_backward(curr_pooler)
        curr_forward = self.z_forward(curr_pooler)
        next_backward = self.z_backward(next_pooler)
        
        forward_pooled_outs = torch.cat((prev_forward, curr_forward), 0)
        backward_pooled_outs = torch.cat((curr_backward, next_backward), 0)
        pooled_outs = torch.cat((forward_pooled_outs, backward_pooled_outs),1)
        
        labels = labels.repeat(2,1)
        #pdb.set_trace()
        seq_relationship_scores = self.cls(pooled_outs)

        forward_backward_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            forward_backward_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

       # if not return_dict:
       #     output = (seq_relationship_scores,) + outputs[2:]
       #     return ((forward_backward_loss,) + output) if forward_backward_loss is not None else output

        return NextSentencePredictorOutput(
            loss=forward_backward_loss,
            logits=seq_relationship_scores,
            #hidden_states=prev_outs.hidden_states,
            #attentions=prev_outs.attentions,
        )
    

        
        
class ForwardModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.forward_function = BertForPreTraining.from_pretrained('bert-base-uncased')
        ### self.backward_function = BertForPreTraining.from_pretrained('bert-base-uncased')

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

        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

       ## prev_sents, next_sents = inputs[0], inputs[1]
        
        # TODO: check input argument
        outputs = self.forward_function(input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,)
        ###next_outs = self.backward_function(next_sents)
        ## outputs = torch.cat((prev_outs, next_outs), 1)

        ## pooled_output_prev = prev_outs[1]
        ## pooled_output_next = next_outs[1]
        ## pooled_outs = torch.cat((pooled_output_prev, pooled_output_next), 1)
        pooled_output = outputs[1]
        
        seq_relationship_scores = self.cls(pooled_outs)

        forward_backward_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            forward_backward_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((forward_backward_loss,) + output) if forward_backward_loss is not None else output

        return pooled_output, outputs.hidden_states, outputs.attentions
##        return NextSentencePredictorOutput(
##           loss=forward_backward_loss,
##            logits=seq_relationship_scores,
##            hidden_states=outputs.hidden_states,
##            attentions=outputs.attentions,
##        )