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
        self.cls = nn.Linear(768*2, 2)
        config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
        self.bert = AutoModel.from_config(config)  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        self.forward_function = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.backward_function = BertForSequenceClassification.from_pretrained('bert-base-uncased')

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

        prev_sents_ids, next_sents_ids = input_ids[0], input_ids[1]
        prev_attention_mask, next_attention_mask = attention_mask[0], attention_mask[1]
        prev_type_ids, next_type_ids = token_type_ids[0], token_type_ids[1]

        # TODO: check input argument
        prev_outs = self.bert(prev_sents_ids,
                                          attention_mask = prev_attention_mask,
                                          token_type_ids = prev_type_ids)
        next_outs = self.bert(next_sents_ids,
                                          attention_mask = next_attention_mask,
                                          token_type_ids = next_type_ids)
        
        
        prev_last_hid = prev_outs[0]
        prev_pooler = prev_outs[1]
        next_lat_hid = next_outs[0]
        next_pooler = next_outs[1]
        
        
        
        pdb.set_trace()
        #outputs = torch.cat((prev_outs, next_outs), 1)


        pooled_outs = torch.cat((prev_pooler, next_pooler), 1)
        pdb.set_trace()
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
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
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