import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import (
    GPT2PreTrainedModel,
    GPT2Model,
    #jang-r
    # RobertaForSequenceClassification,
    BertModel,
    RobertaConfig,
    BertPreTrainedModel,
    RobertaPreTrainedModel,
    RobertaModel,
    AlbertPreTrainedModel,
    AlbertModel,
    DebertaPreTrainedModel,
    DebertaModel
)


# #jang-r
from transformers.models.albert.modeling_albert import AlbertMLMHead
from transformers.models.roberta.modeling_roberta import RobertaLMHead,RobertaClassificationHead
from transformers.models.deberta.modeling_deberta import DebertaOnlyMLMHead,ContextPooler,StableDropout
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from torch.nn import Softmax, CrossEntropyLoss,MSELoss


def log_softmax(x):
    return torch.exp(x) - torch.sum(torch.exp(x), dim=1, keepdim=True)

def CrossEntropyLoss_Custom(outputs, targets):
    num_examples = targets.shape[0]
    batch_size = outputs.shape[0]
    
    outputs = Softmax(dim=1)(outputs)



    #original
    # outputs = log_softmax(outputs)
    # outputs = outputs[range(batch_size), targets]
    # return -torch.sum(outputs)/num_examples
    
    #jang
    cand_num = outputs.shape[1]
    # outputs[range(batch_size), targets]=-(cand_num-1)*outputs[range(batch_size), targets]
    

    target_logit=outputs[range(batch_size), targets]
    target_logit=torch.log(target_logit)
    target_loss=-torch.sum(target_logit)/num_examples

    unlike_mask=torch.ones_like(outputs)
    unlike_mask[range(batch_size), targets]=0
    unlike_logit=outputs*unlike_mask
    unlike_logit=1-unlike_logit
    unlike_logit=torch.log(unlike_logit)
    # unlike_loss=-torch.sum(unlike_logit)/(num_examples*(cand_num-1))
    unlike_loss=-torch.sum(unlike_logit)/(num_examples)



    return target_loss+0.1*unlike_loss


    


class RobertaForMultipleChoice_MLM(RobertaPreTrainedModel):
       
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.lm_head = RobertaLMHead(config)
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        
        self.post_init()
    
    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings
    
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        masked_lm_labels=None
    ):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_masked_lm_labels = masked_lm_labels.view(-1, masked_lm_labels.size(-1)) if masked_lm_labels is not None else None
        
        
        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
        )
        
        pooled_output = outputs[1]

        if masked_lm_labels is not None:
            sequence_output = outputs[0]
            mlm_logit = self.lm_head(sequence_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            #jang
            # loss_fct=CrossEntropyLoss_Custom
            # loss_fct = BCEWithLogitsLoss()
            loss_fct = CrossEntropyLoss()
            
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(mlm_logit.view(-1, self.config.vocab_size), flat_masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  #(masked_lm_loss,) (lm_loss), reshaped_logits, (hidden_states), (attentions)

class RobertaForSequenceClassification_MLM(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

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
        masked_lm_labels=None
    ):
       
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        
        if masked_lm_labels is not None:
            mlm_logit = self.lm_head(sequence_output)
        mc_logits = self.classifier(sequence_output)

        outputs= (mc_logits,)+outputs[2:]
        loss = None
        if labels is not None:
           
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(mc_logits.squeeze(), labels)
            outputs =(loss,) + outputs
            
            # elif self.config.problem_type == "single_label_classification":
            #     loss_fct = CrossEntropyLoss()
            #     loss = loss_fct(mc_logits.view(-1, self.num_labels), labels.view(-1))
         

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(mlm_logit.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs=(masked_lm_loss,) + outputs

        return outputs


class DebertaForSequenceClassification_MLM(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaModel(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim
        
        self.lm_head = DebertaOnlyMLMHead(config)
        
        self.classifier = nn.Linear(output_dim, num_labels)
        
        drop_out = getattr(config, "cls_dropout", None)
        
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        
        self.dropout = StableDropout(drop_out)

        # Initialize weights and apply final processing
        self.post_init()
    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)
    
    def get_output_embeddings(self):
        return self.lm_head.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        masked_lm_labels=None
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        
        
        if masked_lm_labels is not None:
            sequence_output = encoder_layer
            mlm_logit = self.lm_head(sequence_output)
        
        
        
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        outputs= (logits,)+outputs[2:]

        loss = None
        if labels is not None:
               
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.squeeze(), labels)
            outputs =(loss,) + outputs
            
            # elif self.config.problem_type == "single_label_classification":
            #     loss_fct = CrossEntropyLoss()
            #     loss = loss_fct(mc_logits.view(-1, self.num_labels), labels.view(-1))
         

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(mlm_logit.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs=(masked_lm_loss,) + outputs

        return outputs



class DebertaForMultipleChoice_MLM(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.deberta = DebertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.pooler = ContextPooler(config)
        self.lm_head = DebertaOnlyMLMHead(config)
        # self.init_weights()
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.predictions.decoder = new_embeddings
    
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        masked_lm_labels=None
    ):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_masked_lm_labels = masked_lm_labels.view(-1, masked_lm_labels.size(-1)) if masked_lm_labels is not None else None
        
        
        outputs = self.deberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask
        )
        encoder_layer = outputs[0]
        if masked_lm_labels is not None:
            sequence_output = encoder_layer
            mlm_logit = self.lm_head(sequence_output)
        
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(mlm_logit.view(-1, self.config.vocab_size), flat_masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  #(masked_lm_loss,) (lm_loss), reshaped_logits, (hidden_states), (attentions)


class DebertaForMultipleChoice(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.deberta = DebertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.pooler = ContextPooler(config)
        # self.init_weights()
        self.post_init()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None
    ):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        
        outputs = self.deberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask
        )
        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  #(masked_lm_loss,) (lm_loss), reshaped_logits, (hidden_states), (attentions)


class BertForMultipleChoice_MLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.lm_head = BertOnlyMLMHead(config)
        self.post_init()
    
    def get_output_embeddings(self):
        return self.lm_head.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.predictions.decoder = new_embeddings
    
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        masked_lm_labels=None
    ):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_masked_lm_labels = masked_lm_labels.view(-1, masked_lm_labels.size(-1)) if masked_lm_labels is not None else None
        
        
        outputs = self.bert(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
        )
        
        pooled_output = outputs[1]

        if masked_lm_labels is not None:
            sequence_output = outputs[0]
            mlm_logit = self.lm_head(sequence_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(mlm_logit.view(-1, self.config.vocab_size), flat_masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  #(masked_lm_loss,) (lm_loss), reshaped_logits, (hidden_states), (attentions)




class AlbertForMultipleChoice_MLM(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.lm_head = AlbertMLMHead(config)
        self.post_init()
    
    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def get_input_embeddings(self):
        return self.albert.embeddings.word_embeddings
    
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        masked_lm_labels=None
    ):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_masked_lm_labels = masked_lm_labels.view(-1, masked_lm_labels.size(-1)) if masked_lm_labels is not None else None
        
        
        outputs = self.albert(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
        )
        
        pooled_output = outputs[1]

        if masked_lm_labels is not None:
            sequence_output = outputs[0]
            mlm_logit = self.lm_head(sequence_output)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(mlm_logit.view(-1, self.config.vocab_size), flat_masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  #(masked_lm_loss,) (lm_loss), reshaped_logits, (hidden_states), (attentions)
class GPT2ClsDoubleHeadsModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.cls_head = SequenceSummary(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        labels=None,
    ):

        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        cls_logits = self.cls_head(hidden_states, mc_token_ids).squeeze(-1)

        outputs = (lm_logits, cls_logits) + transformer_outputs[1:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(cls_logits, labels)
            outputs = (loss,) + outputs
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (lm loss), (mc loss), lm logits, mc logits, presents, (all hidden_states), (attentions)


