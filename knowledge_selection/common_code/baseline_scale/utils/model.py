import torch
import torch.nn.functional as F
import logging


logger = logging.getLogger(__name__)

def run_batch_detection_train(args, model, batch):
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    
    if args.multi_task:
        input_ids, token_type_ids, labels, mlm_label = batch
        
        model_outputs = model(
            input_ids=input_ids, 
            token_type_ids=token_type_ids,
            labels=labels,
            masked_lm_labels=mlm_label
        )
        mlm_loss=model_outputs[0]
        mc_loss = model_outputs[1]
        mc_logits = model_outputs[2]
        loss=mlm_loss+mc_loss
    else:
        input_ids, token_type_ids, labels = batch
        
        model_outputs = model(
            input_ids=input_ids, 
            token_type_ids=token_type_ids,
            labels=labels,
        )
        mc_loss = model_outputs[0]
        mc_logits = model_outputs[1]
        loss=mc_loss     
        
    return loss, mc_logits, labels

def run_batch_detection_eval(args, model, batch):
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids,labels = batch
    
    model_outputs = model(
        input_ids=input_ids, 
        token_type_ids=token_type_ids,
        labels=labels
    )
    mc_loss = model_outputs[0]
    mc_logits = model_outputs[1]
    loss=mc_loss
        
    return loss, mc_logits, labels



def run_batch_selection_train(args, model, batch):
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    
    if args.multi_task:
        input_ids, token_type_ids, mc_labels, mlm_label = batch
        model_outputs = model(
            input_ids=input_ids, token_type_ids=token_type_ids,
            labels=mc_labels,
            masked_lm_labels=mlm_label
        )
        mlm_loss=model_outputs[0]
        mc_loss = model_outputs[1]
        mc_logits = model_outputs[2]
        #jang
        #debug
        # loss=mc_loss
        loss=mlm_loss+mc_loss
        
        return loss, mc_logits, mc_labels
    else:
        input_ids, token_type_ids, mc_labels = batch
        model_outputs = model(
            input_ids=input_ids, token_type_ids=token_type_ids,
            labels=mc_labels
        )
        mc_loss = model_outputs[0]
        mc_logits = model_outputs[1]
    
    
        return mc_loss, mc_logits, mc_labels




def run_batch_selection_eval(args, model, batch):
    candidates_per_forward = args.max_candidates_per_forward_eval * (args.n_gpu if isinstance(model, torch.nn.DataParallel) else 1)
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    
    input_ids, token_type_ids, mc_labels = batch

    all_mc_logits = []

    for index in range(0, input_ids.size(1), candidates_per_forward):
        model_outputs = model(
            input_ids=input_ids[0, index:index+candidates_per_forward].unsqueeze(1),
            token_type_ids=token_type_ids[0, index:index+candidates_per_forward].unsqueeze(1)
        )
        mc_logits = model_outputs[0]
        all_mc_logits.append(mc_logits.detach())
    all_mc_logits = torch.cat(all_mc_logits, dim=0).unsqueeze(0)
    return torch.tensor(0.0),all_mc_logits, mc_labels


def run_batch_selection_ranking(args, model, batch):
    candidate_keys=batch[-1]["candidate_keys"]
    candidates_per_forward = args.max_candidates_per_forward_eval * (args.n_gpu if isinstance(model, torch.nn.DataParallel) else 1)
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    
    input_ids, token_type_ids, _ = batch

    all_mc_logits = []

    for index in range(0, input_ids.size(1), candidates_per_forward):
        model_outputs = model(
            input_ids=input_ids[0, index:index+candidates_per_forward].unsqueeze(1),
            token_type_ids=token_type_ids[0, index:index+candidates_per_forward].unsqueeze(1)
        )
        mc_logits = model_outputs[0]
        all_mc_logits.append(mc_logits.detach())
    
    all_mc_logits = torch.cat(all_mc_logits, dim=0).view([-1])
    
    return  all_mc_logits, candidate_keys