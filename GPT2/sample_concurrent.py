'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import torch
import torch.nn.functional as F
import torch.nn as nn
from GPT2.config import GPT2Config
from tqdm import trange

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

def sample_sequence(model, inv_model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True):

    context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)

    prev = context              # saving previous token 
    outputs = None            # appending token in every step
    past = None                 # kv cache
    result = None
    
    with torch.no_grad():


        print("< iteration - >")
        print("SAMPLE > pre: ", prev.shape)
        hidden_states, past = model(prev, past=past)

        """ need to fix here """
        outputs = hidden_states[0].unsqueeze(0)
        prev = hidden_states[0][-1].unsqueeze(0).unsqueeze(0)
        print("SAMPLE > outputs: ", outputs.shape)
        print("SAMPLE > prev: ", prev.shape)

        # for i in range(length): # length is 512
        for i in range(3):
            # run model
            # logits, past = model(prev, past=past)
            print("< iteration ", i, ">")
            print("SAMPLE > pre: ", prev.shape)
            hidden_states, past = model(prev, past=past)


            """ need to fix here """
            prev = hidden_states[0][-1].unsqueeze(0).unsqueeze(0)
            outputs = torch.cat((outputs, prev), dim = 1)
            print("SAMPLE > prev: ", prev.shape)
            print("SAMPLE > outputs: ", outputs.shape)
        
        print("SAMPLE > total output: ", outputs.shape)
        logits = inv_model(outputs)
        print("SAMPLE > logits: ", logits, logits.shape)
    
        """ starting here """
        # inverse embedding after linear layer

        vector = logits[:, 0, :] / temperature 
        print(vector)
        vector = top_k_logits(vector, k=top_k)
        log_probs = F.softmax(vector, dim=-1)
        vector = torch.multinomial(log_probs, num_samples=1)
        result = vector
        
        for i in range(1, logits.shape[1]):

            vector = logits[:, i, :] / temperature 
            print(vector)
            vector = top_k_logits(vector, k=top_k)
            log_probs = F.softmax(vector, dim=-1)
            vector = torch.multinomial(log_probs, num_samples=1)

            result = torch.cat((result, vector), dim = 1)

        print("SAMPLE > result: ", result, '\n', result.shape)

    return result