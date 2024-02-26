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

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True):

    context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)

    prev = context              # saving previous token 
    output = None            # appending token in every step
    past = None                 # kv cache
    
    with torch.no_grad():
        # for i in range(length): # length is 512
        for i in range(10):
            # run model
            # logits, past = model(prev, past=past)
            print("< iteration ", i, ">")
            print("hidden_states: ", prev.shape)
            hidden_states, past = model(prev, past=past)

            prev = hidden_states[0][-1].unsqueeze(0).unsqueeze(0)


    # # concurrent token generation
    # # linear layer

    # # inverse-embedding
    # print("<iteration " , i, ">")
    # print("logits: (type)", type(logits), "(shape) ", logits.shape)
    # print("past <kv cache>: (type)", type(past), "(size) ", len(past))
    # # logits: (type) <class 'torch.Tensor'> (shape)  torch.Size([1, 1, 50257])
    # # past <kv cache>: (type) <class 'list'> (size)  12
    
    # # inverse embedding after linear layer
    # logits = logits[:, -1, :] / temperature 
    # logits = top_k_logits(logits, k=top_k)
    # log_probs = F.softmax(logits, dim=-1)
    # if sample:
    #     prev = torch.multinomial(log_probs, num_samples=1)
    # else:
    #     _, prev = torch.topk(log_probs, k=1, dim=-1)
    
    # print("prev: ", prev)
    # # prev:  tensor([[661]], device='cuda:0')

    # output = torch.cat((output, prev), dim=1)

    return output