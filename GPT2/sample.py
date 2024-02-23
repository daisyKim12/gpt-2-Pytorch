'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import torch
import torch.nn.functional as F
from tqdm import trange

def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True):

    # start_token is for starting generating text
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    # print("start_token: ", start_token)
    # print("context: ", context)
    # start_token:  None
    # context:  tensor([[31373,   616,  1438,   318]], device='cuda:0')
    prev = context
    output = context
    past = None
    with torch.no_grad():
        # for i in range(length): # length is 512
        for i in range(10):
            # run model
            logits, past = model(prev, past=past)

            print("<iteration " , i, ">")
            print("logits: (type)", type(logits), "(shape) ", logits.shape)
            print("past <kv cache>: (type)", type(past), "(size) ", len(past))
            # logits: (type) <class 'torch.Tensor'> (shape)  torch.Size([1, 1, 50257])
            # past <kv cache>: (type) <class 'list'> (size)  12
            
            ## inverse embedding after linear layer
            logits = logits[:, -1, :] / temperature 
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            
            print("prev: ", prev)
            # prev:  tensor([[661]], device='cuda:0')

            output = torch.cat((output, prev), dim=1)
    return output