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

def embed_context(input_ids):
    if past is None:
        past_length = 0
        past = [None] * len(self.h)
    else:
        past_length = past[0][0].size(-2)
    if position_ids is None:
        position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                    device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_ids.size(-1))
    position_ids = position_ids.view(-1, position_ids.size(-1))

    inputs_embeds = self.wte(input_ids)
    position_embeds = self.wpe(position_ids)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        token_type_embeds = self.wte(token_type_ids)
    else:
        token_type_embeds = 0

    # create hidden_states using inputs_emb and position_emb and token_type_emb
    hidden_states = inputs_embeds + position_embeds + token_type_embeds
    return hidden_states

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True):

    context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)

    # embed context
    context = embed_context(context)

    prev = context
    output = context
    past = None
    
    with torch.no_grad():
        # for i in range(length): # length is 512
        for i in range(10):
            # run model
            # logits, past = model(prev, past=past)
            hidden_states, past = model(prev, past=past)
            
            # save hidden_states
            print(i, ": ", hidden_states)

            prev = hidden_states

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