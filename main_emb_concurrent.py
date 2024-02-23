'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import os
import sys
import torch
import random
import argparse
import numpy as np
import time
from GPT2.model_emb_concurrent import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder

def append_runtime(filename, runtime):
    with open(filename, 'a') as file:
        file.write(str(runtime) + "\n")

def text_generator(state_dict):
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--quiet", type=bool, default=False)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    if args.quiet is False:
        print(args)

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("model running on gpu...")
        device = torch.device("cuda")
    else:
        print("!!!!!WARNING!!!!!: cannot execute using cuda, operating on cpu")
        device = torch.device("cpu")

    # Load Model
    enc = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    if args.length == -1:
        args.length = config.n_ctx // 2
    elif args.length > config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)

    print("input text: ", args.text)
    context_tokens = enc.encode(args.text)
    print("context_tokens: ", context_tokens)
    # input text:  hello my name is
    # context_tokens:  [31373, 616, 1438, 318]

    # need to embed context_token

    generated = 0
    for _ in range(args.nsamples // args.batch_size):
        start_time = time.time_ns()

        print("=" * 40 + " RUN MODEL " + "=" * 40)
        out = sample_sequence(
            model=model, length=args.length,
            context=context_tokens  if not  args.unconditional else None,
            start_token=enc.encoder['<|endoftext|>'] if args.unconditional else None,
            batch_size=args.batch_size,
            temperature=args.temperature, top_k=args.top_k, device=device
        )

        print("printing out list ...")
        print(out)
        end_time = time.time_ns()
        run_time = (end_time - start_time)/(10 ** 9)
        out = out[:, len(context_tokens):].tolist()     # slicing out input token
        print("=" * 40 + " ANALYSIS " + "=" * 40)
        print("output length:", len(out[0]))
        print("run_time:", run_time, "sec")
        print("run_time per token:", run_time / len(out[0]), "sec")
        append_runtime('runtimes.txt', run_time / len(out[0]))

        # inverse-embedding
        
        # scalar to string
        for i in range(args.batch_size):
            generated += 1
            text = enc.decode(out[i])
            if args.quiet is False:
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            print(text)

if __name__ == '__main__':
    if os.path.exists('gpt2-pytorch_model.bin'):
        state_dict = torch.load('gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
        text_generator(state_dict)
        
    else:
        print('Please download gpt2-pytorch_model.bin')
        sys.exit()
