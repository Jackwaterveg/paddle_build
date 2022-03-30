import logging
import os
import random
import sys
import numpy as np
from pprint import pprint
import paddle
import yaml
import torch

root_dir="exp/conformer"
model_path= os.path.join(root_dir, "init.pt")
save_path=os.path.join(root_dir, "init.pdparams")
model_state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)


def torch_paddle_param(model_state_dict):
    paddle_state_dict = {}
    for n, p in model_state_dict.items():
        print(f'-> name: {n} dim: {p.ndim}')

        # change norm.mean and norm variance
        name_change=True
        if 'norm.running_mean' in n:
            new_n = n.replace('norm.running_', 'norm._')
        elif 'norm.running_var' in n:
            new_n = n.replace('norm.running_var', 'norm._variance')
        else:
            name_change=False
            new_n = n
        if name_change:
            print(f"\t norm mean/var: {n} -> {new_n}")

        p = p.cpu().detach().numpy()

        # weight which is rank 2, transpose it
        if n.endswith('weight') and p.ndim == 2:
            new_p = p.T
            print(f"\t liner transpose: {n}: {p.shape} -> {new_p.shape}")
        else:
            new_p = p

        # text embedding layer
        if 'decoder.embed.0.weight' in n:
            new_p = p

        if 'global_cmvn.mean' in n:
            print("\t cmvn:", p, p.dtype)
        if 'global_cmvn.istd' in n:
            print("\t istd:", p, p.dtype)

        paddle_state_dict[new_n] = new_p
    return paddle_state_dict

paddle_model_state_dict = torch_paddle_param(model_state_dict)

eq = []
# for key, val in paddle_model_state_dict.items():
#     print (key)

for key, val in model_state_dict.items():
#     print (key)
    if 'decoder.embed.0.weight' in key:
        # text embeding
        val = val
    elif key.endswith('weight') and val.ndim == 2:
        # linear weight
        val = val.T
    try:
        eq.append(np.allclose(val, paddle_model_state_dict[key]))
    except:
        if 'norm.running_mean' in key:
            new_key = key.replace('norm.running_', 'norm._')
        elif 'norm.running_var' in key:
            new_key = key.replace('norm.running_var', 'norm._variance')
        eq.append(np.allclose(val, paddle_model_state_dict[new_key]))
print("all param equal:", np.array(eq).all())

paddle.save(paddle_model_state_dict, save_path)
