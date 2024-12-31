from transformers import AutoModel, PreTrainedModel
import torch
from model import GPT
import json
import os
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel, PretrainedConfig

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config, idx_layer):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.idx_layer = idx_layer
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.scale_attn_by_inverse_layer_idx:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)) / float(self.idx_layer + 1))
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) 
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config, idx_layer):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, idx_layer)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig(PretrainedConfig):
    """定义模型配置"""
    model_type = "custom_model"
    
    def __init__(
        self,
        block_size: int = 1024,
        vocab_size: int = 50304, # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        bias: bool = True, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        scale_attn_by_inverse_layer_idx: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.n_head = n_head
        self.dropout = dropout
        self.bias = bias
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        
class GPT(PreTrainedModel):
    config_class = GPTConfig
    def __init__(self, config):
        super().__init__(config)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, idx_layer) for idx_layer in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            if not isinstance(targets, int):
                logits = self.lm_head(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            else:
                logits = self.lm_head(x)
                loss = None
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

def convert_ckpt_to_hf(
    original_model,
    hf_model_class: PreTrainedModel,
    save_directory: str,
    config = None,
    weight_map = None,
):
    """
    将PyTorch checkpoint转换为Hugging Face格式
    
    Args:
        ckpt_path: checkpoint文件路径
        original_model_class: 原始PyTorch模型类
        hf_model_class: Hugging Face模型类
        save_directory: 保存目录
        config: Hugging Face配置对象
        weight_map: 权重映射字典，用于将原始模型的权重名称映射到HF模型
    """

    hf_model = hf_model_class(config)
    
    # 创建新的state dict
    new_state_dict = original_model.state_dict()
    
    # 加载转换后的权重
    missing_keys, unexpected_keys = hf_model.load_state_dict(
        new_state_dict, 
        strict=False
    )
    
    # 创建保存目录
    os.makedirs(save_directory, exist_ok=True)

    hf_model.save_pretrained(save_directory, safe_serialization=False)
    
    # 创建转换信息文件
    conversion_info = {
        "original_checkpoint": ckpt_path,
        "missing_keys": [str(k) for k in missing_keys],
        "unexpected_keys": [str(k) for k in unexpected_keys],
        "weight_map": weight_map
    }
    
    with open(os.path.join(save_directory, "conversion_info.json"), "w") as f:
        json.dump(conversion_info, f, indent=2)
    print(hf_model)
    return hf_model

if __name__ == "__main__":
    import argparse
    from model import GPT as old_GPT
    parser = argparse.ArgumentParser(description='Training Script with Job ID')

    # Single train_data parameter to specify the dataset prefix
    parser.add_argument('--n_layer', type=int, default=12, help='n_layer')
    parser.add_argument('--n_head', type=int, default=12, help='n_head')
    parser.add_argument('--n_embd', type=int, default=768, help='n_embd')
    parser.add_argument('--block_size', type=int, default=1024, help='block_size')
    
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    # Other parameters
    parser.add_argument('--ckpt_path', type=str, default='out_small_muon_100k/ckpt.pt', help='Checkpoint Path')
    parser.add_argument('--save_dir', type=str, default="./out", help='Save Path')
    parser.add_argument('--save_tokenizer', action='store_true', help='Save Tokenizer')
    parser.add_argument('--local_tokenizer_pth', type=str, default='', help='Local Tokenizer Path')
    args = parser.parse_args()
    model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, block_size=args.block_size,
                  bias=False, vocab_size=None, dropout=args.dropout, scale_attn_by_inverse_layer_idx=True) # start with model_args from command line
    ckpt_path = args.ckpt_path
    device = "cuda"
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = old_GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    convert_ckpt_to_hf(original_model=model,
                hf_model_class=GPT,
                save_directory=args.save_dir,
                config=gptconf)
    if args.save_tokenizer:
        from transformers import AutoTokenizer
        if args.local_tokenizer_pth:
            tokenizer = AutoTokenizer.from_pretrained(args.local_tokenizer_pth, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.save_pretrained(args.save_dir)