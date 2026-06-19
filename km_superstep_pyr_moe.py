#commented out learning rate schedule for trying next time

#includes kmoment, engrammlp, routing long context(aka sparse attention except in most recent tokens), and sparse for 
#all tokens in later layers. should parse this out to see which is helpful. 
#maybe should try token routing separately for pairwise attention vs simplicial att. 
import os
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


sys.stdout.reconfigure(encoding="utf-8")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device is: ', device)

# parameters to tweak
max_iters = int(os.getenv('MAX_ITERS', '20001'))
eval_iters = int(os.getenv('EVAL_ITERS', '50'))
eval_interval = int(os.getenv('EVAL_INTERVAL', '5000'))
n_embed = int(os.getenv('N_EMBED', '128'))
bign_embed = 256
block_size = 64  # small context window size
long_block_size = 256  # long context window size
batch_size = int(os.getenv('BATCH_SIZE', '16'))
learning_rate = float(os.getenv('LEARNING_RATE', '3e-4'))
n_head = int(os.getenv('N_HEAD', '4'))
moment_n_layer = int(os.getenv('N_LAYER', '2'))  #number of kmoment attention blocks
n_ff_layers = int(os.getenv('N_FF_LAYERS', '2'))  # Extra FFN-only layers after attention blocks
att_n_layers = 2          #number of regular attention blocks
dropout = float(os.getenv('DROPOUT', '0.2'))
router_top_k = 32  # topk for the further back window
topk_start_layer = 1  # Layer index (0-based) to start doing topk for both windows
small_window_top_k = 16  # topk for the small window when active
n_kv_head = int(os.getenv('N_KV_HEAD', '2'))  # Number of KV heads for Grouped Query Attention (GQA)


#for moe portion
num_experts = 8
topk = 2

vocab_size = int(os.getenv('VOCAB_SIZE', '1000'))
num_merges = vocab_size - 256

engram_layer_ids = [1]  # Engram active on first and last layers
engram_max_ngram_size = 3
# engram_vocab_size = [1024, 1024]
engram_n_embed_per_ngram = n_embed
engram_n_head_per_ngram = 4
engram_kernel_size = 4

assert vocab_size >= 256, "vocab_size must include the 256 raw byte tokens"
assert n_embed % n_head == 0, "n_embed must divide evenly across n_head"
assert n_head % n_kv_head == 0, "n_head must be divisible by n_kv_head for GQA"
assert moment_n_layer > 0, "n_layer must be positive"


class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"""
        self.compiled_pattern = re.compile(self.pattern)

    def train(self, text, vocab_size, verbose=False):
        num_merges = vocab_size - 256
        text_chunks = self.compiled_pattern.findall(text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        for i in range(num_merges):
            stats = Counter()
            for chunk_ids in ids:
                for pair in zip(chunk_ids, chunk_ids[1:]):
                    stats[pair] += 1
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [self._merge(chunk_ids, pair, idx) for chunk_ids in ids]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (i + 1) % 100 == 0:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx}")

    def _merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def encode(self, text):
        all_ids = []
        for chunk in self.compiled_pattern.findall(text):
            chunk_ids = list(chunk.encode("utf-8"))
            while len(chunk_ids) >= 2:
                stats = Counter(zip(chunk_ids, chunk_ids[1:]))
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                chunk_ids = self._merge(chunk_ids, pair, self.merges[pair])
            all_ids.extend(chunk_ids)
        return all_ids

    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            part_bytes.append(self.vocab[idx])
        text_bytes = b"".join(part_bytes)
        return text_bytes.decode("utf-8", errors="replace")


def nearby_cache_path(filename):
    here = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / filename,
        here / filename,
        here.parent / filename,
        here.parent.parent / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return Path.cwd() / filename


cache_file = nearby_cache_path(f"wikitext_bpe_cache_v2_{vocab_size}.pkl")
tokenizer = BPETokenizer()

if cache_file.exists():
    print(f"Loading cached data from {cache_file}...")
    with open(cache_file, 'rb') as f:
        cache_data = pickle.load(f)
    data = cache_data['data']
    tokenizer.merges = cache_data['merges']
    tokenizer.vocab = cache_data['vocab']
else:
    if load_dataset is None:
        raise ImportError("datasets is required when the WikiText cache is not already available")

    print(f"Downloading and processing wikitext dataset...")
    textraw = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
    sample = textraw['train']
    text = " ".join(sample["text"])

    print("Training regex BPE tokenizer...")
    tokenizer.train(text, vocab_size, verbose=True)

    print("Encoding dataset...")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    print(f"Saving cache to {cache_file}...")
    with open(cache_file, 'wb') as f:
        pickle.dump({'data': data, 'merges': tokenizer.merges, 'vocab': tokenizer.vocab}, f)


def decode(ids):
    return tokenizer.decode(ids)


def encode(text):
    return tokenizer.encode(text)


print("vocab size ", vocab_size)
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

torch.manual_seed(1337)


def get_batch(split):
    data_src = train_data if split == 'train' else test_data
    ix = torch.randint(len(data_src) - long_block_size, (batch_size,))
    indices = ix.unsqueeze(1) + torch.arange(long_block_size)
    x = data_src[indices].to(device)
    y = data_src[indices + 1].to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Expert(nn.Module):
    """ An expert network, which is a simple feed-forward network. """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class MixtureOfExperts(nn.Module):
    """
    A Mixture of Experts layer.

    Args:
        n_embed (int): The embedding dimension.
        num_experts (int): The total number of expert networks.
        top_k (int): The number of experts to route each token to.
    """
    def __init__(self, n_embed, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # A list of expert networks
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        
        # The gating network is a linear layer that outputs a logit for each expert
        self.gate = nn.Linear(n_embed, num_experts)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, n_embed) -> b, t, c
        b, t, c = x.shape
        
        # Flatten the input for token-wise processing
        x_flat = x.view(-1, c) # -> (b*t, c)

        # 1. Gating: Get logits for each token and each expert
        gate_logits = self.gate(x_flat) # -> (b*t, num_experts)
        
        # 2. Routing: Find the top_k experts with the highest logits for each token
        # topk returns a tuple of (values, indices)
        top_k_logits, top_k_indices = gate_logits.topk(self.top_k, dim=-1) # -> (b*t, top_k)
        
        # 3. Normalize the weights of the selected experts using softmax
        top_k_weights = F.softmax(top_k_logits, dim=-1) # -> (b*t, top_k)
        
        # 4. Combine results: Weighted sum of expert outputs
        final_output_flat = torch.zeros_like(x_flat)
        
        # Get the indices of tokens and experts to be processed
        flat_token_indices = torch.arange(x_flat.size(0), device=x.device).repeat_interleave(self.top_k)
        flat_expert_indices = top_k_indices.view(-1)
        
        # Group inputs by expert to process them in batches
        # This is more efficient than looping through each token
        for i in range(self.num_experts):
            # Find which tokens are routed to this expert
            token_mask = (flat_expert_indices == i)
            if token_mask.any():
                # Get the indices of the tokens for the current expert
                expert_token_indices = flat_token_indices[token_mask]

                # Get the input for this expert
                expert_input = x_flat[expert_token_indices]
                
                # Process the input with the expert
                expert_output = self.experts[i](expert_input)
                
                # Get the corresponding weights
                weights_for_expert = top_k_weights.view(-1)[token_mask]
                
                # Weight the expert's output
                weighted_output = expert_output * weights_for_expert.unsqueeze(1)
                
                # Add the weighted output back to the final result tensor
                # index_add_ is an efficient in-place scatter-add operation
                final_output_flat.index_add_(0, expert_token_indices, weighted_output)

        # Reshape the output back to the original input shape
        return final_output_flat.view(b, t, c)



class EngramMLP(nn.Module):
    def __init__(self, n_embed, max_ngram_size, engram_n_embed_per_ngram, shared_embedding):
        super().__init__()
        self.max_ngram_size = max_ngram_size
        self.n_embed = n_embed
        self.engram_n_embed_per_ngram = engram_n_embed_per_ngram
        # Shared with the main token embedding table — no separate vocab lookup
        self.embedding = shared_embedding
        
        self.mlps = nn.ModuleList()
        for n in range(2, max_ngram_size + 1):
            mlp = nn.Sequential(
                nn.Linear(n * n_embed, 4 * n_embed),
                nn.SiLU(),
                nn.Linear(4 * n_embed, engram_n_embed_per_ngram)
            )
            self.mlps.append(mlp)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.embedding(input_ids)
        
        results = []
        for i, n in enumerate(range(2, self.max_ngram_size + 1)):
            pad = torch.zeros(B, n-1, self.n_embed, device=x.device)
            x_padded = torch.cat([pad, x], dim=1)
            
            # Using unfold to get sliding windows of size n
            windows = x_padded.unfold(1, n, 1) # (B, T, n_embed, n)
            windows = windows.transpose(2, 3) # (B, T, n, n_embed)
            # unfold views can be unstable for reshaping on some systems; make contiguous first
            windows = windows.contiguous().view(B, T, n * self.n_embed)
            
            res = self.mlps[i](windows)
            results.append(res)
            
        return torch.cat(results, dim=-1)



class EngramLayer(nn.Module):
    def __init__(self, layer_id, n_embed, shared_embedding):
        super().__init__()
        self.layer_id = layer_id
        
        self.engram_mlp = EngramMLP(
            n_embed=n_embed,
            max_ngram_size=engram_max_ngram_size,
            engram_n_embed_per_ngram=engram_n_embed_per_ngram,
            shared_embedding=shared_embedding,
        )
       
        engram_hidden_size = (engram_max_ngram_size - 1) * engram_n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, n_embed)
        self.key_proj = nn.Linear(engram_hidden_size, n_embed)
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states, input_ids):
        embeddings = self.engram_mlp(input_ids)

        key = self.key_proj(embeddings)
        normed_key = self.norm1(key)
        normed_query = self.norm2(hidden_states)
        
        gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(n_embed)
        # Use a more stable signed square root to avoid gradient discontinuities at zero
        gate = torch.sign(gate) * torch.sqrt(torch.abs(gate) + 1e-4)
        gate = torch.sigmoid(gate).unsqueeze(-1)
        
        value = gate * self.value_proj(embeddings)
        output = self.dropout(value) #+ self.short_conv(value)
        return output

# --- End Engram Implementation ---


class GatedRouter(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.router = nn.Linear(n_embed, 1)

    def forward(self, x):
        logits = self.router(x) # (B, T, 1)
        probs = torch.sigmoid(logits)
        gate = (probs > 0.5).float()
        return gate - probs.detach() + probs

class ParallelHybridAttention(nn.Module):
    def __init__(self, num_heads, head_size, mode='local', global_range=None, layer_idx=0, num_kv_heads=None):
        super().__init__()
        self.mode = mode
        self.global_range = global_range
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_queries_per_kv = num_heads // self.num_kv_heads
        
        # Local path projections
        self.q_local = nn.Linear(n_embed, num_heads * head_size, bias=False)
        self.k_local = nn.Linear(n_embed, self.num_kv_heads * head_size, bias=False)
        self.v_local = nn.Linear(n_embed, self.num_kv_heads * head_size, bias=False)
        
        # Global path projections if needed
        if mode != 'local':
            self.q_global = nn.Linear(n_embed, num_heads * head_size, bias=False)
            self.k_global = nn.Linear(n_embed, self.num_kv_heads * head_size, bias=False)
            self.moment_value = nn.Linear(n_embed, self.num_kv_heads * head_size, bias=False)
            self.moment_mix = nn.Linear(4 * head_size, head_size)
            self.moment_gate = nn.Linear(n_embed, num_heads * head_size)
            
            self.router = GatedRouter(n_embed)
            if mode == 'gated_routed':
                self.selection_router = nn.Linear(n_embed, 1)

        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer('tril', torch.tril(torch.ones(global_range, global_range)))

    def weighted_moments(self, values, weights):
        # values: (B, H, T, K, D), weights: (B, H, T, K)
        w = weights.unsqueeze(-1) # (B, H, T, K, 1)
        mean = (w * values).sum(dim=3) # (B, H, T, D)
        # Clamp centered values to prevent higher powers from exploding
        centered = (values - mean.unsqueeze(3)).clamp(-10.0, 10.0)
        
        variance = (w * centered.pow(2)).sum(dim=3)
        third = (w * centered.pow(3)).sum(dim=3)
        fourth = (w * centered.pow(4)).sum(dim=3)

        # Ensure stability with epsilon
        std = torch.sqrt(variance + 1e-4)
        skew = third / (std.pow(3) + 1e-4)
        kurtosis = fourth / (variance.pow(2) + 1e-4)
        return torch.cat([mean, variance, skew, kurtosis], dim=-1)

    def _attend(self, q, k, v, mask=None):
        b, h, t, d = q.shape
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
            
        wei = q @ k.transpose(-2, -1) * (d ** -0.5)
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        if mask is not None:
            wei = wei.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

    def _attend_moments(self, q, k, v_moment, mask):
        b, h, t, d = q.shape
        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v_moment = v_moment.repeat_interleave(self.num_queries_per_kv, dim=1)
            
        wei = q @ k.transpose(-2, -1) * (d ** -0.5)
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        if mask is not None:
            wei = wei.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        routed_k = min(router_top_k + small_window_top_k, t)
        top_scores, top_idx = torch.topk(wei, routed_k, dim=-1)
        weights = F.softmax(top_scores, dim=-1)
        weights = self.dropout(weights)

        # Gather values for moments
        # v_moment is (B, H, T_keys, D). We gather values for each query T_query.
        v_exp = v_moment.unsqueeze(2).expand(b, h, t, t, d) # (B, H, T_q, T_k, D)
        idx_exp = top_idx.unsqueeze(-1).expand(b, h, t, routed_k, d)
        gathered = torch.gather(v_exp, 3, idx_exp) # (B, H, T, K, D)
        
        moment_summary = self.weighted_moments(gathered, weights)
        return self.moment_mix(moment_summary)

    def forward(self, x):
        b, t, c = x.shape
        
        # 1. Mandatory Local Attention (last block_size)
        t_local = min(t, block_size)
        x_local = x[:, -t_local:, :]
        
        q_l = self.q_local(x_local).view(b, t_local, self.num_heads, self.head_size).transpose(1, 2)
        k_l = self.k_local(x_local).view(b, t_local, self.num_kv_heads, self.head_size).transpose(1, 2)
        v_l = self.v_local(x_local).view(b, t_local, self.num_kv_heads, self.head_size).transpose(1, 2)
        
        local_out = self._attend(q_l, k_l, v_l)
        local_out = local_out.transpose(1, 2).contiguous().view(b, t_local, self.num_heads * self.head_size)
        
        if t > block_size:
            local_out = F.pad(local_out.transpose(1, 2), (t - block_size, 0)).transpose(1, 2)
        
        if self.mode == 'local':
            out = local_out
        else:
            gate = self.router(x)
            
            global_mask = None
            if self.mode == 'gated_routed':
                scores = self.selection_router(x).squeeze(-1) # (B, T)
                
                cols = torch.arange(t, device=x.device).unsqueeze(0)  # (1, T)
                rows = torch.arange(t, device=x.device).unsqueeze(1)  # (T, 1)

                # Segment 1: Further back window (keys j <= i - block_size)
                back_mask = (rows - cols >= block_size)  # (T, T)
                back_scores_matrix = scores.unsqueeze(1).masked_fill(~back_mask, float('-inf'))  # (B, T, T)
                k_back = min(router_top_k, t)
                global_mask_back = torch.zeros((b, t, t), device=x.device)
                if k_back > 0:
                    topk_back_idx = torch.topk(back_scores_matrix, k=k_back, dim=-1).indices  # (B, T, k_back)
                    global_mask_back.scatter_(2, topk_back_idx, 1.0)
                    global_mask_back = global_mask_back * back_mask

                # Segment 2: Small window (keys i - block_size < j <= i)
                small_mask = (rows - cols >= 0) & (rows - cols < block_size)  # (T, T)
                if self.layer_idx < topk_start_layer:
                    global_mask_small = small_mask.float().expand(b, -1, -1)
                else:
                    small_scores_matrix = scores.unsqueeze(1).masked_fill(~small_mask, float('-inf'))  # (B, T, T)
                    k_small = min(small_window_top_k, t)
                    global_mask_small = torch.zeros((b, t, t), device=x.device)
                    if k_small > 0:
                        topk_small_idx = torch.topk(small_scores_matrix, k=k_small, dim=-1).indices  # (B, T, k_small)
                        global_mask_small.scatter_(2, topk_small_idx, 1.0)
                        global_mask_small = global_mask_small * small_mask

                global_mask = global_mask_back + global_mask_small
            
            q_g = self.q_global(x).view(b, t, self.num_heads, self.head_size).transpose(1, 2)
            k_g = self.k_global(x).view(b, t, self.num_kv_heads, self.head_size).transpose(1, 2)
            v_m = self.moment_value(x).view(b, t, self.num_kv_heads, self.head_size).transpose(1, 2)
            
            global_out = self._attend_moments(q_g, k_g, v_m, mask=global_mask)
            global_out = global_out.transpose(1, 2).contiguous().view(b, t, self.num_heads * self.head_size)
            
            # Per-head moment gating
            m_gate = torch.sigmoid(self.moment_gate(x))
            out = local_out + gate * (m_gate * global_out)
            
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class FFOnlyBlock(nn.Module):
    """A plain residual FFN block with pre-norm, no attention."""
    def __init__(self, n_embed):
        super().__init__()
        self.ln = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_embed)

    def forward(self, x):
        return x + self.ffwd(self.ln(x))


class MomentAttentionBlock(nn.Module):
    def __init__(self, n_embed, n_head, layer_id, shared_embedding):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = ParallelHybridAttention(n_head, head_size, mode='gated_routed', global_range=long_block_size, layer_idx=layer_id, num_kv_heads=n_kv_head)
        self.moe = MixtureOfExperts(n_embed, num_experts, topk)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.engram = EngramLayer(layer_id, n_embed, shared_embedding) if layer_id in engram_layer_ids else None

    def forward(self, x, idx):
        if self.engram is not None:
            x = x + self.engram(x, idx)
        x = x + self.sa(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x



class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(bign_embed, head_size, bias=False)
        self.query = nn.Linear(bign_embed, head_size, bias=False)
        self.value = nn.Linear(bign_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(long_block_size, long_block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b,t,c = x.shape
        k = self.key(x)
        q = self.query(x)
        #compute attention scores
        wei = q @ k.transpose(-2,-1) * c**-0.5
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf')) #(b,t,t)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        #perform weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out

class MultiheadAttention(nn.Module):
    def __init__(self,num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(bign_embed, bign_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out



class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiheadAttention(n_head, head_size)
        
        self.moe = MixtureOfExperts(n_embed, num_experts, topk)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(long_block_size, n_embed)
        # Pass the shared embedding so EngramMLP doesn't create a duplicate lookup table
        self.mblocks = nn.ModuleList([
            MomentAttentionBlock(n_embed, n_head=n_head, layer_id=i, shared_embedding=self.token_embedding_table)
            for i in range(moment_n_layer)
        ])
        # Optional stack of FFN-only residual layers after the attention blocks
        self.ff_layers = nn.ModuleList([FFOnlyBlock(n_embed) for _ in range(n_ff_layers)])
        # self.fourier_proj = RandomFourierFeatures(n_embed, bign_embed)
        self.proj = nn.Linear(n_embed, bign_embed)
        self.blocks = nn.Sequential(*[Block(bign_embed, n_head=n_head) for _ in range(att_n_layers)])
        self.ln_f = nn.LayerNorm(bign_embed)
        self.lm_head = nn.Linear(bign_embed, vocab_size)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        token_embed = self.token_embedding_table(idx)
        pos_embed = self.position_embedding_table(torch.arange(t, device=device))
        x = pos_embed + token_embed
        for block in self.mblocks:
            x = block(x, idx)
        for ff in self.ff_layers:
            x = ff(x)
        x = self.proj(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            logits = logits.view(b * t, c)
            targets = targets.view(b * t)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -long_block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    model = Transformer()
    total_params = sum(p.numel() for p in model.parameters())
    print('size of model', total_params)
    m = model.to(device)

    # Optional torch.compile JIT acceleration
    if os.getenv('COMPILE', '0') == '1':
        print("Compiling model with torch.compile...")
        m = torch.compile(m)

    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    start_time = time.time()

    use_amp = os.getenv('USE_AMP', '0') == '1'
    if use_amp:
        print("Using Automatic Mixed Precision (AMP) with bfloat16...")

    # Cosine LR decay with a short linear warmup
    # warmup_iters = 200
    # def get_lr(it):
    #     if it < warmup_iters:
    #         return learning_rate * (it + 1) / warmup_iters
    #     progress = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    #     return learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    for iter in range(max_iters):
        # Update LR
        # lr = get_lr(iter)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr

        if not iter % eval_interval:
            losses = estimate_loss(m)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')

        if use_amp and device == 'cuda':
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = m(xb, yb)
        else:
            logits, loss = m(xb, yb)
            
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        optimizer.step()

    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

    generate_tokens = int(os.getenv('GENERATE_TOKENS', '200'))
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=generate_tokens)[0].tolist()))