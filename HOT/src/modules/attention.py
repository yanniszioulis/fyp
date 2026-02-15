import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import math
from einops import einsum, rearrange


def compute_avg_stable_rank(att):
    ## bs, l, l, nh
    bs, _, _, nh = att.shape
    fro_norm = torch.linalg.norm(att, ord='fro', dim=(1,2))
    second_norm = torch.linalg.norm(att, ord=2, dim=(1,2))
    rank = fro_norm.square()/second_norm.square()      # bs, nh
    return rank.mean()

class KroneckerAttention(nn.Module):
    def __init__(
        self, 
        num_modes,
        d_model,
        n_head,
        dropout=0.,
        rotary_emb=None,
        mode='product',
        rope_dims=[]
    ):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_model // n_head
        self.rotary_emb = rotary_emb
        self.query_proj = nn.Linear(d_model, d_model*num_modes)
        self.key_proj = nn.Linear(d_model, d_model*num_modes)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.att_dropout = nn.Dropout(p=dropout)
        self.proj_dropout = nn.Dropout(p=dropout)
        self.rope_dims = rope_dims
        self.mode = mode
        self.q_norm = nn.LayerNorm(self.d_head)
        self.k_norm = nn.LayerNorm(self.d_head)

    
    def compute_attention(self, query, key, value, dim, use_rope=True):
        
        def pool(x):
            return einsum(x, 'bs ... l nh dh -> bs l nh dh')        

        l = query.shape[dim]
        q = query.transpose(dim, -2)   #(bs ... l d)  
        k = key.transpose(dim, -2)     #(bs ... l d)
        v = value.transpose(dim, -3)   #(bs ... l nh dh)

        q = q.unflatten(dim=-1, sizes=(self.n_head, self.d_head))   # (bs ... l nh dh)
        k = k.unflatten(dim=-1, sizes=(self.n_head, self.d_head))   # (bs ... l nh dh)

        q = self.q_norm(pool(q))
        k = self.k_norm(pool(k))
        if use_rope:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)

        att = einsum(q, k, 'bs l1 nh d, bs l2 nh d -> bs l1 l2 nh') / math.sqrt(q.shape[3])
        att = self.att_dropout(F.softmax(att, dim=2))
        h = einsum(att, v, 'bs l1 l2 nh, bs ... l2 nh d -> bs ... l1 nh d')
        return h.transpose(dim, -3), att

    def _compute_avg_kronecker_stable_rank(self, atts, mode):
        if mode == 'product':
            fro_norms = torch.stack([
                torch.linalg.norm(att, ord='fro', dim=(1,2)) for att in atts
            ])
            second_norms = torch.stack([
                torch.linalg.norm(att, ord=2, dim=(1,2)) for att in atts
            ])
            ranks = torch.prod(fro_norms.square()/second_norms.square(), dim=0)

        elif mode == 'sum':
            lengths = [att.shape[1] for att in atts] # bs, l, l, nh
            k = len(lengths)
            bs, _, _, nh = atts[0].shape
            fro_sizes = torch.tensor([
                math.prod(lengths[:i] + lengths[i+1:]) for i in range(k)
            ]).unsqueeze(1).unsqueeze(1).to(atts[0].device)

            # ord, bs, nh
            fro_norms = torch.stack([
                torch.linalg.norm(att, ord='fro', dim=(1,2)) for att in atts
            ]).square()
            fro_norms = (fro_norms * fro_sizes).sum(0) # bs, nh
            
            second_norms = torch.stack([
                torch.linalg.norm(att, ord=2, dim=(1,2)) for att in atts
            ]).sum(0).square() 
            
            traces = [einsum(att, 'bs l l nh -> bs nh') for att in atts]
            trace_prods = 0
            for i in range(k):
                for j in range(i+1, k):
                    trace_prods += 2 * torch.prod(torch.tensor(lengths[:i] + lengths[i+1:j] + lengths[j+1:])) * traces[i] * traces[j]

            ranks = (fro_norms + trace_prods)/second_norms
        return ranks.mean()


    def forward(self, X):
        query = self.query_proj(X).split(self.d_model, dim=-1)
        key = self.key_proj(X).split(self.d_model, dim=-1)
        value = self.value_proj(X).unflatten(dim=-1, sizes=(self.n_head, self.d_head))
        
        if self.mode == 'product':
            for idx, dim in enumerate(range(1, len(X.shape) - 1)):
                value, _ = self.compute_attention(
                    query=query[idx], 
                    key=key[idx],
                    value=value,
                    dim=dim,
                    use_rope=self.rotary_emb is not None and dim in self.rope_dims
                )

            value = value.flatten(start_dim=-2)
        elif self.mode == 'sum':
            res = 0
            for idx, dim in enumerate(range(1, len(X.shape) - 1)):
                v, _ = self.compute_attention(
                    query=query[idx], 
                    key=key[idx],
                    value=value,
                    dim=dim,
                    use_rope=self.rotary_emb is not None and dim in self.rope_dims
                )
                res += v
            value = res.flatten(start_dim=-2)

        return self.proj_dropout(self.out_proj(value))


class StandardAttention(nn.Module):
    def __init__(
        self, 
        d_model,
        n_head,
        dropout=0.,
        rotary_emb=None
    ):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.rotary_emb = rotary_emb


    def _compute_attention(self, q, k, v):
        att = einsum(q, k, 'bs l1 nh d, bs l2 nh d -> bs l1 l2 nh')
        att = F.softmax(att / math.sqrt(q.shape[3]), dim=2)
        h = einsum(att, v, 'bs l1 l2 nh, bs l2 nh d -> bs l1 nh d')
        return self.dropout(h)
     

    def forward(self, X):
        original_dims = X.shape[1:-1]
        query = self.query_proj(X).unflatten(dim=-1, sizes=(self.n_head, self.d_head)) 
        key = self.key_proj(X).unflatten(dim=-1, sizes=(self.n_head, self.d_head))
        value = self.value_proj(X).unflatten(dim=-1, sizes=(self.n_head, self.d_head))

        query = query.flatten(start_dim=1, end_dim=-3)
        key = key.flatten(start_dim=1, end_dim=-3)
        value = value.flatten(start_dim=1, end_dim=-3)

        if self.rotary_emb is not None:
            query = self.rotary_emb(query)
            key = self.rotary_emb(key)
        
        h = self._compute_attention(query, key, value)
        h = h.unflatten(dim=1, sizes=original_dims)
        return self.out_proj(h.flatten(start_dim=-2))


class DividedAttention(nn.Module):
    """TimeSFormer Divided Space-Time Attention"""
    
    def __init__(self, d_hidden, n_head=8, dropout=0., rotary_emb=None):
        super().__init__()
        self.temporal_att = StandardAttention(d_hidden, n_head, dropout=dropout, rotary_emb=rotary_emb)
        self.temporal_drop = nn.Dropout(dropout)
        self.temporal_norm = nn.LayerNorm(d_hidden)
        
        self.spatial_att = StandardAttention(d_hidden, n_head, dropout=dropout)
        self.spatial_drop = nn.Dropout(dropout)
        self.spatial_norm = nn.LayerNorm(d_hidden)
        
    def forward(self, x):
        # Temporal attention
        b, h, w, t, d = x.shape
        xt = rearrange(x, 'b h w t d -> (b h w) t d')
        xt = self.temporal_drop(self.temporal_att(self.temporal_norm(xt)))
        xt = x + rearrange(xt, '(b h w) t d -> b h w t d', b=b, h=h, w=w)
        
        
        # Spatial attention
        xs = rearrange(xt, 'b h w t d -> (b t) (h w) d')
        xs = self.spatial_drop(self.spatial_att(self.spatial_norm(xs)))
        xs = xt + rearrange(xs, '(b t) (h w) d -> b h w t d', b=b, t=t, h=h, w=w)
        return xs


class MultiScaleAttention(nn.Module):
    def __init__(self, d_hidden, input_size, n_head=8, kernel=5, dropout=0.):
        super().__init__()
        self.att = torchvision.models.video.mvit.MultiscaleAttention(
            input_size=[input_size]*3,
            embed_dim=d_hidden,
            output_dim=d_hidden,
            num_heads=n_head,
            kernel_q=[kernel]*3,
            kernel_kv=[kernel]*3,
            stride_q=[1,1,1],
            stride_kv=[1,1,1],
            residual_pool=False,
            residual_with_cls_embed=False,
            rel_pos_embed = False,
            dropout = dropout,
        )

    def forward(self, x):
        b, w, h, t, d = x.shape
        temp_cls = torch.zeros(b, 1, d).to(x.device).float()
        x = torch.cat([temp_cls, x.flatten(1,3)], dim=1)
        x, _ = self.att(x, (w, h, t))
        x = x[:, 1:].unflatten(1, (w,h,t))
        return x
