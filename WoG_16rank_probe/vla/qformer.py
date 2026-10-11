import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import BertConfig, BertModel


class QFormer(nn.Module):
    def __init__(self, hidden_dim, nheads, dim_feedforward, num_layers=2, num_query_tokens=16):
        super().__init__()

        qformer_config = BertConfig(
            hidden_size=hidden_dim,
            num_attention_heads=nheads,
            intermediate_size=dim_feedforward,
            num_hidden_layers=num_layers,
            add_cross_attention=True,
            is_decoder=True,
        )
        self.bert = BertModel(config=qformer_config)
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_dim))
        self.v_proj = nn.Linear(hidden_dim, 32)
        self.vae_proj = nn.Linear(784, hidden_dim)

    def encode_queries(self, visual_src, vae_latent=None):
        """Wide query features before v_proj (future Stage-3 readout interface).

        No parameters or checkpoint keys are added; forward remains identical.
        These queries are NOT per-timestep states.
        """
        if vae_latent is not None:
            vae_latent = self.vae_proj(vae_latent)
            visual_src = torch.cat([visual_src, vae_latent], dim=1)

        query_embeds = self.query_tokens.expand(visual_src.shape[0], -1, -1)

        outputs = self.bert(inputs_embeds=query_embeds, encoder_hidden_states=visual_src, return_dict=True)

        return outputs.last_hidden_state

    def forward(self, visual_src, vae_latent=None):
        return self.v_proj(self.encode_queries(visual_src, vae_latent))
