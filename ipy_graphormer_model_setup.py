#%%
from transformers import GraphormerConfig,GraphormerForGraphClassification
# %%
cnfg = GraphormerConfig(
    num_classes = 3,
    embedding_dim = 128,
    num_attention_heads = 8,
    num_hidden_layers = 8
)

#num_classes: int = 1num_atoms: int = 4608num_edges: int = 1536
#num_in_degree: int = 512num_out_degree: int = 512
#num_spatial: int = 512num_edge_dis: int = 128
#multi_hop_max_dist: int = 5spatial_pos_max: int = 1024
#edge_type: str = 'multi_hop'max_nodes: int = 512
#share_input_output_embed: bool = False
#num_hidden_layers: int = 12embedding_dim: int = 768
#ffn_embedding_dim: int = 768num_attention_heads: int = 32
#dropout: float = 0.1attention_dropout: float = 0.1
#layerdrop: float = 0.0encoder_normalize_before: bool = False
#pre_layernorm: bool = Falseapply_graphormer_init: bool = False
#activation_fn: str = 'gelu'embed_scale: float = None
#freeze_embeddings: bool = Falsenum_trans_layers_to_freeze: int = 0
#traceable: bool = Falseq_noise: float = 0.0qn_block_size: int = 8
#kdim: int = Nonevdim: int = Nonebias: bool = True
#self_attention: bool = Truepad_token_id = 0bos_token_id = 1
#os_token_id = 2
# %%
gf = GraphormerForGraphClassification(cnfg)

# %%
gf
# %%
paramsum = pytorch_total_params = sum(p.numel() for p in gf.parameters() if p.requires_grad)
print(paramsum)
# %%
