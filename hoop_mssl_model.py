import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

class HoopDualPositionalEncoding(nn.Module):
    def __init__(self, seq_len, num_players, d_model):
        super().__init__()
        self.spatial_emb = nn.Parameter(torch.randn(1, seq_len, 1, d_model))
        self.temporal_emb = nn.Parameter(torch.randn(1, 1, num_players, d_model))

    def forward(self, x):
        return x + self.spatial_emb + self.temporal_emb

class AxialTransformerBlock(nn.Module):
  def __init__(self, seq_len, emb_dim, num_heads, mlp_hidden_dim):
    super().__init__()
    self.N = seq_len
    self.D = emb_dim
    self.k = num_heads
    self.mlp_hidden_dim = mlp_hidden_dim
    self.dropout = torch.nn.Dropout(p=0.2)
    self.GeLU = torch.nn.GELU()

    #transformer
    self.msp_ln_1 = torch.nn.LayerNorm([self.N, self.D])
    self.msp_1 = torch.nn.MultiheadAttention(self.D, self.k, batch_first=True)
    self.mlp_ln_1 = torch.nn.LayerNorm([self.N, self.D])
    self.mlp_1 = torch.nn.Sequential(
        torch.nn.Linear(self.D, self.mlp_hidden_dim),
        self.GeLU,
        self.dropout,
        torch.nn.Linear(self.mlp_hidden_dim, self.D),
        self.dropout
    )

  def forward(self, x):
    residual = x
    x = self.msp_ln_1(x)
    x, _ = self.msp_1(x,x,x, need_weights=False)
    x = residual + x

    residual = x
    x = self.mlp_ln_1(x)
    x = self.mlp_1(x)
    return residual + x

class HoopTransformerBlock(nn.Module):
  def __init__(self, timesteps=121, agents=11, emb_dim=256, num_heads=16, transformer_mlp_hidden_dim=256):
    super().__init__()
    self.timesteps = timesteps
    self.agents = agents
    self.emb_dim = emb_dim
    self.flatten = nn.Flatten(start_dim=0, end_dim=1)
    self.spatial_transformer = AxialTransformerBlock(seq_len=timesteps, emb_dim=emb_dim, num_heads=num_heads, mlp_hidden_dim=transformer_mlp_hidden_dim)
    self.temporal_transformer = AxialTransformerBlock(seq_len=agents, emb_dim=emb_dim, num_heads=num_heads, mlp_hidden_dim=transformer_mlp_hidden_dim)

  def forward(self, x):
    # x: (32, 121, 11, 256)

    # temporal attention
    batch_size = x.shape[0]
    x = x.view(batch_size*self.timesteps, self.agents, self.emb_dim) # -> (32*121, 11, 256) WHY?!
    x = self.temporal_transformer(x) # -> (32*121, 11, 256)
    x = x.view(batch_size, self.timesteps, self.agents, self.emb_dim) # -> (32, 121, 11, 256)

    # spatial attention
    x = x.permute(0, 2, 1, 3).contiguous() # -> (32, 11, 121, 256)
    x = x.view(batch_size*self.agents, self.timesteps, self.emb_dim) # -> (32*11, 121, 256)
    x = self.spatial_transformer(x) # -> (32*11, 121, 256)
    x = x.view(batch_size, self.agents, self.timesteps, self.emb_dim) # -> (32, 11, 121, 256)
    x = x.permute(0, 2, 1, 3).contiguous()

    return x

class MotionReconstructionBlock(nn.Module):
  def __init__(self):
    super().__init__()
    self.mlp = nn.Sequential(nn.Linear(in_features=256, out_features=256),
                             nn.ReLU(),
                             nn.Dropout(p=0.3),
                             nn.Linear(in_features=256, out_features=64),
                             nn.ReLU(),
                             nn.Linear(in_features=256, out_features=3))

  def forward(self, x, mask_mask):
    return self.mlp(x[mask_mask]) #(number of masked tokens, 256) -> (number of masked tokens, 3)

class PlayerRoleIdentificationBlock(nn.Module):
  def __init__(self):
    super().__init__()
    # self.pool = nn.AdaptiveAvgPool1d(1)
    self.mlp = nn.Sequential(nn.Linear(in_features=256, out_features=256),
                             nn.ReLU(),
                             nn.Dropout(p=0.3),
                             nn.Linear(in_features=256, out_features=64),
                             nn.ReLU(),
                             nn.Linear(in_features=64, out_features=1))

  def forward(self, x):
    x = x.mean(dim=1)  # pooling to get player embeddings (B, A, D) 
    x = x[:, 1:, :]    # Remove ball -> (B, 10, D)
    logits = self.mlp(x)  # (B, 10, 1)
    return logits

class ContrastiveLearningBlock(nn.Module):
  def __init__(self, play_emb_dim=256):
    super().__init__()
    self.mlp = nn.Sequential(nn.Linear(in_features=256, out_features=256),
                             nn.ReLU(),
                             nn.Linear(in_features=256, out_features=play_emb_dim))


  def forward(self, x):
    x = x.mean(dim=(1, 2))  # Global average pooling
    x = self.mlp(x)  # (B, play_emb_dim)
    return x
  
class HoopMsslBase(nn.Module, ABC):
  def __init__(self, timesteps=121, agents=11, features=3, hidden_size=256):
    super().__init__()

    # masking (learnable token for 80% of the data)
    self.mask_token = nn.Parameter(torch.randn(1, 1, features))
    self.dropout = nn.Dropout(0.1)

    # mlp to linearly project the data into hidden size of 256
    self.init_proj = nn.Sequential(
        nn.Linear(features, 16),
        nn.LayerNorm(16),
        nn.ReLU(),
        
        # Step 2: 16 -> 64
        nn.Linear(16, 64),
        nn.LayerNorm(64),
        nn.ReLU(),
        
        # Step 3: 64 -> 256
        nn.Linear(64, hidden_size),
        nn.LayerNorm(hidden_size),
        nn.ReLU()
    )

    # dual positional encoding
    self.pos = HoopDualPositionalEncoding(timesteps, agents, hidden_size)

    # normalization
    self.norm = nn.LayerNorm(hidden_size)

    # 5 layers of axial-attention blocks
    self.b1 = HoopTransformerBlock()
    self.b2 = HoopTransformerBlock()
    self.b3 = HoopTransformerBlock()
    self.b4 = HoopTransformerBlock()
    self.b5 = HoopTransformerBlock()

    # motion reconstruction
    self.mr = MotionReconstructionBlock()

    # player-role identification
    self.pri = PlayerRoleIdentificationBlock()

    # contrastive learning
    self.cl = ContrastiveLearningBlock()

  def _apply_augmentations(self, x, apply_masking=True):
    B = x.shape[0]
    T = x.shape[1]
    A = x.shape[2]
    rand_noise = torch.rand(B, T, A-1)
    shuffled_idx = torch.argsort(rand_noise, dim=2) + 1

    shuffled_idx = torch.concatenate([torch.zeros(B, T, 1), shuffled_idx], dim=2)

    shuffled_idx_4d = shuffled_idx.view(B, T, A, 1).expand(B, T, A, 3).to(dtype=torch.long)

    x = torch.gather(x, dim=2, index=shuffled_idx_4d)

    # 2. Random masking
    mask_mask = None
    if apply_masking:
      mask_mask = torch.rand(x.shape[0], x.shape[1], x.shape[2]) < 0.8
      x[mask_mask] = self.mask_token

    # 3. Random dropout (feature-level)
    x = self.dropout(x)

    return x, mask_mask, shuffled_idx
  
  @abstractmethod
  def forward(self):
    pass
  
class HoopMssl_V1(HoopMsslBase):
  def forward(self, x: torch.Tensor):
    x = x.to(dtype=torch.float32)

    # 1. apply two augmentation and apply mask
    view1, mask_mask1, shuffled_idx1 = self._apply_augmentations(x)
    view2, mask_mask2, shuffled_idx2 = self._apply_augmentations(x)
    x = torch.cat([view1, view2], dim=0)  # (64, 121, 11, 3)

    # 2. project the data to 256 dim
    x = self.init_proj(x)

    # 3. add positional encoding
    x = self.pos(x)

    # 4. normalize
    x = self.norm(x)

    # 5. feed into transformer encoder
    x = x + self.b1(x)
    x = x + self.b2(x)
    x = x + self.b3(x)
    x = x + self.b4(x)
    x = x + self.b5(x)

    # 6. feed into downstream task decoders
    mask_mask = torch.cat([mask_mask1, mask_mask2], dim=0)
    mr_out = self.mr(x, mask_mask) #(number of masked, 256)
    pri_out = self.pri(x) # (64, 10, 2)
    cl_out = self.cl(x) # (64, 256)

    return mr_out, pri_out, cl_out, mask_mask1, mask_mask2, shuffled_idx1, shuffled_idx2
  
class PlayerLevelFineTuneHeadBlock(nn.Module):
  def __init__(self, out_dim):
    super().__init__()
    # self.pool = nn.AdaptiveAvgPool1d(1)
    self.mlp = nn.Sequential(
      nn.Linear(in_features=256, out_features=256),
      nn.LayerNorm(256),
      nn.ReLU(),
      nn.Dropout(p=0.2),
      
      nn.Linear(in_features=256, out_features=64),
      nn.ReLU(),
      
      nn.Linear(in_features=64, out_features=out_dim)
  )

  def forward(self, x):
    x = x.mean(dim=1)  # (B, A, D)
    x = x[:, 1:, :]    # Remove ball -> (B, 10, D) D is 256 (player embeddings)
    pred = self.mlp(x)  # (B, 10, 6)
    return pred

class PlayerLevelFineTuningModel(HoopMsslBase):
  def __init__(self, out_dim):
    super().__init__()
    self.height_head = PlayerLevelFineTuneHeadBlock(out_dim)

    #freeze backbone
    self.backbone = [self.mask_token, self.dropout, self.init_proj, self.pos, self.norm, self.b1, self.b2, self.b3, self.b4, self.b5]
    for block in self.backbone:
      for param in block.parameters():
        param.requires_grad = False

  def forward(self, x: torch.Tensor):
    x = x.to(dtype=torch.float32)

    # 1. apply two augmentation and apply mask
    x, mask_mask1, shuffled_idx1 = self._apply_augmentations(x, apply_masking=False)

    # 2. project the data to 256 dim
    x = self.init_proj(x)

    # 3. add positional encoding
    x = self.pos(x)

    # 4. normalize
    x = self.norm(x)

    # 5. feed into transformer encoder
    x = x + self.b1(x)
    x = x + self.b2(x)
    x = x + self.b3(x)
    x = x + self.b4(x)
    x = x + self.b5(x)

    # 6. regression head
    height_pred = self.height_head(x)

    return height_pred, shuffled_idx1
