import torch
import numpy as np

class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, dir, target_fn=None, target_col=None):
    super().__init__()
    self.dir = dir
    self.game_paths = sorted(list(dir.rglob("*.npy")))
    self.target_fn = target_fn
    self.target_col = target_col

    if (target_col is None and target_fn is None) or (target_col is not None and target_fn is not None):
       raise ValueError("provide either target_fn or target_col")

    self.samples = []

    for game_path in self.game_paths:
       data = np.load(game_path, mmap_mode='r')
       num_possessions = data.shape[0]
       
       for idx in range(num_possessions):
            self.samples.append((game_path, idx))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, i):
    game_path, idx = self.samples[i]
    possession = np.load(game_path, mmap_mode='r')[idx] 
    target = None
    if (self.target_col is None) and (self.target_fn):
        offense_team_id = self.target_fn(possession)
        target = (possession[0, :, 0][possession[0, :, 0] != -1] == offense_team_id)
        if target.sum() != 5:
            raise ValueError(f"invalid possession as offensive players are not 5, got {target.sum()}")
    elif (self.target_fn is None) and (self.target_col):
       target = possession[0, 1:, self.target_col]
       if self.target_col == 10:
          target[target == 7] = 6
    target = torch.from_numpy(target.copy())
    possession = possession[:, :, [2, 3, 8]] #extract x, y, speed
    possession = torch.from_numpy(possession.copy()).to(dtype=torch.float32)

    return possession, target
  
