import torch
import random

class BinSimDataset(torch.utils.data.Dataset):

  def __init__(self, args, raw_ds):
    self.args = args
    self.raw_ds = raw_ds

  def __len__(self):
    return len(self.raw_ds)
  
  def __getitem__(self, idx):
    # 1 positive sample, 1 negative samples
    selected = self.raw_ds[idx%len(self.raw_ds)]
    first_selected_function = selected['functions'][0]
    # randomly pick an int from 0 to len(selected['functions']) - 1
    random_idx = random.randint(1, len(selected['functions']) - 1)
    pos_selected_function = selected['functions'][random_idx]
    # randomly pick an int from 0 to len(self.raw_ds) - 1
    random_idx = random.randint(0, len(self.raw_ds) - 1)
    while random_idx == idx%len(self.raw_ds):
      random_idx = random.randint(0, len(self.raw_ds) - 1)
    neg_selected_function = random.choice(self.raw_ds[random_idx]['functions'])
    return {
      'functions': [first_selected_function, pos_selected_function, neg_selected_function]
    }
    
