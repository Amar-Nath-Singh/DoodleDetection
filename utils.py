import torch
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from config import *

def predict(model, loader, le):
  model.eval()

  loop = tqdm(loader, leave=True)
  predictions = {'key_id':[],'word':[]}
  for data, key_id in loop:
    with torch.no_grad():
      pred = model(data)
    top_3_preds = pred.argsort(dim = 1, descending = True)[:,:3].cpu().detach().numpy()

    for p, k in zip(top_3_preds, key_id):
      classes = le.inverse_transform(p)
      predictions['key_id'].append(k.detach().numpy())
      predictions['word'].append(" ".join(classes))
  return pd.DataFrame.from_dict(predictions)