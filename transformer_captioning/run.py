
import argparse
import torch
from utils import * 
from torch.utils.data import DataLoader
from trainer import Trainer
from transformer import TransformerDecoder
from matplotlib import pyplot as plt

# the three settings from the README (and the swiglu run) without editing the file each time
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='case1')
parser.add_argument('--num_heads', type=int, default=2)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--activation', default='relu', choices=['relu', 'swiglu'])
args = parser.parse_args()

set_all_seeds(42) ### DO NOT CHANGE THIS LINE
exp_name = args.exp_name

train_dataset = CocoDataset(load_coco_data(max_train=1024), 'train')
train_dataloader =  DataLoader(train_dataset, batch_size=64)

val_dataset = CocoDataset(load_coco_data(max_val = 1024), 'val')
val_dataloader =  DataLoader(val_dataset, batch_size=64)


device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
transformer = TransformerDecoder(
          word_to_idx=train_dataset.data['word_to_idx'],
          idx_to_word = train_dataset.data['idx_to_word'],
          input_dim=train_dataset.data['train_features'].shape[1],
          embed_dim=256,
          num_heads=args.num_heads,
          num_layers=args.num_layers,
          max_length=30,
          device = device,
          activation=args.activation,
        )

trainer = Trainer(transformer, train_dataloader, val_dataloader,
          num_epochs=100,
          learning_rate=args.lr,
          device = device
        )

trainer.train()
print('final train loss: %.4f   min train loss: %.4f' % (trainer.loss_history[-1], min(trainer.loss_history)))

# Plot the training losses.
plt.plot(trainer.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
os.makedirs('plots', exist_ok=True)
plt.title('Training loss history')
plt.savefig('plots/' + exp_name + '_loss_out.png')


def vis_imgs(split):
    data = {'train': train_dataset.data, 'val': val_dataset.data}[split]
    loader = {'train': train_dataloader, 'val': val_dataloader}[split]
    num_imgs = 0 
    for batch in loader:
      features, gt_captions, idxs = batch
      urls = data["%s_urls" % split][idxs]
      
      gt_captions = decode_captions(gt_captions, transformer.idx_to_word)
      sample_captions = transformer.sample(features, max_length=30)
      sample_captions = decode_captions(sample_captions, transformer.idx_to_word)
      
      for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
          img = image_from_url(url)
          # Skip missing URLs.
          if img is not None: 
            plt.imshow(img)            
            plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
            plt.axis('off')
            plt.savefig('plots/' + exp_name + '_%s_%d.png' % (split, num_imgs))
            num_imgs += 1
            if num_imgs >= 5: break
      return 

vis_imgs('train')
vis_imgs('val')