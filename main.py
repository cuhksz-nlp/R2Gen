import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.r2gen import R2GenModel
import timm
import time
from torch.cuda import amp
import json

def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json', help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    args = parser.parse_args()
    return args

def test_model(model, optimizer, metric_ftns, dataloader, epoch, device='cuda'):
    
  model.eval()
  with torch.no_grad():
    test_gts, test_res = [], []            

    for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(dataloader):

      images, reports_ids, reports_masks = images.to(device), reports_ids.to(device), reports_masks.to(
      device)
      optimizer.zero_grad()
      output = model(images, mode='sample')
      reports = model.tokenizer.decode_batch(output.cpu().numpy())
      ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
      test_gts.extend(reports)
      test_res.extend(ground_truths)
      
    gen_values = {i: [re] for i, re in enumerate(test_res)}
    
    filename = f"test_gen_values{epoch}.json"
    with open(filename, "w") as write_file:
        json.dump(gen_values, write_file, indent=4)

    test_met = metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                           {i: [re] for i, re in enumerate(test_res)})
    
    print(test_met)
    print()
    
def eval_model(model, criterion, optimizer, metric_ftns, scaler, epoch, scheduler, dataloader, device='cuda'):
    
  model.eval()
  with torch.no_grad():
    val_gts, val_res = [], []            

    for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(dataloader):

      images, reports_ids, reports_masks = images.to(device), reports_ids.to(device), reports_masks.to(
      device)
      optimizer.zero_grad()
      output = model(images, mode='sample')
      reports = model.tokenizer.decode_batch(output.cpu().numpy())
      ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
      val_res.extend(reports)
      val_gts.extend(ground_truths)

    val_met = metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                          {i: [re] for i, re in enumerate(val_res)})
    print(val_met)
    print()
    
def train_model(model, criterion, optimizer, metric_ftns, scheduler, train_dataloader, val_dataloader, test_dataloader, 
                scaler = None, num_epochs = 70, device='cuda', checkpoint = None):
    since = time.time()
    model = model.to(device)
    model.train()
    if checkpoint != None:
        print('Checkpoint found')
        start_epoch = checkpoint['epochs']
        print(f"Resuming training from {start_epoch}")
    else:
        start_epoch = 0
        print('------------TRAINING STARTED---------------------')

    for epoch in range(start_epoch, num_epochs):
        epoch_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0
        
        # Iterate over data.
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(train_dataloader):
            images, reports_ids, reports_masks = images.to(device), reports_ids.to(device), reports_masks.to(
            device)
            optimizer.zero_grad()

            if scaler is not None:
                with amp.autocast():
                    output = model(images, reports_ids, mode='train')
                    loss = criterion(output, reports_ids, reports_masks)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                with torch.set_grad_enabled(mode = True):
                    output = model(images, reports_ids, mode='train')
                    loss = criterion(output, reports_ids, reports_masks)
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
                    optimizer.step()

            running_loss += float(loss.item())
        
        epoch_loss = running_loss / len(train_dataloader)
        
        checkpoint = {'state_dict': model.state_dict(),
                      'optimizer' : optimizer.state_dict(),
                      'scheduler' : scheduler.state_dict(),
                      'epochs' : epoch}

        torch.save(checkpoint, '/content/drive/MyDrive/DL & ML Models/Automatic Medical Report Generation/R2Gen/ViTR2Gen/checkpoint_modified_vit_30epochs.pt')
        del checkpoint

        
        
        print('Train Loss: {:.4f}'.format(epoch_loss))
        eval_model(model, criterion, optimizer, metric_ftns, scaler, epoch, scheduler, val_dataloader, device='cuda')
        if epoch == 29:
          test_model(model, optimizer, metric_ftns, test_dataloader, epoch, device='cuda')
        print(f"Time taken for epoch {epoch}: {(time.time() - epoch_time) // 60:.0f}m {(time.time() - epoch_time) % 60:.0f}")
        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

def main():
    # parse arguments
    args = parse_agrs()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    model = R2GenModel(args, tokenizer)

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)
    
    scaler = amp.GradScaler()
    
    train_model(model, criterion, optimizer, metrics, lr_scheduler, train_dataloader, val_dataloader, test_dataloader, 
                None, args.epochs, checkpoint = None)

    # build trainer and start to train
    # trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
    # trainer.train()
    
    


if __name__ == '__main__':
    main()
