import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.distributed as dist

from Dataset import get_loader
import math
from Models.PVST import PVST
import os

# Function to save loss metrics to a file
def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch):
    fh = open(save_dir, 'a')
    epoch_total_loss = str(epoch_total_loss)
    epoch_loss = str(epoch_loss)
    fh.write('until_' + str(epoch) + '_run_iter_num' + str(whole_iter_num) + '\n')
    fh.write(str(epoch) + '_epoch_total_loss' + epoch_total_loss + '\n')
    fh.write(str(epoch) + '_epoch_loss' + epoch_loss + '\n')
    fh.write('\n')
    fh.close()

# Function to adjust the learning rate of the optimizer
def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: ', param_group['lr'])
    return optimizer

# Function to save the current learning rate to a file
def save_lr(save_dir, optimizer):
    update_lr_group = optimizer.param_groups[0]
    fh = open(save_dir, 'a')
    fh.write('encode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('decode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('\n')
    fh.close()


# Custom loss function calculating IoU (Intersection over Union)
def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()

# Main function to start network training
def train_net(num_gpus, args):
    # Using multiprocessing to spawn training processes on each GPU
    mp.spawn(main, nprocs=num_gpus, args=(num_gpus, args))

# The main function for each process
def main(local_rank, num_gpus, args):

    cudnn.benchmark = True
    # Initialize process group for distributed training
    dist.init_process_group(backend='gloo', init_method=args.init_method, world_size=num_gpus, rank=local_rank)

    torch.cuda.set_device(local_rank)

    print("Args:",args)

    # Initialize the network model
    net = PVST(args)

    net.train()
    net.cuda()

    # Convert batch normalization to SyncBatchNorm for distributed training
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True)

    # Set different learning rates for different parts of the network
    base_params = [params for name, params in net.named_parameters() if ("backbone" in name)]
    other_params = [params for name, params in net.named_parameters() if ("backbone" not in name)]

    optimizer = optim.Adam([{'params': base_params, 'lr': args.lr * 0.1},
                            {'params': other_params, 'lr': args.lr}])
    # Load training dataset
    train_dataset = get_loader(args.trainset, args.data_root, args.img_size, mode='train')

    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=num_gpus,
        rank=local_rank,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=6,
                                               pin_memory=True,
                                               sampler=sampler,
                                               drop_last=True,
                                               )

    print('''
        Starting training:
            Train steps: {}
            Batch size: {}
            Learning rate: {}
            Training size: {}
        '''.format(args.train_steps, args.batch_size, args.lr, len(train_loader.dataset)))

    # Training loop initialization
    N_train = len(train_loader) * args.batch_size

    loss_weights = [1, 0.8, 0.8, 0.5, 0.5, 0.5]
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    criterion = nn.BCEWithLogitsLoss()
    # Training loop
    whole_iter_num = 0
    iter_num = math.ceil(len(train_loader.dataset) / args.batch_size)
    # Iterate over each training epoch
    for epoch in range(args.epochs):

        # Print the current epoch number and learning rate
        print('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))
        print('epoch:{0}-------lr:{1}'.format(epoch + 1, args.lr))

        # Initialize total loss for the epoch
        epoch_total_loss = 0
        epoch_loss = 0

        # Iterate over batches of data from the training loader
        for i, data_batch in enumerate(train_loader):
            # Break if the iteration number exceeds a predefined number
            if (i + 1) > iter_num: break
            # Unpack the data batch
            images, label_224, label_14, label_28, label_56, label_112 = data_batch
            # Move images and labels to GPU and wrap them in Variable for automatic differentiation
            images, label_224 = Variable(images.cuda(local_rank, non_blocking=True)), \
                                        Variable(label_224.cuda(local_rank, non_blocking=True))

            label_14, label_28, label_56, label_112 = Variable(label_14.cuda()), Variable(label_28.cuda()),\
                                                      Variable(label_56.cuda()), Variable(label_112.cuda())

            # Forward pass through the network
            outputs_saliency = net(images)

            # Unpack the output masks from the network
            mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency

            # Calculate loss for each scale
            loss5 = criterion(mask_1_16, label_14)
            loss4 = criterion(mask_1_8, label_28)
            loss3 = criterion(mask_1_4, label_56)
            loss1 = criterion(mask_1_1, label_224)

            # Combine losses from all scales
            img_total_loss = loss_weights[0] * loss1 + loss_weights[2] * loss3 + loss_weights[3] * loss4 + loss_weights[4] * loss5

            # Aggregate the total loss
            total_loss = img_total_loss

            epoch_total_loss += total_loss.cpu().data.item()
            epoch_loss += loss1.cpu().data.item()

            # Print current iteration number, portion of dataset processed, and losses
            print(
                'whole_iter_num: {0} --- {1:.4f} --- total_loss: {2:.6f} --- saliency loss: {3:.6f}'.format(
                    (whole_iter_num + 1),
                    (i + 1) * args.batch_size / N_train, total_loss.item(), loss1.item()))

            # Reset gradients
            optimizer.zero_grad()

            # Backward pass for gradient computation
            total_loss.backward()

            # Update model parameters
            optimizer.step()
            whole_iter_num += 1

            # Save the model state and update learning rate at specified intervals
            if (local_rank == 0) and (whole_iter_num == args.train_steps):
                torch.save(net.state_dict(), args.save_model_dir + 'PVST.pth')

            if whole_iter_num == args.train_steps:
                return 0

            if whole_iter_num == args.stepvalue1 or whole_iter_num == args.stepvalue2:
                optimizer = adjust_learning_rate(optimizer, decay_rate=args.lr_decay_gamma)
                save_dir = './loss.txt'
                save_lr(save_dir, optimizer)
                print('have updated lr!!')

        # Save the final state of the model for the epoch
        torch.save(net.state_dict(), args.save_model_dir + 'PVST_Final.pth')
        # Print and save epoch loss
        print('Epoch finished ! Loss: {}'.format(epoch_total_loss / iter_num))
        save_lossdir = './loss.txt'
        save_loss(save_lossdir, whole_iter_num, epoch_total_loss / iter_num, epoch_loss/iter_num, epoch+1)






