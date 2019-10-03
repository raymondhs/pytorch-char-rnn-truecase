'''

This file trains a character-level multi-layer RNN on text data

Code is converted to PyTorch from the Lua implementation in 
https://github.com/karpathy/char-rnn/blob/master/train.lua

'''

import argparse
import os
import time
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import CharSplitLMMinibatchLoader
from model import LSTM
from util import copy_state


parser = argparse.ArgumentParser(description='Train a character-level language model')
# data
parser.add_argument('-data_dir', default='data/tinyshakespeare', 
                    help='data directory. Should contain the file input.txt with input data')
# model params
parser.add_argument('-rnn_size', type=int, default=128, help='size of LSTM internal state')
parser.add_argument('-num_layers', type=int, default=2, help='number of layers in the LSTM')
# optimization
parser.add_argument('-learning_rate', type=float, default=2e-3, help='learning rate')
parser.add_argument('-learning_rate_decay', type=float, default=0.97, help='learning rate decay')
parser.add_argument('-learning_rate_decay_after', type=int, default=10,
                    help='in number of epochs, when to start decaying the learning rate')
parser.add_argument('-decay_rate', type=float, default=0.95, help='decay rate for rmsprop')
parser.add_argument('-dropout', type=float, default=0,
                    help='dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
parser.add_argument('-seq_length', type=int, default=50, help='number of timesteps to unroll for')
parser.add_argument('-batch_size', type=int, default=50, help='number of sequences to train on in parallel')
parser.add_argument('-max_epochs', type=int, default=50, help='number of full passes through the training data')
parser.add_argument('-grad_clip', type=float, default=5, help='clip gradients at this value')
parser.add_argument('-train_frac', type=float, default=0.95, help='fraction of data that goes into train set')
parser.add_argument('-val_frac', type=float, default=0.05, help='fraction of data that goes into validation set')
            # test_frac will be computed as (1 - train_frac - val_frac)
parser.add_argument('-init_from', default='', help='initialize network parameters from checkpoint at this path')
# bookkeeping
parser.add_argument('-seed', type=int, default=123, help='torch manual random number generator seed')
parser.add_argument('-print_every', type=int, default=1,
                    help='how many steps/minibatches between printing out the loss')
parser.add_argument('-eval_val_every', type=int, default=1000,
                    help='every how many iterations should we evaluate on validation data?')
parser.add_argument('-checkpoint_dir', default='cv', help='output directory where checkpoints get written')
parser.add_argument('-savefile', default='lstm',
                    help='filename to autosave the checkpont to. Will be inside checkpoint_dir/')
# GPU/CPU
parser.add_argument('-gpuid', type=int, default=0, help='which gpu to use. -1 = use CPU')

# parse input params
opt = parser.parse_args()
torch.manual_seed(opt.seed)

# train / val / test split for data, in fractions
test_frac = max(0, 1 - (opt.train_frac + opt.val_frac))
split_sizes = [opt.train_frac, opt.val_frac, test_frac]

# initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and torch.cuda.is_available():
    print('using CUDA on GPU {} ...'.format(opt.gpuid))
    device = torch.device('cuda', opt.gpuid)
else:
    print('Falling back on CPU mode')
    opt.gpuid = -1 # overwrite user setting
    device = torch.device("cpu")

# create the data loader class
loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes)
vocab_size = loader.vocab_size  # the number of distinct characters
vocab = loader.vocab_mapping
print('vocab size: {}'.format(vocab_size))
# make sure output directory exists
if not os.path.exists(opt.checkpoint_dir):
    os.mkdir(opt.checkpoint_dir)

# define the model: prototypes for one timestep, then clone them in time
do_random_init = True
if opt.init_from:
    print('loading an LSTM from checkpoint {}'.format(opt.init_from))
    checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    # make sure the vocabs are the same
    vocab_compatible = True
    for i,c in enumerate(checkpoint.vocab):
        if not vocab[c] == i: 
            vocab_compatible = False
    assert vocab_compatible, 'error, the character vocabulary for this dataset and the one in the saved checkpoint are not the same. This is trouble.'
    # overwrite model settings based on checkpoint to ensure compatibility
    print('overwriting rnn_size={}, num_layers={}, based on the checkpoint.'.format(checkpoint.opt.rnn_size, checkpoint.opt.num_layers))
    opt.rnn_size = checkpoint.opt.rnn_size
    opt.num_layers = checkpoint.opt.num_layers
    do_random_init = False
else:
    print('creating an lstm with {} layers '.format(opt.num_layers))
    protos = SimpleNamespace()
    protos.rnn = LSTM(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    protos.criterion = nn.NLLLoss(reduction='mean')

# the initial state of the cell/hidden states
init_state = protos.rnn.init_hidden(opt.batch_size, device)

# ship the model to the GPU if desired
if opt.gpuid >= 0:
    for v in protos.__dict__.values():
        v.to(device)

# initialization
if do_random_init:
    for weight in protos.rnn.parameters():
        nn.init.uniform_(weight, -0.08, 0.08) # small uniform numbers
# initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
# reference: https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/4
l = protos.rnn.lstm
for names in l._all_weights:
    for name in filter(lambda n: "bias_ih" in n, names):
        print('setting forget gate biases {} to 1'.format(name))
        bias = getattr(l, name)
        n = bias.size(0)
        start, end = n//4, n//2
        bias.data[start:end].fill_(1.)

num_params = sum(p.numel() for p in protos.rnn.parameters())
print('number of parameters in the model: {}'.format(num_params))

# preprocessing helper function
def prepro(x,y):
    x = x.transpose(0, 1).contiguous()
    y = y.transpose(0, 1).contiguous()
    if opt.gpuid >= 0:
        x = x.to(device)
        y = y.to(device)
    return x,y

init_state_global = copy_state(init_state)
rnn = protos.rnn
criterion = protos.criterion

def eval_split(split_index, max_batches=None):
    with torch.no_grad():
        print('evaluating loss over split index {}'.format(split_index))
        n = loader.split_sizes[split_index]
        if max_batches is not None:
            n = min(max_batches, n)

        loader.reset_batch_pointer(split_index) # move batch iteration pointer for this split to front
        loss = 0
        rnn_state = copy_state(init_state, False)
        
        for i in range(1,n+1): # iterate over batches in the split
            # fetch a batch
            x, y = loader.next_batch(split_index)
            x,y = prepro(x,y)
            # forward pass
            rnn.eval()
            predictions, rnn_state = rnn(x, rnn_state)
            loss += criterion(predictions.view(-1, vocab_size), y.view(-1))
            print('{}/{}...'.format(i,n))

        loss = loss / n
        return loss

def train():
    global init_state_global

    rnn.zero_grad()

    # get minibatch
    x,y = loader.next_batch(0)
    x,y = prepro(x,y)

    # forward pass
    rnn.train() # make sure we are in correct mode (this is cheap, sets flag)
    predictions, rnn_state = rnn(x, init_state_global)
    loss = criterion(predictions.view(-1, vocab_size), y.view(-1))
    
    # backward pass
    loss.backward()
    
    # transfer final state to initial state (BPTT)
    init_state_global = copy_state(rnn_state, False)

    # clip gradient element-wise
    nn.utils.clip_grad_value_(rnn.parameters(), opt.grad_clip)

    optimizer.step()

    return loss

# start optimization here
train_losses = []
val_losses = []
iterations = opt.max_epochs * loader.ntrain
iterations_per_epoch = loader.ntrain
loss0 = None
optimizer = optim.RMSprop(rnn.parameters(), lr=opt.learning_rate, alpha=opt.decay_rate)
milestones = [opt.learning_rate_decay_after] + list(range(opt.learning_rate_decay_after+1, opt.max_epochs+1))
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, opt.learning_rate_decay)

for i in range(1, iterations+1):
    epoch = i / loader.ntrain

    start_time = time.time()
    loss = train_loss = train()
    train_losses.append(train_loss)
    elapsed = time.time()-start_time

    # exponential learning rate decay
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1:
        scheduler.step()
        if epoch >= opt.learning_rate_decay_after:
            decay_factor = opt.learning_rate_decay
            lr = optimizer.param_groups[0]['lr']
            print('decayed learning rate by a factor {} to {}'.format(decay_factor, lr))

    # every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations:
        # evaluate loss on validation data
        val_loss = eval_split(1) # 1 = validation
        val_losses.append(val_loss)

        savefile = '{}/lm_{}_epoch{:.2f}_{:.4f}.pt'.format(opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to {}'.format(savefile))
        checkpoint = SimpleNamespace()
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(checkpoint, savefile)

    if i % opt.print_every == 0:
        print("{}/{} (epoch {:.3f}), train_loss = {:6.8f}, time/batch = {:.4f}s".format(i, iterations, epoch, train_loss, elapsed))
    
    # handle early stopping if things are going really bad
    if loss != loss:
        print('loss is NaN, aborting.')
        break # halt

    if loss0 is None: loss0 = loss
    if loss > loss0 * 3:
        print('loss is exploding, aborting.')
        break # halt
