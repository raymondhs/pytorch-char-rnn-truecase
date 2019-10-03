import gc
import math
import os
import sys

import torch


class CharSplitLMMinibatchLoader(object):
    ''' Line-by-line conversion from CharSplitLMMinibatchLoader.lua '''

    @classmethod
    def create(self, data_dir, batch_size, seq_length, split_fractions):
        # split_fractions is e.g. {0.9, 0.05, 0.05}

        input_file = os.path.join(data_dir, 'input.txt')
        vocab_file = os.path.join(data_dir, 'vocab.pt')
        tensor_file = os.path.join(data_dir, 'data.pt')

        # rhs: validation file
        self.has_val_data = True
        val_data = True
        
        val_input_file = os.path.join(data_dir, 'val_input.txt')
        val_tensor_file = os.path.join(data_dir, 'val_data.pt')
        if not (os.path.exists(val_input_file)):
            self.has_val_data = False
        
        # fetch file attributes to determine if we need to rerun preprocessing
        run_prepro = False
        if not (os.path.exists(vocab_file) or os.path.exists(tensor_file)):
            # prepro files do not exist, generate them
            print('vocab.pt and data.pt do not exist. Running preprocessing...')
            run_prepro = True
        else:
            # check if the input file was modified since last time we 
            # ran the prepro. if so, we have to rerun the preprocessing
            input_attr = os.stat(input_file)
            vocab_attr = os.stat(vocab_file)
            tensor_attr = os.stat(tensor_file)
            if input_attr.st_mtime > vocab_attr.st_mtime or input_attr.st_mtime > tensor_attr.st_mtime:
                print('vocab.pt or data.pt detected as stale. Re-running preprocessing...')
                run_prepro = True
        if run_prepro:
            #construct a tensor with all the data, and vocab file
            print('one-time setup: preprocessing input text file {} ...'.format(input_file))
            CharSplitLMMinibatchLoader.text_to_tensor(input_file, vocab_file, tensor_file)
            # rhs: tensor with val data
            if self.has_val_data:
                CharSplitLMMinibatchLoader.text_to_tensor(val_input_file, vocab_file, val_tensor_file)

        print('loading data files...')
        data = torch.load(tensor_file)
        self.vocab_mapping = torch.load(vocab_file)

        # cut off the end so that it divides evenly
        length = data.size(0)
        if length % (batch_size * seq_length) != 0:
            print('cutting off end of data so that the batches/sequences divide evenly')
            data = data[:batch_size * seq_length 
                        * math.floor(length / (batch_size * seq_length))]

        # load val data
        if self.has_val_data:
            val_data = torch.load(val_tensor_file)
            length = val_data.size(0)
            if length % (batch_size * seq_length) != 0:
                print('cutting off end of data so that the batches/sequences divide evenly')
                val_data = val_data[:batch_size * seq_length 
                            * math.floor(length / (batch_size * seq_length))]
        
        # count vocab
        self.vocab_size = len(self.vocab_mapping)

        # self.batches is a table of tensors
        print('reshaping tensor...')
        self.batch_size = batch_size
        self.seq_length = seq_length

        ydata = data.clone()
        ydata[:-1] = data[1:]
        ydata[-1] = data[0]
        self.x_batches = data.view(batch_size, -1).split(seq_length, 1)  # #rows = #batches
        self.nbatches = len(self.x_batches)
        self.y_batches = ydata.view(batch_size, -1).split(seq_length, 1)  # #rows = #batches
        assert len(self.x_batches) == len(self.y_batches)

        # same thing for val data
        if self.has_val_data:
            val_ydata = val_data.clone()
            val_ydata[:-1] = val_data[1:]
            val_ydata[-1] = val_data[0]
            self.val_x_batches = val_data.view(batch_size, -1).split(seq_length, 1)  # #rows = #batches
            self.val_nbatches = len(self.val_x_batches)
            self.val_y_batches = val_ydata.view(batch_size, -1).split(seq_length, 1)  # #rows = #batches
            assert len(self.val_x_batches) == len(self.val_y_batches)

        # lets try to be helpful here
        if self.nbatches < 50:
            print('WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')

        # perform safety checks on split_fractions
        assert split_fractions[0] >= 0 and split_fractions[0] <= 1, 'bad split fraction {} for train, not between 0 and 1'.format(split_fractions[0])
        assert split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction {} for val, not between 0 and 1'.format(split_fractions[1])
        assert split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction {} for test, not between 0 and 1'.format(split_fractions[2])

        if self.has_val_data:
            self.ntrain = self.nbatches
            self.nval = self.val_nbatches
            self.ntest = 0
        else:
            if split_fractions[2] == 0: 
                # catch a common special case where the user might not want a test set
                self.ntrain = math.floor(self.nbatches * split_fractions[0])
                self.nval = self.nbatches - self.ntrain
                self.ntest = 0
            else:
                # divide data to train/val and allocate rest to test
                self.ntrain = math.floor(self.nbatches * split_fractions[1])
                self.nval = math.floor(self.nbatches * split_fractions[2])
                self.ntest = self.nbatches - self.nval - self.ntrain # the rest goes to test (to ensure this adds up exactly)

        self.split_sizes = [self.ntrain, self.nval, self.ntest]
        self.batch_ix = [-1,-1,-1]

        print('data load done. Number of data batches in train: {}, val: {}, test: {}'.format(self.ntrain, self.nval, self.ntest))
        gc.collect()
        return self()


    def reset_batch_pointer(self, split_index, batch_index=-1):
        self.batch_ix[split_index] = batch_index


    def next_batch(self, split_index):
        if self.split_sizes[split_index] == 0:
            # perform a check here to make sure the user isn't screwing something up
            split_names = ['train', 'val', 'test']
            print('ERROR. Code requested a batch for split {}, but this split has no data.'.format(split_names[split_index]))
            sys.exit() # crash violently
        # split_index is integer: 0 = train, 1 = val, 2 = test
        self.batch_ix[split_index] = self.batch_ix[split_index] + 1
        if self.batch_ix[split_index] >= self.split_sizes[split_index]:
            self.batch_ix[split_index] = 0 # cycle around to beginning
        # pull out the correct next batch
        ix = self.batch_ix[split_index]
        if not (self.has_val_data):
            if split_index == 1: ix = ix + self.ntrain # offset by train set size
            if split_index == 2: ix = ix + self.ntrain + self.nval # offset by train + val
            return self.x_batches[ix], self.y_batches[ix]
        else:
            if split_index == 0:
                return self.x_batches[ix], self.y_batches[ix]
            else:
                return self.val_x_batches[ix], self.val_y_batches[ix]


    @staticmethod
    def text_to_tensor(in_textfile, out_vocabfile, out_tensorfile):

        print('loading text file...{}'.format(in_textfile))
        cache_len = 10000
        with open(in_textfile, encoding='utf-8') as f:
            rawdata = f.read()
        length = len(rawdata)
        vocab_mapping = {}

        # create vocabulary if it doesn't exist yet
        if os.path.exists(out_vocabfile):
            print('vocab.pt found')
            vocab_mapping = torch.load(out_vocabfile)
        else:
            print('creating vocabulary mapping...')
            # record all characters to a set
            unordered = set(rawdata)
            unordered.add("<unk>")
            # sort into a table (i.e. keys become 1..N)
            ordered = dict(enumerate(unordered))
            # invert `ordered` to create the char->int mapping
            vocab_mapping = {char: i for i, char in ordered.items()}

            # save vocab
            print('saving {}'.format(out_vocabfile))
            torch.save(vocab_mapping, out_vocabfile)

        # construct a tensor with all the data
        print('putting data into tensor...')
        data = torch.LongTensor(length) # store it into 1D first, then rearrange
        for pos, char in enumerate(rawdata):
            if char in vocab_mapping:
                data[pos] = vocab_mapping[char]
            else:
                data[pos] = vocab_mapping["<unk>"]

        # save output preprocessed files
        print('saving {}'.format(out_tensorfile))
        torch.save(data, out_tensorfile)
