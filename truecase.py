'''

This file truecases an input text using a trained model

Code is based on implementation in
https://github.com/raymondhs/char-rnn-truecase/blob/master/sample.lua

Note: The beam search is re-written for better readability.

'''

import argparse
import os
import sys

from tqdm import tqdm

import torch


parser = argparse.ArgumentParser(description='Sample from a character-level language model')
# required:
parser.add_argument('model', type=str, help='model checkpoint to use for sampling')
# optional parameters
parser.add_argument('-seed', type=int, default=123, help='random number generator\'s seed')
parser.add_argument('-gpuid', type=int, default=0, help='which gpu to use. -1 = use CPU')
parser.add_argument('-verbose', type=int, default=1, help='set to 0 to ONLY print the sampled text, no diagnostics')
parser.add_argument('-beamsize', type=int, default=1, help='defaults to 1')
parser.add_argument('-sent', action='store_true', default=False, help='perform sentence-level truecasing')

opt = parser.parse_args()

def gprint(s):
    s = s.replace('\n', '<n>')
    if opt.verbose == 1:
        print(s, file=sys.stderr)

# initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and torch.cuda.is_available():
    gprint('using CUDA on GPU {} ...'.format(opt.gpuid))
    device = torch.device('cuda', opt.gpuid)
else:
    gprint('Falling back on CPU mode')
    opt.gpuid = -1 # overwrite user setting
    device = torch.device("cpu")

torch.manual_seed(opt.seed)

# load the model checkpoint
if not os.path.exists(opt.model):
    gprint("Error: File {} does not exist. Are you sure you didn't forget to prepend cv/ ?".format(opt.model))
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn.to(device)
protos.rnn.eval() # put in eval mode so that dropout works properly

# initialize the vocabulary (and its inverted version)
vocab = checkpoint.vocab

#initialize the rnn state to all zeros
gprint('creating an lstm...')
current_state = checkpoint.protos.rnn.init_hidden(1, device)

def beam_search_decoder(s, k, progress_bar=True):
    with torch.no_grad():
        h, c = current_state
        h, c = h.clone().squeeze(), c.clone().squeeze()
        sequences = [[s[0], (h, c), 0.]]

        # iterate over each character
        progress = range(1, len(s))
        if progress_bar:
            progress = tqdm(progress, desc="num characters")
        for t in progress:
            all_candidates = []

            # treat the candidates in the beam as one batch
            inputs = []
            hs = []
            cs = []
            for i in range(len(sequences)):
                seq, (h, c), _ = sequences[i]
                inputs.append(vocab.get(seq[-1], vocab["<unk>"]))
                hs.append(h)
                cs.append(c)
            
            # input shape: 1 x beam_size
            inputs = torch.tensor(inputs, device=device).unsqueeze(0)
            # hidden shape: num_layers x beam_size x H
            hs = torch.stack(hs).transpose(0, 1).contiguous()
            cs = torch.stack(cs).transpose(0, 1).contiguous()
            
            # forward the candidates in beam
            next_scores, (next_hs, next_cs) = protos.rnn(inputs, (hs, cs))

            # expand each current candidate
            for i in range(len(sequences)):
                seq, _, score = sequences[i]
                next_chars = [s[t]]
                if s[t].upper() != s[t]:
                    next_chars.append(s[t].upper())
                for c in next_chars:
                    next_score = next_scores[0, i, vocab.get(c, vocab["<unk>"])]
                    # same hidden/cell state is re-used for next char, do clone here to be safe
                    next_h, next_c = next_hs[:, i].clone(), next_cs[:, i].clone()
                    candidate = [seq + c, (next_h, next_c), score + next_score]
                    all_candidates.append(candidate)
            
            # order all candidates by highest score
            ordered = sorted(all_candidates, key=lambda tup:-tup[2])
            # select k best
            sequences = ordered[:k]
            
        decoded = sequences[0][0].strip()
        return decoded

if opt.sent:
    gprint('performing sentence-level truecasing...')
    gprint('memory will be reset after every line')
    lines = sys.stdin.readlines()
    for line in tqdm(lines, desc="num lines"):
        # truecase line by line
        print(beam_search_decoder('\n' + line.rstrip(), opt.beamsize), progress_bar=False)
else:
    gprint('performing document-level truecasing...')
    gprint('memory is carried over to the next line')
    lines = sys.stdin.read()
    # truecase whole text at once
    print(beam_search_decoder('\n' + lines.rstrip(), opt.beamsize))
