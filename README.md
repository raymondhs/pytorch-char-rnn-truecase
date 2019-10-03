# pytorch-char-rnn-truecase

This code is a PyTorch implementation of a character-level LSTM for truecasing. The original code is based on Torch7 [char-rnn-truecase](https://github.com/raymondhs/char-rnn-truecase/), which is in turn based on [char-rnn](https://github.com/karpathy/char-rnn).

## Running the code

* PyTorch version >= 1.1.0
* Python version >= 3.5

Below example shows training and truecasing on the Wikipedia data set. Refer to `train.sh` and `test.sh` for running the experiments in the paper (only LSTM is supported).

### Training

```
python train.py \
  -data_dir data/wiki \
  -rnn_size 700 \
  -num_layers 3 \
  -dropout 0.25 \
  -batch_size 100 \
  -seq_length 50 \
  -max_epochs 30 \
  -learning_rate 0.001 \
  -checkpoint_dir cv/wiki_lstm_700hidden_3layer \
  -gpuid 0
```

### Truecasing

```
# retrieve best checkpoint on valid data
model=`ls cv/wiki_lstm_700hidden_3layer/*.pt | python best_model.py`

cat data/wiki/test.lower.txt \
| python truecase.py \
    $model \
    -beamsize 10 \
    -verbose 0 \
    -gpuid 0 \
> data/wiki/output.txt

# calculate performance on test set
python word_eval.py data/wiki/test.txt cv/wiki_lstm_700hidden_3layer/output.txt
```

On my machine, this gives the following scores:

```
Accuracy: 97.55
Precision: 94.23
Recall: 92.78
F1: 93.50
```

## References

```
@inproceedings{susanto-etal-2016-learning,
  title = "Learning to Capitalize with Character-Level Recurrent Neural Networks: An Empirical Study",
  author = "Susanto, Raymond Hendy and Chieu, Hai Leong and Lu, Wei",
  booktitle = "Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing",
  year = "2016",
}
```
