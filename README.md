# pytorch-char-rnn-truecase

This code is a PyTorch implementation of a character-level RNN for truecasing. The original code is based on Torch7 [char-rnn-truecase](https://github.com/raymondhs/char-rnn-truecase/), which is in turn based on [char-rnn](https://github.com/karpathy/char-rnn).

## Running the code

* PyTorch version >= 1.1.0
* Python version >= 3.5

Scripts for training and evaluation are provided. They can be run as follows:

```
bash (train|test).sh <corpus> (lstm|gru) (small|large) <gpuid>
```

`<corpus>` is one of the following: `wiki`, `wsj`, `conll_eng`, `conll_deu`. Data splits should be put in `data/<corpus>` and named: `input.txt` (train), `val_input.txt` (tune), `test_lower.txt` (test, lowercased). Only Wikipedia data is uploaded (same splits as [William Coster and David Kauchak (2011)](http://www.cs.pomona.edu/~dkauchak/simplification/data.v1/data.v1.split.tar.gz)). The remaining datasets should be obtained from their respective sources in LDC.  For example, issue the following commands to train and evaluate LSTM-Large on Wikipedia using GPU0:

```
bash train.sh wiki lstm large 0
bash test.sh wiki lstm large 0
```

The truecased file can be found in `cv/wiki_lstm_700hidden_3layer/output.txt`. For evaluation, use `word_eval.py` to compute accuracy and F1-score:

```
python word_eval.py <gold file> <output file>
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