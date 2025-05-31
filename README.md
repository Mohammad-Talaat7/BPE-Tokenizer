<div align="center">
    <h1>Byte-Pair Encoding Tokenizer</h1>
    <h3>From Scratch<h3>
    <h3></h3>
</div>

<div align="center">

![Last-Commit](https://img.shields.io/github/last-commit/mohammad-talaat7/bpe-tokenizer?label=last-commit&logo=github)
![Repo-Size](https://img.shields.io/github/repo-size/mohammad-talaat7/bpe-tokenizer?label=repo-size&logo=googledrive)
![Tests](https://img.shields.io/github/actions/workflow/status/mohammad-talaat7/bpe-tokenizer/tests.yml?label=tests&logo=github)

</div>

A simple implementation for the Byte-Pair Tokenizer based on Stanford CS336 Course, Philip Gage 1994, Sennrich+ 2015, and Radford+ 2019

The implementation are divided into two files in the `src` directory:

- bpe_tokenizer.py:
  - In this file exist the actual tokenizer in the `tokenizer` class that take the vocab, merge list, and the special tokens list (if applicable) and initialize a tokenizer
  - In this file also there is a simple test function `test_tokenizer` to verify the tokenizer functionality which will run if you run this file directly (`python3 bpe_tokenizer.py`)
- training_tokenizer.py:
  - In this file exist couple of functions needed to generate the vocab, and the merge list we call this process "training the BPE Tokenizer"
  - There is also functions for saving and loading checkpoints for stopping and resume training
  - When running this file directly ('python3 training_tokenizer.py') you can pass arguments needed for training through the `argparser` and it not passing any arguments it will train using the default parameters (you can inspect the arguments and default parameters using -h)
