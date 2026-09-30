# GPT from scratch (PyTorch)

I've been learning AI for the past few months, and I wanted to properly understand how transformers work instead of just using them through libraries. So I built a small GPT from scratch in PyTorch and trained it on the Tiny Shakespeare dataset.

I followed Andrej Karpathy's "Let's build GPT" video and used the Shakespeare dataset from his char-rnn repo. I wrote the notebook cell by cell and printed tensor shapes after every step, because that was the best way for me to actually understand the reshaping and attention math.

I know there are many problems in this and it's not an optimized model. That was never the goal. This project was about learning how the pieces fit together.

## What's in the notebook

- Loading Tiny Shakespeare and splitting it into train/val
- Tokenizing with the pretrained GPT-2 tokenizer (Hugging Face)
- Batching: `x` is a chunk of tokens, `y` is the same chunk shifted by one
- Token embeddings + learned position embeddings
- Single-head causal attention, then multi-head
- Feed-forward layer, residual connections, LayerNorm
- Stacked transformer blocks with a final LayerNorm and output head
- Training loop with AdamW
- Text generation

## Setup

- 4 layers, 4 heads, embedding dim 128
- Context length 64, batch size 16
- GPT-2 vocab (50257), about 14M parameters
- AdamW, lr 3e-4, 1000 steps on a free Colab T4

## How to run

Open the notebook in Colab, or locally:

```bash
pip install torch transformers
jupyter notebook
```

The notebook downloads the dataset by itself.

## Results

Loss went from about 10.9 to about 4.8 over 1000 steps. The model picks up the script format and Shakespeare-style phrasing. It's undertrained, so the output is repetitive. Prompted with `ROMEO:`:

```
ROMEO:
I'll be a man, and the king,
And I'll be a man, and the king,
And I'll be the king, and the king.
```

## Next

I'm planning a second project: training a ~40M parameter model from scratch on a bigger dataset.

## Credits

- Andrej Karpathy's "Let's build GPT" video
- GPT-2 tokenizer via Hugging Face `transformers`
