# AI-ML

Learning projects built from scratch, following Andrej Karpathy's [micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) and ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) videos.

## micrograd.ipynb

A tiny scalar-valued autograd engine (`Value` class with backprop), checked against PyTorch, and a small MLP (3 → 4 → 4 → 1) built on top of it.

**Results:** training the MLP for 20 steps on a 4-example toy dataset drops the squared-error loss from **6.04 to 0.015**. Its predictions end up close to the targets `[1, -1, -1, 1]`:

```
[0.946, -0.988, -0.919, 0.927]
```

## GPT_Replica.ipynb

The first part of a character-level GPT on the Tiny Shakespeare dataset (`input.txt`, 1.1M characters, 65-character vocabulary):

- character tokenizer (encode/decode) and a 90/10 train/validation split
- batching with block size (context length)
- a bigram language model trained with AdamW
- self-attention built up step by step (averaging loop → matrix multiply → softmax mask → a single attention head), plus LayerNorm

**Results (bigram model):** training loss goes from **4.88** (random init, about −ln(1/65)) to **2.38** after 10,000 steps. A sample of the generated text:

```
lso br. ave aviasurf my, yxMPZI ivee iuedrd whar ksth y h bora s be hese, woweee; the! KI 'de, ulsee
```

A bigram model only looks at the previous character, which is why its output isn't real Shakespeare yet. The next step is to stack the attention heads into full Transformer blocks.
