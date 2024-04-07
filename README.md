# mistral-bitnet

This is an exploration, where I attempted to implement the BitNet b1.58 model, using the [`mistral-src`](https://github.com/mistralai/mistral-src/tree/main) repository as a base.

The [whitepaper](https://arxiv.org/pdf/2402.17764.pdf) illustrates some promising results. Inference is much lighter, with negligible degradation to accuracy, and significantly outperforms popular post-quantization techniques.  

Interestingly, they did not provide any figures for training loss with respect to training time, but they do provide figures for scaling law of loss versus model size. 

Due to my limited attention span (and exams), I did not implement the following:
- NVIDIA FasterTransformer (because I built on Mistral first)
- Ladder 2-bit kernel (I could not find the referenced whitepaper anyways)
- Pre-trained model, or a pre-training script (GPU-poor)

I could spend some additional time to implement a 2-bit kernel. PyTorch (or perhaps CUDA) does not support `int8` matrix multiplication, and my intuition tells me that I won't be implementing an efficient kernel by my own skill. Without an efficient kernel, BitNet is pretty much pointless. 

but in reality this project cannot progress much further. One setback of bitnet is that it changes the architecture of the attention layer. This means that we cannot use weights from existing models, and so a useful bitnet model must be pre-trained. I do not have the hardware, time, or money to do this.