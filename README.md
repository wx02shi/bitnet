# mistral-bitnet

Implementing the BitNet b1.58 model, using the [`mistral-src`](https://github.com/mistralai/mistral-src/tree/main) repository as a base.


> NOTE: I'm still trying to figure out if this is true...  
> The whitepaper recommends quantizing offline to `int8`. Not mentioned anywhere (maybe it's obvious to experienced users) is that PyTorch cannot do matrix multiplication between different types. The whitepaper implies that the custom CUDA `gemm_lowbit_kernel` exists for optimization, but in reality, it is a requirement for BitNet techniques to be viable. If we continued to use `float16` as the type, then we will have saved no space (in fact, it will use more), while losing precision.

## Motivation
The original whitepaper [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/pdf/2402.17764.pdf) used "LLaMA-alike components". From a naive perspective, there are few differences between the architectures of LLaMA and Mistral (especially with respect to BitNet quantization). 

Credits to @nkotak, who quickly put out the (imo) best implementation of the whitepaper. I would not have been able to create this project without reading it over and having some chats about working with it, they have helped me to digest the contents much faster!  
Check out [their repo](https://github.com/nkotak/1.58BitNet), especially if you want to run a BitNet on Apple silicon. 

I chose to build a BitNet model based on the [`mistral-src`](https://github.com/mistralai/mistral-src/tree/main) repository. While I admire nkotak's work, my intuition says BitNet is still not good enough for ARM processors. GPU is still favourable, both in terms of raw computational power as well as existing optimizations. E.g. `xformers`, sliding window attention, and possibly Paged Attention or Flash Attention.  
Secondly, I wanted to explore how BitNet quantization interacts with MoE.  

## Overview
(TODO)