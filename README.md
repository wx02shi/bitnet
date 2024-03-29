# mistral-bitnet

Implementing the BitNet b1.58 model, using the [`mistral-src`](https://github.com/mistralai/mistral-src/tree/main) repository as a base.

Currently, there is no implementation for `gemm_lowbit_kernel`. I can't figure it out. As a result, during inference mode, the weights and activations are still type `float16`, because PyTorch does not support matrix multiplication for `int8`.  
In theory, this implementation should produce the same outputs as a correct BitNet, but without the efficiency gains.

> NOTE: For convenience, there should be two model definitions: one with the training mode of BitLinear, and one with the inference mode. Generation scripts should load the inference-mode model.  
> Training scripts should load the training-mode model, and then after training, replace the training BitLinears with inference BitLinears.  
> The current implementation just has an `if self.training`. But having two different definitions will save on space and latency during inference

## Motivation
The original whitepaper [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/pdf/2402.17764.pdf) used "LLaMA-alike components". From a naive perspective, there are few differences between the architectures of LLaMA and Mistral (especially with respect to BitNet quantization). 

Credits to @nkotak, who quickly put out the (imo) best implementation of the whitepaper. I would not have been able to create this project without reading it over and having some chats about working with it, they have helped me to digest the contents much faster!  
Check out [their repo](https://github.com/nkotak/1.58BitNet), especially if you want to run a BitNet on Apple silicon. 

I chose to build a BitNet model based on the [`mistral-src`](https://github.com/mistralai/mistral-src/tree/main) repository. While I admire nkotak's work, my intuition says BitNet is still not good enough for ARM processors. GPU is still favourable, both in terms of raw computational power as well as existing optimizations. E.g. `xformers`, sliding window attention, and possibly Paged Attention or Flash Attention.  
Secondly, I wanted to explore how BitNet quantization interacts with MoE.  

## Overview
(TODO)