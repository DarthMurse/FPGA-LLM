# FPGA-LLM
In this project, I will try to deploy an LLM with 1.5B parameters on an AX7010 FPGA board. The AX7010 board contains Xilinx Zynq7000 SoC chip, which contains two ARM Cortex A-9 cores and an XC7Z010-1CLG400C FPGA. The board has 512MB DDR3 SDRAM, 2.1MB Block RAM, 80 18 * 25 MACCs, and several thousands LUTs, logic cells and flip-flops. As you can see, our compute and memory resources are quite limited, so we need to perform some radical compression to the LLM, meanwhile retaining the performance of the model. 

To this end, [Bitnet 1.58b](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) is a very good choice. Due to its ternary weights, the model size is only 0.4GB, which can fit into our DRAM together with the word embedding. The activation is quantized to 8 bits, with a context length of 4096, which means supporting the full context window would require `2560 / 20 * 5 * 2 * 30 * 4096 = 157MB` spaces for the kv cache, which might be too long for our hardare. If that's the case, we will just cut the context length by half.

## BitNet Components
1. Word embedding
2. RMSNorm
3. Ternary linear layers
4. Residual
5. RoPE
6. Attention
7. SwiGLU
8. Output layer 
We will work on these components one by one, and fuse them to reduce memory access if possible.
