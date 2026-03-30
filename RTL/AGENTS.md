# Project Summary
In this project, you will implement a ternary LLM Bitnet on a small FPGA board. I have little experience with FPGAs before, so you should colloborate with me to complete the project while teaching me how to program the FPGA and optimize the performance, instead of just completing the codes.

## Bitnet Model Information
1. Model config and weights are stored in `../bitnet_700M/`. Bitnet_700M is a model with 700M parameters, but each parameter can only take a ternary value from -1, 0 or 1. And for each weight matrix, there is a floating point scale.
2. Bitnet is 700M Transformer model supporting a context window of 2048.

## FPGA Information
1. We will implement Bitnet on a AX7035B FPGA board with XC7A35T-2FGG484I FPGA and 256MB DDR3. The FPGA has 90 DSPs, 33280 logic cells, 5200 slices, 41600 flip-flops, and 1800kb BRAM.
2. For other information about the hardware, you can search online.

## Additional Info
1. Please use RTL verilog to implement the project. Do not use HLS for now because I need to get familiar with RTL first.
2. Vivado is installed on this computer. You can run relevant command in terminal.
3. Please do the project in a top-down approach, i.e. let us figure out the top modules first, then define the interface for all the modules, then implement and test each module with unit test. When we finish designing a module, we should always test it to make sure it works as expected.
4. During the process, we should discuss the design choices for each module, and finalize our design into comments and documents for better maintainence.
5. The file structure, module interfaces and other format specifications are stored under the `docs/` folder. You should referance them when doing the project.
6. The latest progress of the project is stored in `docs/progress.md`. You should read it when resuming the work.
