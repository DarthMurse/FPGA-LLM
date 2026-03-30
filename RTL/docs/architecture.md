# BitNet FPGA RTL Architecture Spec

## 1. Goal

This project implements a BitNet language model on the AX7035B FPGA board using RTL Verilog.

The architecture explicitly treats prompt prefill and autoregressive decode as different execution modes.

The project follows a top-down flow:

1. Define the system target and top modules.
2. Define interfaces and memory layout.
3. Build a software golden model and data-packing tools.
4. Implement RTL modules one by one with unit tests.
5. Integrate modules into end-to-end prefill and decode pipelines.
6. Optimize throughput, memory traffic, and timing after correctness is established.

This document is the baseline architecture for the first implementation.

## 2. Design Principles

The design is constrained more by memory bandwidth and on-chip storage than by raw arithmetic.

We therefore adopt the following principles:

- Keep tokenizer, token embedding, final LM head, softmax, and sampling on the host machine.
- Use one reusable transformer layer engine, not 24 physical layer instances.
- Stream weights from DDR3 in bursts.
- Keep only small working sets on chip in BRAM or distributed RAM.
- Start with batch size 1.
- Treat prefill and decode as separate schedules built on shared compute blocks.
- Keep the first implementation simple and debuggable before optimizing.
- Define clear interfaces so each module can be tested independently.

## 3. First Implementation Target

The initial target is intentionally limited but no longer collapses prefill into decode.

- Model: BitNet 700M from `../bitnet_700M/`
- Hidden size: 1536
- Intermediate size: 4096
- Layers: 24
- Attention heads: 16
- KV heads: 16
- Vocab size: 32002
- Max trained context: 2048

Initial runtime assumptions:

- Batch size = 1
- Separate prefill path and decode path
- Tokenizer runs on the host
- Token embeddings are produced on the host and sent to the FPGA
- Model weights and metadata are initialized from SD card or QSPI into DDR3
- Final RMSNorm output is sent back to the host
- LM head projection, softmax, and sampling run on the host
- Greedy decoding first
- Reduced supported context during bring-up, target `512` or `1024`
- Weights stored in external DDR3
- KV cache stored in external DDR3
- Activations use symmetric per-token `int8` quantization with `fp16` scale
- KV cache supports button-triggered clear

Non-goals for phase 1:

- Advanced sampling
- Multi-batch scheduling
- Multi-sequence serving
- Aggressive operator fusion before correctness is proven

Mode definitions:

- Prefill mode processes a prompt sequence and populates the KV cache for all prompt positions.
- Decode mode processes one new token at a time using the existing KV cache.

Why we split them:

- Prefill has much more parallel work across sequence positions.
- Decode is dominated by matrix-vector style execution and sequential cache growth.
- The optimal memory schedule and buffering policy differ enough that one unified controller would hide important design choices.

Host/FPGA partition:

- Host is responsible for tokenization and detokenization.
- Host is responsible for embedding lookup.
- FPGA is responsible for transformer-layer execution and KV-cache maintenance.
- Host is responsible for final LM head projection, softmax, and token selection.

Why this partition is reasonable:

- The LM head is very large because vocab size is 32002.
- Keeping LM head on the host avoids streaming a very large output projection matrix through the FPGA for every decode step.
- Tokenization and sampling are control-heavy and easy to implement on the host.
- The FPGA can focus on the part of the workload where BitNet ternary arithmetic is most useful.

Communication interface choice:

- Use Gigabit Ethernet as the primary host-to-FPGA runtime interface.
- On this board, Gigabit Ethernet is the highest-bandwidth general host communication interface exposed for application data flow.
- USB 2.0 is slower in peak throughput than Gigabit Ethernet.
- UART is useful for debug only.

Note:

- This statement is based on the AX7035B board documentation describing one Gigabit Ethernet interface and one USB 2.0 interface. The practical throughput will depend on the MAC design, packetization overhead, and host software stack.

## 4. Top-Level System View

At a high level, the system has two execution paths.

Prefill path:

1. Host tokenizes the prompt and sends a tile of token embeddings to the FPGA.
2. Process the tile through all 24 transformer layers using the prefill schedule.
3. Write K and V results for all processed positions into the KV cache.
4. Produce the final hidden state for the last prompt token.
5. Return the final hidden state to the host.
6. Host runs final RMSNorm, LM head, and token selection.

Decode path:

1. Host sends the current token embedding.
2. Process the hidden state through all 24 transformer layers using the decode schedule.
3. Return the final hidden state to the host.
4. Host runs final RMSNorm, LM head, softmax, and sampling.
5. Host sends the next token embedding.
6. Repeat for the next decode step.

The top-level hardware blocks are:

- `top_system`
- `host_if`
- `boot_if`
- `prefill_controller`
- `decode_controller`
- `ddr_if`
- `onchip_buffer_bank`
- `prefill_path`
- `decode_path`
- `layer_engine`
- `kv_cache_manager`

The corresponding host-side software blocks are:

- `tokenizer`
- `embedding_lookup`
- `lm_head`
- `softmax_sampler`
- `ethernet_runtime`

## 5. System Graph

The following graph shows the intended first-pass partition.

```text
 +------------------------------- Host Machine --------------------------------+
 |                                                                              |
 |  tokenizer -> token ids -> embedding_lookup -> embedding vectors             |
 |                                                    |                         |
 |                                                    v                         |
 |                                             ethernet_runtime                 |
 |                                                    |                         |
 |  hidden state <- ethernet_runtime <- final hidden state from FPGA            |
 |         |                                                                    |
 |         v                                                                    |
 |   final_rmsnorm -> lm_head -> softmax / argmax / sampling -> next token      |
 |                                                                              |
 +-----------------------------------+------------------------------------------+
                                     |
                                     | Gigabit Ethernet
                                     v
 +------------------------------- AX7035B FPGA ---------------------------------+
 |                                                                              |
 |  +---------+     +--------------------+     +-----------------------------+   |
 |  | host_if |<--->| prefill_controller |---->|                             |   |
 |  |         |     +--------------------+     |                             |   |
 |  |         |<--->| decode_controller  |---->|         layer_engine        |   |
 |  +----+----+     +--------------------+     |                             |   |
 |       |                                        rmsnorm / linear / rope /  |   |
 |       |     +--------------------+             attention / swiglu / add   |   |
 |       +---->|  boot_if (SD/QSPI) |----------------------------------------+   |
 |             +--------------------+                                             |
 |                                             |  rmsnorm / linear / rope /  |   |
 |                                             |  attention / swiglu / add   |   |
 |                                             +-------------+---------------+   |
 |                                                           |                   |
 |                                     +---------------------+----------------+  |
 |                                     |                                      |  |
 |                                     v                                      v  |
 |                           +-------------------+                  +-------------+
 |                           | onchip_buffer_bank|                  |ddr_if       |
 |                           |                   |                  |             |
 |                           | BRAM / LUTRAM     |                  | DDR3 ctrl   |
 |                           +---------+---------+                  +------+------+ 
 |                                     |                                      |
 |                                     |                                      |
 |                 button -> kv_clear_req / status                   External DDR3
 |                         CLB fabric / control paths                 weights + KV
 |                         DSPs used only where arithmetic            weights + KV
 |                         really benefits from them                  cache + metadata
 |                                                                              |
 +------------------------------------------------------------------------------+
```

## 5. Top Modules

### 5.1 `top_system`

Responsibility:

- Board-level top module.
- Instantiates clocks, reset, external interfaces, controller, DDR interface, and compute modules.
- Connects FPGA-side control and data paths.

Notes:

- Keep board-specific wiring here.
- Keep model logic out of this module.

### 5.2 `host_if`

Responsibility:

- Provides a simple control path between host software and FPGA.
- Sends commands such as reset, initialize model metadata, load prompt embeddings, run prefill, run one decode step, and read results.
- Sends embedding vectors to the FPGA and receives final hidden states back.
- Does not stream model weights during inference.

First-pass recommendation:

- Implement this module around Gigabit Ethernet.
- Keep the protocol simple and software-friendly before optimizing.

First-pass payloads:

- control packets
- embedding vectors from host to FPGA
- final hidden-state vectors from FPGA to host
- status and debug messages

### 5.3 `boot_if`

Responsibility:

- Initializes DDR from SD card or QSPI before inference begins.
- Loads model weights, scales, and metadata into the DDR layout defined by `memory_layout.md`.
- Reports initialization success or failure to the rest of the system.

### 5.4 `prefill_controller`

Responsibility:

- Main finite-state machine or microcoded scheduler.
- Sequences all operations for prompt ingestion.
- Tracks current layer index, sequence tile, sequence position, and buffer ownership.

Expected sequence:

1. Receive one prompt embedding tile from `host_if`.
2. For each layer:
   - pre-attention norm
   - Q/K/V projections
   - RoPE
   - KV cache write for all positions in the tile
   - prefill attention score/value path
   - attention output projection
   - residual add
   - pre-MLP norm
   - gate/up projections
   - SwiGLU
   - down projection
   - residual add
3. Final norm
4. Return the final hidden state for the last prompt token to `host_if`

### 5.5 `decode_controller`

Responsibility:

- Main finite-state machine or microcoded scheduler for autoregressive generation.
- Sequences all operations for one token decode step.
- Tracks current layer index, current position, and buffer ownership.

Expected sequence:

1. Receive the input token embedding from `host_if`.
2. For each layer:
   - pre-attention norm
   - Q/K/V projections
   - RoPE
   - KV cache write for current position
   - decode attention score/value path using cached history
   - attention output projection
   - residual add
   - pre-MLP norm
   - gate/up projections
   - SwiGLU
   - down projection
   - residual add
3. Final norm
4. Return the final hidden state to `host_if`

### 5.6 `ddr_if`

Responsibility:

- Provides burst-oriented memory transactions to external DDR3.
- Hides board-specific DDR controller details from the model pipeline.

Interface goals:

- Command request and completion handshake
- Read burst interface
- Write burst interface
- Address generation delegated from higher modules

### 5.7 `onchip_buffer_bank`

Responsibility:

- Owns small local buffers for current hidden state, partial sums, temporary attention data, and double-buffered tiles.

Expected contents:

- Hidden-state ping-pong buffers
- Prompt tile buffers for prefill
- Partial output accumulation buffers
- Small tiles of ternary weights or decoded weight signs
- Temporary Q, K, V vectors
- Temporary attention score storage for the active token window

### 5.8 `prefill_path`

Responsibility:

- Wrapper datapath for prompt ingestion.
- Reuses common compute blocks but schedules them for sequence tiles instead of single-token decode.

Notes:

- This path may use different tiling and buffering from decode.
- The first implementation can keep it simple and less optimized than the final target, but it is still a distinct path.

### 5.9 `decode_path`

Responsibility:

- Wrapper datapath for autoregressive token generation.
- Reuses common compute blocks but is optimized for one-token-at-a-time execution.

### 5.10 `layer_engine`

Responsibility:

- Reusable implementation of one transformer layer.
- Parameterized by layer index, execution mode, and scale metadata.

Subfunctions:

- RMSNorm
- Q/K/V linear projections
- RoPE
- Attention
- Output projection
- Residual path
- RMSNorm
- Gate/up projections
- SwiGLU
- Down projection
- Residual path

Important constraint:

- Only one physical layer engine is instantiated in the first architecture.
- All 24 model layers are executed by reusing this engine over time.
- Prefill and decode use different schedules around the same lower-level compute blocks.

### 5.11 `kv_cache_manager`

Responsibility:

- Owns KV-cache layout and address generation.
- Performs cache writes for prompt tiles during prefill.
- Performs cache writes for the current token and reads for previous tokens during decode.
- Handles button-triggered KV invalidation.

Why separate it:

- The cache layout will likely change during optimization.
- Separating it from generic DDR logic keeps the rest of the design cleaner.

## 6. Core Compute Modules

These are the main reusable datapath blocks below the top-level modules.

### 6.1 `linear_engine`

Responsibility:

- Implements ternary linear operations for BitNet.
- Supports matrix-vector or tiled matrix-vector execution for decode mode.
- Supports matrix-matrix or tiled matrix-matrix style execution for prefill mode when beneficial.

Key idea:

- Ternary weights avoid general multipliers for the weight term.
- Each weight element contributes one of:
  - `+x`
  - `0`
  - `-x`

Open design choice:

- Whether to decode packed ternary weights on the fly or to unpack small tiles into local buffers before accumulation.

### 6.2 `rmsnorm_core`

Responsibility:

- Computes RMSNorm for a vector.

Key operations:

- Sum of squares
- Reciprocal square root or approximation
- Scale by learned weight and quantization scale

### 6.3 `rope_core`

Responsibility:

- Applies rotary position embedding to Q and K vectors for the current token position.

### 6.4 `attention_engine`

Responsibility:

- Computes attention for both prefill and decode.

Subfunctions:

- Prefill score path for prompt tiles
- Decode score path for current Q against cached K entries
- Causal valid-length handling
- Score scaling
- Softmax or a suitable approximation
- Weighted sum over V entries

### 6.5 `swiglu_core`

Responsibility:

- Implements the MLP activation path.
- Consumes gate and up projection results and produces the elementwise gated activation.

### 6.6 `quant_scale_core`

Responsibility:

- Handles activation quantization, dequantization, clipping, and layer-specific scales.

### 6.7 `residual_add_core`

Responsibility:

- Adds residual connections with correct fixed-point growth and saturation handling.

## 7. Resource Mapping Intent

This section does not fix exact utilization. It states where each class of work should live.

### 7.1 DDR3

- Stores packed ternary weights
- Stores per-layer metadata and scales
- Stores the full KV cache
- Stores data too large for BRAM

### 7.2 BRAM and LUTRAM

- Hidden-state ping-pong buffers
- Prompt tiles for prefill
- Temporary Q/K/V vectors
- Partial sums and accumulation buffers
- Small decoded weight tiles
- Short-lived attention score buffers

### 7.3 CLB fabric

- Main control FSMs
- Address generation
- Handshake logic
- Ternary add/subtract/skip datapaths
- Fixed-point glue logic

### 7.4 DSP blocks

- Use only where they clearly help
- Candidate uses include:
  - accumulation trees
  - norm-related arithmetic
  - reciprocal or scaling support
  - selected attention arithmetic

Design rule:

- Do not spend DSPs on general weight multipliers for ternary layers unless measurement shows a clear gain.

## 8. Memory and Data Model

This section defines the architectural intent. Exact bit widths and packing details will be frozen in a separate memory-layout spec.

### 8.1 Weight storage

- All model weights are stored in DDR3.
- Ternary weights are packed offline by host-side scripts.
- Each matrix also stores its scale metadata.
- Weight format should favor burst reads and simple address generation.

### 8.2 KV cache

- KV cache lives in DDR3.
- Cache is indexed by:
  - layer
  - sequence position
  - head
  - feature slice

First-pass constraint:

- Support a reduced context length first.
- The cache manager must make the supported context length configurable.

### 8.3 On-chip buffers

On-chip memory is reserved for:

- Current hidden state
- Next hidden state
- Temporary Q/K/V vectors
- Partial sums
- Small tiles of weights or decoded weight symbols
- Temporary attention score data

### 8.4 Data movement strategy

The first design should favor a simple, explicit schedule:

1. Read a weight tile from DDR3.
2. Read or hold the relevant activation vector.
3. Compute partial sums.
4. Accumulate into a local buffer.
5. Write result to the next stage buffer.

This schedule is not maximally fast, but it is easy to verify and optimize incrementally.

## 9. Interface Philosophy

Each RTL module should use a small, explicit handshake-based interface.

Preferred control signals:

- `clk`
- `rst_n`
- `start`
- `busy`
- `done`
- `error`

Preferred data-flow conventions:

- Valid/ready or request/ack handshake for streams
- Explicit vector length or tile size metadata where needed
- Clear ownership of buffer read and write ports

Rule:

- Do not bury memory side effects inside opaque modules.
- Address generation should remain visible at the controller, cache manager, or memory wrapper level.

## 10. Verification Strategy

The project must be verified from the beginning, not after all RTL is written.

### 9.1 Golden software reference

Create Python reference code that:

- Loads model config and weights from `../bitnet_700M/`
- Implements both prompt prefill and one-token decode in the same tensor order expected by RTL
- Exports packed test vectors for module-level tests
- Produces expected outputs for comparison

### 9.2 Module-level tests

Each module should have:

- Directed tests with small hand-checkable vectors
- Randomized tests against the Python reference where applicable
- Pass/fail self-checking behavior

### 9.3 Integration tests

Integration should proceed in increasing scope:

1. `linear_engine`
2. `rmsnorm_core`
3. `rope_core`
4. `attention_engine`
5. `layer_engine`
6. prefill path
7. decode path
8. full prompt-plus-decode step

### 9.4 Bring-up milestones

- RTL simulation passes unit tests
- RTL simulation passes layer-level tests
- End-to-end prefill matches software for short prompts
- End-to-end decode matches software after prefill
- Synthesis succeeds in Vivado
- Timing and utilization are reviewed
- Hardware bring-up starts only after simulation confidence is adequate

## 11. Implementation Phases

### Phase 0: Specification

Deliverables:

- Architecture spec
- Project structure
- Initial module list
- Interface drafts

### Phase 1: Software tooling

Deliverables:

- Model inspection scripts
- Weight packing scripts
- Golden reference for prefill and one-token decode
- Test-vector generation scripts

### Phase 2: Utility RTL blocks

Deliverables:

- Buffer primitives
- Fixed-point helpers
- Saturation logic
- Residual add
- Quantization helpers

### Phase 3: Core arithmetic blocks

Deliverables:

- `linear_engine`
- `rmsnorm_core`
- `rope_core`
- `swiglu_core`

### Phase 4: Memory-connected compute

Deliverables:

- `kv_cache_manager`
- `attention_engine`
- `ddr_if` wrapper integration

### Phase 5: Layer integration

Deliverables:

- `layer_engine`
- Layer-level testbench

### Phase 6: Prefill path

Deliverables:

- `prefill_controller`
- `prefill_path`
- Prefill integration testbench

### Phase 7: Full decode path

Deliverables:

- `decode_controller`
- `decode_path`
- `top_system`

### Phase 8: Optimization

Deliverables:

- Better buffering
- Better tiling
- Better packing
- Performance and utilization analysis

## 11. Project Structure

Recommended repository layout under `RTL/`:

```text
RTL/
  AGENTS.md
  build/                 # Generated build outputs, logs, reports
  constraints/           # XDC and board constraints
  docs/
    architecture.md      # This document
    memory_layout.md     # Weight/KV/activation packing spec
    interfaces.md        # Interface definitions for top modules
    verification.md      # Verification plan and milestones
  rtl/
    top/
      top_system.v
      prefill_controller.v
      decode_controller.v
      host_if.v
    core/
      prefill_path.v
      decode_path.v
      layer_engine.v
      linear_engine.v
      attention_engine.v
      rmsnorm_core.v
      rope_core.v
      swiglu_core.v
      quant_scale_core.v
      residual_add_core.v
    mem/
      ddr_if.v
      kv_cache_manager.v
      onchip_buffer_bank.v
    common/
      fifo.v
      bram_sp.v
      bram_dp.v
      fixed_point_pkg.vh
      util_macros.vh
  tb/
    unit/
      tb_linear_engine.v
      tb_rmsnorm_core.v
      tb_rope_core.v
      tb_attention_engine.v
    integration/
      tb_layer_engine.v
      tb_prefill_path.v
      tb_decode_step.v
  sim/
    run_unit_tests.tcl
    run_integration_tests.tcl
  scripts/
    python/
      inspect_model.py
      pack_weights.py
      gen_test_vectors.py
      golden_prefill.py
      golden_decode.py
      host_runtime.py
  tools/
    vivado_project.tcl
```

Notes:

- `build/` should contain generated artifacts only.
- `docs/` should be updated whenever an interface or memory format changes.
- `scripts/python/` is part of the hardware workflow because it defines packing and golden behavior.
