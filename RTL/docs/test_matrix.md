# BitNet FPGA RTL Test Matrix

## 1. Goal

This document defines the unit tests and integration tests that should exist for each module.

The intent is:

- you implement the RTL
- I provide the expected verification contract
- each module is considered done only after its unit test passes

## 2. Testbench Rules

Every testbench should be:

- self-checking
- deterministic
- small enough to debug in waves
- explicit about expected numeric behavior
- written first for reduced dimensions before full model dimensions

Use naming:

- unit testbenches: `tb_<module>.v`
- integration testbenches: `tb_int_<subsystem>.v`

## 3. Unit Test Matrix

### 3.1 `linear_engine`

Unit test name:

- `tb_linear_engine`

Must verify:

- input stream handshake
- weight stream handshake
- exact ternary accumulation
- output backpressure
- `busy_o` and `done_o`
- refusal to accept a new job while busy

Golden model:

- direct integer accumulation in the testbench

### 3.2 `quant_scale_core`

Unit test name:

- `tb_quant_scale_core`

Must verify:

- dequant/requant round-trip on simple vectors
- saturation at positive and negative limits
- scale metadata propagation
- end-of-vector handling
- output stability during backpressure

Golden model:

- software-style fixed-point reference in the testbench

### 3.3 `residual_add_core`

Unit test name:

- `tb_residual_add_core`

Must verify:

- aligned stream consumption from both inputs
- exact addition on nominal values
- saturation or wrap policy, whichever the spec freezes
- mismatched `last` handling reports an error if that is the chosen policy

Golden model:

- elementwise integer add

### 3.4 `rmsnorm_core`

Unit test name:

- `tb_rmsnorm_core`

Must verify:

- accumulation of squares
- normalization gain on a short vector
- learned weight application
- acceptable approximation error bounds

Golden model:

- precomputed expected vectors from a reference script or hardcoded examples

### 3.5 `rope_core`

Unit test name:

- `tb_rope_core`

Must verify:

- no-op behavior at position `0` if the chosen convention implies that
- correct rotation on a small `head_dim`
- both Q and K transformed consistently

Golden model:

- precomputed small-vector RoPE examples

### 3.6 `swiglu_core`

Unit test name:

- `tb_swiglu_core`

Must verify:

- aligned gate and up streams
- activation and multiply behavior
- zero gate edge cases
- negative gate edge cases

Golden model:

- precomputed small-vector examples

### 3.7 `kv_cache_manager`

Unit test name:

- `tb_kv_cache_manager`

Must verify:

- address generation matches `docs/memory_layout.md`
- read and write requests map to the expected DDR addresses
- logical clear invalidates old cache state
- read order across positions is correct

Golden model:

- direct address calculation in the testbench

### 3.8 `attention_engine`

Unit test name:

- `tb_attention_engine`

Must verify:

- decode read pattern for a short context
- causal handling
- output shape and ordering
- handshake with KV read data source

Golden model:

- tiny head-dim reference vectors prepared offline

### 3.9 `layer_engine`

Unit test name:

- `tb_layer_engine`

Must verify:

- submodule invocation order
- correct layer-level start/busy/done behavior
- correct sequencing for one decode step
- correct sequencing for one prefill tile
- expected status/error behavior on a forced submodule fault

Golden model:

- use stubs for submodules first
- later use real lower modules once their unit tests pass

### 3.10 `decode_controller`

Unit test name:

- `tb_decode_controller`

Must verify:

- host command acceptance
- one embedding load then 24 layer invocations
- correct final result return
- no new command acceptance while busy

Golden model:

- stubbed `layer_engine` with deterministic latency and output

### 3.11 `prefill_controller`

Unit test name:

- `tb_prefill_controller`

Must verify:

- prompt tile acceptance
- correct tile/layer loops
- final tile result behavior
- non-final tile ACK behavior

Golden model:

- stubbed `layer_engine`

### 3.12 `host_if`

Unit test name:

- `tb_host_if`

Must verify:

- packet decode into commands
- payload streaming behavior
- response packet formatting
- sequence ID and session ID echoing

Golden model:

- hardcoded packet frames

### 3.13 `ddr_if`

Unit test name:

- `tb_ddr_if`

Must verify:

- burst command formatting
- read data ordering
- write data framing
- backpressure handling at controller boundary

Golden model:

- behavioral DDR shim

## 4. Integration Test Matrix

### 4.1 Arithmetic Chain

Integration test name:

- `tb_int_arith_chain`

Modules:

- `linear_engine`
- `quant_scale_core`
- `residual_add_core`

Must verify:

- one projection followed by requantization
- residual add of the projected output with a bypass vector

### 4.2 Attention Slice

Integration test name:

- `tb_int_attention_slice`

Modules:

- `linear_engine`
- `rope_core`
- `kv_cache_manager`
- `attention_engine`

Must verify:

- Q/K/V generation flow
- KV write then KV read
- one decode attention step on a very short context

### 4.3 MLP Slice

Integration test name:

- `tb_int_mlp_slice`

Modules:

- `rmsnorm_core`
- `linear_engine` for gate
- `linear_engine` for up
- `swiglu_core`
- `linear_engine` for down
- `residual_add_core`

Must verify:

- correct block ordering
- correct stream connectivity
- expected completion signaling

### 4.4 Layer Decode

Integration test name:

- `tb_int_layer_decode`

Modules:

- real `layer_engine`
- real arithmetic submodules
- stubbed `ddr_if` if needed

Must verify:

- one complete decode layer call
- KV interaction
- residual boundaries
- final output timing and formatting

### 4.5 Layer Prefill

Integration test name:

- `tb_int_layer_prefill`

Modules:

- real `layer_engine`
- real arithmetic submodules
- stubbed or behavioral KV/DDR backend

Must verify:

- tile iteration behavior
- KV writes for multiple positions

### 4.6 Decode Pipeline

Integration test name:

- `tb_int_decode_pipeline`

Modules:

- `host_if`
- `decode_controller`
- `layer_engine`
- `kv_cache_manager`
- behavioral `ddr_if`

Must verify:

- one end-to-end decode request from host command to returned hidden state

### 4.7 Prefill Pipeline

Integration test name:

- `tb_int_prefill_pipeline`

Modules:

- `host_if`
- `prefill_controller`
- `layer_engine`
- `kv_cache_manager`
- behavioral `ddr_if`

Must verify:

- one end-to-end prefill tile request from host command to returned final hidden state

## 5. Recommended Bring-Up Order

Implement and test in this order:

1. `residual_add_core`
2. `quant_scale_core`
3. `linear_engine`
4. `kv_cache_manager`
5. `rope_core`
6. `swiglu_core`
7. `rmsnorm_core`
8. `tb_int_arith_chain`
9. `attention_engine`
10. `tb_int_attention_slice`
11. `layer_engine`
12. `tb_int_layer_decode`
13. `tb_int_layer_prefill`
14. controllers
15. host and DDR wrappers
16. full decode and prefill pipeline integration tests

## 6. Completion Rule

A module is ready for integration only if:

- its interface is frozen in `docs/module_contracts.md`
- its unit test exists
- its unit test passes
- any open numeric assumptions are documented
