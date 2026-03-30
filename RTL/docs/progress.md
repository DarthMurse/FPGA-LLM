# BitNet FPGA Project Progress

## 1. Purpose

This file tracks current project status, completed design work, active decisions, and upcoming tasks.

The design documents in `docs/` should describe the design itself.
This file should describe project progress.

## 2. Completed Work

The following documents are drafted:

- `docs/architecture.md`
- `docs/interfaces.md`
- `docs/memory_layout.md`

Current high-level architecture decisions:

- prefill and decode are treated as separate execution paths
- tokenizer, embedding lookup, final RMSNorm, LM head, softmax, and sampling are on the host
- host and FPGA communicate over Gigabit Ethernet
- model weights and metadata are initialized from SD card or QSPI into DDR
- activations use symmetric per-token `int8` quantization with `fp16` scale
- weight scales use `fp16`
- KV cache can be cleared by a hardware button signal
- the first RTL compute target is `linear_engine` in decode-style matrix-vector mode
- first-pass RTL simulation uses `iverilog`
- first-pass `linear_engine` boundary uses local vector load, full-row packed weight load, and streamed row outputs
- first-pass `linear_engine` uses modest chunk-parallel accumulation intended to map onto DSP-friendly arithmetic
- Vivado out-of-context synthesis was run for `linear_engine` on `xc7a35tfgg484-2` with `COLS=128`, `ROWS=16`, `IN_W=8`, `ACC_W=16`
- measured `PAR_COLS` sweep result: `PAR_COLS=2/4/8` all mapped to `1` DSP, while LUT cost and timing worsened as `PAR_COLS` increased
- `linear_engine` was then redesigned with one partial accumulator per lane and re-synthesized
- measured result for the partial-accumulator redesign: Vivado used `0` DSPs for `PAR_COLS=2/4/8`, so the structure still did not induce the intended multi-DSP mapping
- `docs/module_contracts.md` now defines the intended module interfaces and interactions for the rest of the project
- `docs/test_matrix.md` now defines unit and integration tests for each major module
- empty RTL scaffolds now exist for the planned top-level, controller, datapath, and helper modules
- empty testbench scaffolds now exist for the planned unit and integration tests

## 3. Current Open Decisions

These items still need to be frozen:

1. exact host embedding payload packing
2. exact returned hidden-state payload packing
3. exact KV numeric format
4. exact prefill tile size
5. exact SD versus QSPI boot path for the first implementation
6. internal stream width between `host_if` and controllers
7. how to restructure `linear_engine` so multiple DSPs are actually used instead of LUT/carry logic
8. whether the next attempt should use explicit DSP48-oriented pipeline staging instead of inference-only RTL
9. what the default `PAR_COLS` should be after a real multi-DSP microarchitecture is implemented
10. when `linear_engine` should move from full-row input payloads to packed-word streaming
11. when `linear_engine` should add prefill tile support and requantization

## 4. Current Status

Project phase:

- architecture definition and first module bring-up

Current status summary:

- top-level architecture is drafted
- module interfaces are drafted
- DDR and buffer layout are drafted
- `docs/verification.md` is drafted
- `docs/linear_engine.md` is drafted
- first RTL module `rtl/linear_engine.v` is implemented
- first self-checking testbench `tb/tb_linear_engine.v` is implemented
- first unit test passes under `iverilog`
- `linear_engine` has been revised from a fully serial accumulator to a chunk-parallel accumulator
- first Vivado OOC synthesis sweep is completed for the current `linear_engine`
- current RTL only uses `1` DSP across `PAR_COLS=2`, `4`, and `8`, so the chunk sum is still mostly LUT-based
- option-1 redesign using multiple partial accumulators is implemented and unit-tested
- Vivado synthesis shows the option-1 redesign uses `0` DSPs and therefore is not the final solution
- module ownership, interaction boundaries, and verification targets are now documented for the remaining blocks
- RTL and testbench file skeletons now exist for the remaining planned modules and tests

## 5. Recommended Near-Term Tasks

Recommended task order:

1. implement `residual_add_core` against `docs/module_contracts.md`
2. fill in `tb_residual_add_core.v`
3. implement `quant_scale_core` against `docs/module_contracts.md`
4. fill in `tb_quant_scale_core.v`
5. implement `kv_cache_manager` address logic against `docs/module_contracts.md`
6. freeze the KV numeric format

## 6. Latest Verification Result

Most recent completed check:

- compiled with `iverilog -g2012`
- ran `tb/tb_linear_engine.v` against `rtl/linear_engine.v`
- observed pass on decode-style ternary matrix-vector accumulation and output backpressure handling
- ran Vivado 2024.2 out-of-context synthesis for `PAR_COLS=2`, `4`, and `8` at a 10.000 ns clock target
- observed resource and timing trend for `COLS=128`, `ROWS=16`:
- `PAR_COLS=2`: `2011` Slice LUTs, `5497` registers, `1` DSP, `WNS = 0.680 ns`
- `PAR_COLS=4`: `2110` Slice LUTs, `5497` registers, `1` DSP, `WNS = -1.341 ns`
- `PAR_COLS=8`: `2170` Slice LUTs, `5490` registers, `1` DSP, `WNS = -2.420 ns`
- redesigned `linear_engine` around multiple partial accumulators and reran Vivado OOC synthesis at a 10.000 ns clock target
- observed redesigned resource and timing trend for `COLS=128`, `ROWS=16`:
- `PAR_COLS=2`: `1882` Slice LUTs, `5501` registers, `0` DSPs, `WNS = 1.633 ns`
- `PAR_COLS=4`: `2234` Slice LUTs, `5545` registers, `0` DSPs, `WNS = 1.012 ns`
- `PAR_COLS=8`: `2479` Slice LUTs, `5607` registers, `0` DSPs, `WNS = -0.306 ns`

## 6. Update Rules

When project status changes:

- update this file
- update the relevant design doc only if the design itself changed

When a decision is finalized:

- move it from `Current Open Decisions` to `Completed Work` or the relevant dated note
