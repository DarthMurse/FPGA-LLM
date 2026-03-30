# BitNet FPGA RTL Verification Plan

## 1. Goal

This document defines the first verification strategy for the RTL project.

The goal is to make every module verifiable in isolation before system integration.
That is especially important here because the user wants to learn RTL and FPGA bring-up through a top-down, test-first flow.

## 2. Verification Principles

The project will follow these rules:

- every RTL module gets a self-checking unit testbench
- design assumptions must be written in docs before implementation when practical
- correctness comes before throughput optimization
- simulation should use small dimensions first
- fixed-point behavior must be explicit in both RTL and tests
- integration happens only after unit behavior is stable

## 3. Tooling

First-pass simulation tool:

- `iverilog`

Reason:

- it is already available in the environment
- it is enough for small RTL unit tests
- it keeps the bring-up loop simple before Vivado project setup

Later verification may add:

- Vivado `xsim`
- waveform dumps for deeper debug
- software-side golden vector generators

## 4. Test Levels

### 4.1 Document-level freeze

Before implementing a module:

- define the responsibility
- define the interface
- freeze numeric assumptions that affect the boundary
- describe what the unit test must prove

### 4.2 Module unit test

For each RTL module:

- instantiate only that module
- drive realistic handshake behavior
- include at least one nominal case
- include at least one edge case
- include at least one backpressure or stall case if streaming is used
- fail the testbench automatically on mismatch

### 4.3 Subsystem integration test

After several unit-tested blocks exist:

- connect them in a small integration harness
- verify transaction ordering and control sequencing
- compare selected outputs against a software golden reference

### 4.4 Board-level bring-up

Only after simulation is stable:

- synthesize selected modules
- check resource usage
- run simple hardware diagnostics

## 5. Initial Module Verification Order

Recommended order:

1. `linear_engine`
2. `residual_add_core`
3. `quant_scale_core`
4. `kv_cache_manager` address generator portions
5. `rmsnorm_core`
6. `rope_core`
7. `swiglu_core`
8. `attention_engine`
9. controllers and path wrappers

This order starts with exact integer datapaths and delays the more numerically sensitive blocks until the test infrastructure is established.

## 6. First Unit-Test Checklist for `linear_engine`

The first `linear_engine` testbench should verify:

- input vector load handshake
- row payload load handshake
- correct decode of `+1`, `0`, and `-1`
- exact row sum for several rows
- output stream backpressure handling
- `busy_o` during active work
- single-cycle `done_o` on job completion
- refusal to accept new work while busy

Suggested first dimensions:

- `COLS = 8`
- `ROWS = 3`
- `IN_W = 8`
- `ACC_W = 16`

Reason:

- small enough to inspect by hand
- large enough to exercise all ternary cases

## 7. Result Recording

When a module is implemented and tested:

- update `docs/progress.md`
- note the verification method used
- note any known gaps that remain

## 8. Immediate Next Step

The current next step is:

1. finish the detailed `linear_engine` spec
2. implement the first decode-style `linear_engine`
3. add a self-checking `iverilog` testbench
4. run the test and record the result in `docs/progress.md`
