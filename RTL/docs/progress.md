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

## 3. Current Open Decisions

These items still need to be frozen:

1. exact host embedding payload packing
2. exact returned hidden-state payload packing
3. exact KV numeric format
4. exact prefill tile size
5. exact `linear_engine` tile sizes
6. exact SD versus QSPI boot path for the first implementation
7. internal stream width between `host_if` and controllers
8. whether `linear_engine` streams outputs or writes directly into `onchip_buffer_bank`

## 4. Current Status

Project phase:

- architecture and interface definition

Current status summary:

- top-level architecture is drafted
- module interfaces are drafted
- DDR and buffer layout are drafted
- no RTL modules have been implemented yet
- no simulation environment has been created yet

## 5. Recommended Near-Term Tasks

Recommended task order:

1. freeze the remaining `linear_engine` assumptions
2. freeze the KV numeric format
3. decide whether SD or QSPI is the first boot target
4. draft `docs/verification.md`
5. write the detailed `linear_engine` module spec
6. implement `linear_engine` and its unit testbench

## 6. Update Rules

When project status changes:

- update this file
- update the relevant design doc only if the design itself changed

When a decision is finalized:

- move it from `Current Open Decisions` to `Completed Work` or the relevant dated note
