# BitNet FPGA RTL Module Contracts

## 1. Goal

This document freezes the module boundaries for the first RTL implementation.

The purpose is to let one person implement RTL from AMD documentation while another person defines:

- what each module is responsible for
- which side owns each buffer
- how modules interact
- what each module must prove in unit test

This document is intentionally more concrete than `docs/architecture.md`.

## 2. Design Rules

All modules in this document follow these rules:

- single local `clk_i`
- active-low reset `rst_n_i`
- coarse operations use `start/busy/done`
- streams use `valid/ready/last`
- configuration values remain stable from `start_i` until `done_o`
- outputs remain stable while `valid=1` and `ready=0`

## 3. Common Stream Types

### 3.1 Quantized vector stream

Used for activations, hidden states, and intermediate vectors.

Fields:

- `vec_valid`
- `vec_ready`
- `vec_data`
- `vec_last`
- `vec_scale`
- `vec_id`

First-pass notes:

- `vec_data` width can stay at `64` bits between controllers and datapath wrappers
- `vec_scale` is `fp16`
- `vec_id` is a token index, row index, or debug tag depending on context

### 3.2 DDR read command

Fields:

- `rd_cmd_valid`
- `rd_cmd_ready`
- `rd_addr`
- `rd_len`
- `rd_tag`

### 3.3 DDR read data

Fields:

- `rd_data_valid`
- `rd_data_ready`
- `rd_data`
- `rd_last`
- `rd_tag`

### 3.4 DDR write command/data

Fields:

- `wr_cmd_valid`
- `wr_cmd_ready`
- `wr_addr`
- `wr_len`
- `wr_tag`
- `wr_data_valid`
- `wr_data_ready`
- `wr_data`
- `wr_last`

## 4. Compute Path Decomposition

The reusable per-layer datapath is decomposed into these modules:

- `rmsnorm_core`
- `linear_engine`
- `rope_core`
- `kv_cache_manager`
- `attention_engine`
- `residual_add_core`
- `swiglu_core`
- `quant_scale_core`
- `layer_engine`

The intended dependency order is:

```text
controller
  -> layer_engine
      -> rmsnorm_core
      -> linear_engine
      -> rope_core
      -> kv_cache_manager
      -> attention_engine
      -> residual_add_core
      -> swiglu_core
      -> quant_scale_core
```

## 5. Per-Module Contracts

### 5.1 `linear_engine`

Responsibility:

- compute one quantized ternary linear transform
- consume one input vector
- consume packed weights for one or more rows
- emit signed accumulators or requantized outputs depending on mode

Owned by:

- datapath implementation

Not owned by:

- DDR arbitration
- matrix address generation
- residual path

Required modes:

- decode matrix-vector first
- prefill tile mode later

Suggested top-level contract:

```verilog
module linear_engine #(
    parameter integer COLS = 128,
    parameter integer ROWS = 16,
    parameter integer IN_W = 8,
    parameter integer ACC_W = 24
) (
    input  wire                    clk_i,
    input  wire                    rst_n_i,
    input  wire                    start_i,
    input  wire [15:0]             cfg_cols_i,
    input  wire [15:0]             cfg_rows_i,
    output wire                    busy_o,
    output wire                    done_o,

    input  wire                    vec_valid_i,
    output wire                    vec_ready_o,
    input  wire [63:0]             vec_data_i,
    input  wire                    vec_last_i,

    input  wire                    weight_valid_i,
    output wire                    weight_ready_o,
    input  wire [63:0]             weight_data_i,
    input  wire                    weight_last_i,

    output wire                    out_valid_o,
    input  wire                    out_ready_i,
    output wire [63:0]             out_data_o,
    output wire                    out_last_o
);
```

Interaction notes:

- `layer_engine` owns configuration and matrix selection
- `ddr_if` and weight unpack logic provide packed row or packed word data
- `quant_scale_core` may consume the output after accumulation if requantization is separated

### 5.2 `quant_scale_core`

Responsibility:

- dequantize if needed for local arithmetic
- requantize accumulator outputs to `int8`
- compute or forward `fp16` scale metadata
- clip and saturate safely

Suggested contract:

```verilog
module quant_scale_core (
    input  wire         clk_i,
    input  wire         rst_n_i,
    input  wire         start_i,
    input  wire [15:0]  cfg_elem_count_i,
    input  wire [15:0]  cfg_mode_i,
    input  wire [15:0]  in_scale_i,
    input  wire [15:0]  weight_scale_i,
    output wire         busy_o,
    output wire         done_o,

    input  wire         in_valid_i,
    output wire         in_ready_o,
    input  wire [63:0]  in_data_i,
    input  wire         in_last_i,

    output wire         out_valid_o,
    input  wire         out_ready_i,
    output wire [63:0]  out_data_o,
    output wire         out_last_o,
    output wire [15:0]  out_scale_o
);
```

Interaction notes:

- first integration can keep this block after `linear_engine`
- if the arithmetic later stays fully integer between blocks, this core can move to layer boundaries

### 5.3 `residual_add_core`

Responsibility:

- add two vectors with explicit growth and saturation behavior
- preserve stream order

Suggested contract:

```verilog
module residual_add_core (
    input  wire         clk_i,
    input  wire         rst_n_i,
    input  wire         start_i,
    input  wire [15:0]  cfg_elem_count_i,
    output wire         busy_o,
    output wire         done_o,

    input  wire         a_valid_i,
    output wire         a_ready_o,
    input  wire [63:0]  a_data_i,
    input  wire         a_last_i,

    input  wire         b_valid_i,
    output wire         b_ready_o,
    input  wire [63:0]  b_data_i,
    input  wire         b_last_i,

    output wire         out_valid_o,
    input  wire         out_ready_i,
    output wire [63:0]  out_data_o,
    output wire         out_last_o
);
```

### 5.4 `rmsnorm_core`

Responsibility:

- compute RMSNorm over one vector
- apply learned weight
- emit normalized vector in the agreed numeric format

Suggested contract:

```verilog
module rmsnorm_core (
    input  wire         clk_i,
    input  wire         rst_n_i,
    input  wire         start_i,
    input  wire [15:0]  cfg_elem_count_i,
    input  wire [31:0]  cfg_weight_base_addr_i,
    output wire         busy_o,
    output wire         done_o,

    input  wire         in_valid_i,
    output wire         in_ready_o,
    input  wire [63:0]  in_data_i,
    input  wire         in_last_i,
    input  wire [15:0]  in_scale_i,

    output wire         out_valid_o,
    input  wire         out_ready_i,
    output wire [63:0]  out_data_o,
    output wire         out_last_o,
    output wire [15:0]  out_scale_o
);
```

### 5.5 `rope_core`

Responsibility:

- apply rotary position embedding to Q and K

Suggested contract:

```verilog
module rope_core (
    input  wire         clk_i,
    input  wire         rst_n_i,
    input  wire         start_i,
    input  wire [15:0]  cfg_head_dim_i,
    input  wire [15:0]  cfg_num_heads_i,
    input  wire [31:0]  cfg_position_i,
    output wire         busy_o,
    output wire         done_o,

    input  wire         q_in_valid_i,
    output wire         q_in_ready_o,
    input  wire [63:0]  q_in_data_i,
    input  wire         q_in_last_i,

    input  wire         k_in_valid_i,
    output wire         k_in_ready_o,
    input  wire [63:0]  k_in_data_i,
    input  wire         k_in_last_i,

    output wire         q_out_valid_o,
    input  wire         q_out_ready_i,
    output wire [63:0]  q_out_data_o,
    output wire         q_out_last_o,

    output wire         k_out_valid_o,
    input  wire         k_out_ready_i,
    output wire [63:0]  k_out_data_o,
    output wire         k_out_last_o
);
```

### 5.6 `swiglu_core`

Responsibility:

- consume gate and up vectors
- perform elementwise activation and multiplication

Suggested contract:

```verilog
module swiglu_core (
    input  wire         clk_i,
    input  wire         rst_n_i,
    input  wire         start_i,
    input  wire [15:0]  cfg_elem_count_i,
    output wire         busy_o,
    output wire         done_o,

    input  wire         gate_valid_i,
    output wire         gate_ready_o,
    input  wire [63:0]  gate_data_i,
    input  wire         gate_last_i,

    input  wire         up_valid_i,
    output wire         up_ready_o,
    input  wire [63:0]  up_data_i,
    input  wire         up_last_i,

    output wire         out_valid_o,
    input  wire         out_ready_i,
    output wire [63:0]  out_data_o,
    output wire         out_last_o
);
```

### 5.7 `attention_engine`

Responsibility:

- compute attention score/value path
- support both decode and prefill modes
- hide the softmax approximation choice from higher controllers

Suggested contract:

```verilog
module attention_engine (
    input  wire         clk_i,
    input  wire         rst_n_i,
    input  wire         start_i,
    input  wire [1:0]   cfg_mode_i,
    input  wire [15:0]  cfg_num_heads_i,
    input  wire [15:0]  cfg_head_dim_i,
    input  wire [15:0]  cfg_context_used_i,
    output wire         busy_o,
    output wire         done_o,

    input  wire         q_valid_i,
    output wire         q_ready_o,
    input  wire [63:0]  q_data_i,
    input  wire         q_last_i,

    output wire         kv_rd_cmd_valid_o,
    input  wire         kv_rd_cmd_ready_i,
    output wire [31:0]  kv_rd_addr_o,
    output wire [15:0]  kv_rd_len_o,

    input  wire         kv_rd_data_valid_i,
    output wire         kv_rd_data_ready_o,
    input  wire [63:0]  kv_rd_data_i,
    input  wire         kv_rd_data_last_i,

    output wire         out_valid_o,
    input  wire         out_ready_i,
    output wire [63:0]  out_data_o,
    output wire         out_last_o
);
```

### 5.8 `kv_cache_manager`

Responsibility:

- own KV-cache logical indexing and physical address generation
- handle write requests from `layer_engine`
- issue DDR reads for `attention_engine`
- manage logical clear state

Suggested contract:

```verilog
module kv_cache_manager (
    input  wire         clk_i,
    input  wire         rst_n_i,
    input  wire         kv_clear_req_i,
    output wire         kv_clear_done_o,

    input  wire         wr_req_valid_i,
    output wire         wr_req_ready_o,
    input  wire [7:0]   wr_layer_i,
    input  wire [0:0]   wr_kind_i,
    input  wire [15:0]  wr_position_i,
    input  wire [63:0]  wr_data_i,
    input  wire         wr_last_i,

    input  wire         rd_req_valid_i,
    output wire         rd_req_ready_o,
    input  wire [7:0]   rd_layer_i,
    input  wire [0:0]   rd_kind_i,
    input  wire [15:0]  rd_position_start_i,
    input  wire [15:0]  rd_position_count_i,

    output wire         ddr_rd_cmd_valid_o,
    input  wire         ddr_rd_cmd_ready_i,
    output wire [31:0]  ddr_rd_addr_o,
    output wire [15:0]  ddr_rd_len_o,

    input  wire         ddr_rd_data_valid_i,
    output wire         ddr_rd_data_ready_o,
    input  wire [63:0]  ddr_rd_data_i,
    input  wire         ddr_rd_data_last_i,

    output wire         rd_data_valid_o,
    input  wire         rd_data_ready_i,
    output wire [63:0]  rd_data_o,
    output wire         rd_data_last_o
);
```

### 5.9 `layer_engine`

Responsibility:

- orchestrate one transformer layer
- select the required weight matrices
- sequence the sub-blocks
- expose a single layer-level contract to the controllers

Suggested contract:

```verilog
module layer_engine (
    input  wire         clk_i,
    input  wire         rst_n_i,
    input  wire         start_i,
    input  wire [1:0]   mode_i,
    input  wire [7:0]   layer_idx_i,
    input  wire [15:0]  position_i,
    input  wire [15:0]  context_used_i,
    output wire         busy_o,
    output wire         done_o,
    output wire [31:0]  status_o,

    input  wire         in_valid_i,
    output wire         in_ready_o,
    input  wire [63:0]  in_data_i,
    input  wire         in_last_i,
    input  wire [15:0]  in_scale_i,

    output wire         out_valid_o,
    input  wire         out_ready_i,
    output wire [63:0]  out_data_o,
    output wire         out_last_o,
    output wire [15:0]  out_scale_o
);
```

Interaction notes:

- `decode_controller` and `prefill_controller` talk only to `layer_engine`, not directly to all submodules
- `layer_engine` owns submodule sequencing
- `kv_cache_manager` and `ddr_if` remain separate because their optimizations are likely to change

## 6. Controller-Level Interactions

### 6.1 `decode_controller`

Owns:

- command acceptance from `host_if`
- one-token embedding buffering
- repeated invocation of `layer_engine` from layer `0` to `23`
- final result return to `host_if`

Calls into:

- `layer_engine`
- `kv_cache_manager`

Must not own:

- arithmetic implementation details

### 6.2 `prefill_controller`

Owns:

- prompt tile buffering
- tile/layer loops
- prompt completion policy

Calls into:

- `layer_engine`
- `kv_cache_manager`

### 6.3 `host_if`

Owns:

- protocol termination
- packet/header parsing
- response formatting

Must not own:

- layer scheduling
- model arithmetic

### 6.4 `ddr_if`

Owns:

- DDR controller adaptation
- read and write burst mechanics

Must not own:

- KV address policy
- layer ordering
- matrix interpretation

## 7. Recommended Implementation Ownership

You should implement first:

1. `residual_add_core`
2. `quant_scale_core`
3. `rmsnorm_core`
4. `rope_core`
5. `swiglu_core`
6. `kv_cache_manager` address logic
7. `attention_engine`
8. `layer_engine`
9. controllers

I should maintain:

- module specs
- interface stability
- unit test definitions
- integration test definitions
- progress tracking
