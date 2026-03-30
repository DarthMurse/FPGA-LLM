# BitNet FPGA RTL `linear_engine` Spec

## 1. Goal

This document freezes the first RTL target for `linear_engine`.

The purpose of the first implementation is not to reach final throughput.
The purpose is to create a correct, testable ternary linear core that fits the top-down architecture and can be reused later inside the transformer layer engine.

## 2. Scope

This first-pass `linear_engine` supports:

- decode-style matrix-vector execution
- ternary weights encoded with the 2-bit format from `docs/memory_layout.md`
- signed `int8` activations
- signed integer accumulation
- chunked parallel accumulation using a small number of lanes per cycle
- one reusable RTL module with parameterized dimensions

This first-pass `linear_engine` does not yet support:

- prefill matrix-matrix execution
- direct DDR reads
- weight-scale application
- activation requantization
- bias terms
- sparse skipping beyond the implicit ternary zero skip

Those features will be layered on after the base datapath is proven correct.

## 3. Why This Is the Right First Block

`linear_engine` is the most important reusable compute block in the design.
It appears in:

- Q projection
- K projection
- V projection
- attention output projection
- MLP gate projection
- MLP up projection
- MLP down projection

If we freeze and test this block early, later modules can reuse a known-good datapath and focus on scheduling and memory movement.

## 4. First-Pass Design Decisions

The following decisions are now frozen for the first RTL version.

### 4.1 Execution mode

Use decode-style matrix-vector only.

Reason:

- it is the simpler case
- it matches the most latency-sensitive inference path
- it avoids committing to prefill tile scheduling too early

### 4.2 Dataflow

The engine works in two phases:

1. load one input vector into local storage
2. consume packed weights and produce output accumulations

Reason:

- the input vector is reused for every output row
- storing the vector locally is simple and realistic for FPGA implementation
- this keeps the first interface independent from DDR details

### 4.3 Weight packing

Use 2-bit ternary coding:

- `2'b00` = `0`
- `2'b01` = `+1`
- `2'b10` = `-1`
- `2'b11` = reserved and treated as illegal input in verification

The first RTL module accepts one packed row at a time.

### 4.4 Internal arithmetic

Input activation format:

- signed `int8`

Output format:

- signed accumulator

First-pass accumulator rule:

```text
ACC_W >= IN_W + ceil(log2(COLS)) + 1
```

For bring-up we use a conservative fixed accumulator width parameter and verify exact integer sums in simulation.

### 4.5 DSP usage

The first RTL version should not be fully serial across columns.

Use a parameterized parallel chunk size:

- `PAR_COLS` input elements are processed per cycle
- each chunk forms several signed contributions in parallel
- each lane owns an independent running partial accumulator
- the final row result is formed by reducing the partial accumulators

Design intent:

- ternary decode and sign selection remain simple logic
- the parallel signed additions are written as multiple independent accumulation paths so Vivado has a better chance to use multiple DSPs instead of only one final adder
- the default lane count should stay modest because the AX7035B only has 90 DSPs total and other blocks will need some of that budget

First-pass recommendation:

- start with `PAR_COLS = 4`

Reason:

- it cuts decode latency versus a 1-column-per-cycle loop
- it is still small enough to understand and verify
- it leaves room for later `rmsnorm_core`, attention math, and scaling paths to use DSPs too

### 4.6 Output schedule

The engine emits one completed output row at a time using a `valid/ready` stream.

Reason:

- this matches later composition with quantization and residual blocks
- it avoids requiring a large full-output buffer inside the engine

### 4.7 Interface style

Use:

- `start/busy/done` for one matrix-vector job
- `valid/ready` for vector load, packed weight row input, and output row stream

Reason:

- this follows the project-wide interface rules in `docs/interfaces.md`
- it is easy to test and easy to embed in a future controller

## 5. Functional Model

For each output row `r`:

```text
acc[r] = sum over c of ternary(w[r][c]) * x[c]
```

Where:

- `x[c]` is the signed input activation
- `ternary(w[r][c])` is in `{-1, 0, +1}`

Equivalent contribution rule:

- `+1` adds `x[c]`
- `0` skips
- `-1` subtracts `x[c]`

## 6. First RTL Boundary

Suggested RTL module boundary:

```verilog
module linear_engine #(
    parameter integer COLS = 16,
    parameter integer ROWS = 4,
    parameter integer PAR_COLS = 4,
    parameter integer IN_W = 8,
    parameter integer ACC_W = 16
) (
    input  wire                     clk_i,
    input  wire                     rst_n_i,
    input  wire                     start_i,
    output wire                     busy_o,
    output wire                     done_o,

    input  wire                     vec_valid_i,
    output wire                     vec_ready_o,
    input  wire signed [IN_W-1:0]   vec_data_i,
    input  wire                     vec_last_i,

    input  wire                     row_valid_i,
    output wire                     row_ready_o,
    input  wire [2*COLS-1:0]        row_weights_i,
    input  wire                     row_last_i,

    output wire                     out_valid_o,
    input  wire                     out_ready_i,
    output wire signed [ACC_W-1:0]  out_data_o,
    output wire                     out_last_o
);
```

Notes:

- one row payload contains all packed ternary weights for one matrix row
- this is not the final high-performance interface
- this is the cleanest interface for the first self-checking testbench

## 7. First Microarchitecture

The first RTL implementation uses a simple chunk-parallel datapath with multiple partial accumulators.

### 7.1 State machine

States:

- `IDLE`
- `LOAD_VEC`
- `LOAD_ROWS`
- `CALC`
- `OUT`

### 7.2 Storage

Local storage:

- input vector register array of depth `COLS`
- row buffer register array of depth `ROWS`
- `PAR_COLS` partial accumulator registers
- row and column counters
- a final reduction path over the partial accumulators

Reason:

- this is small enough for simulation and initial FPGA mapping
- it makes expected behavior obvious

### 7.3 Accumulation loop

For one row:

1. clear accumulator
2. inspect `PAR_COLS` columns in parallel
3. decode each 2-bit ternary weight
4. update one partial accumulator per lane with add, subtract, or skip contribution
5. after the last chunk, reduce the partial accumulators into the final row sum
6. latch the final result to the output stream

This is still intentionally simple, but it is no longer purely serial.
We can later increase `PAR_COLS`, pipeline the chunk sum, or replace the row-wide interface with a DDR-oriented streaming interface once correctness and resource usage are understood.

## 8. Verification Targets

The first testbench must cover:

- positive, zero, and negative ternary weights
- multiple rows
- exact signed accumulation
- stream backpressure on output
- illegal early `start`
- correct `done` pulse at end of job

The golden model for the first testbench can be written directly in Verilog testbench code because the arithmetic is still small and exact.

## 9. Planned Follow-On Work

After this block is stable, the next likely changes are:

1. add a packed-word streaming weight interface instead of full-row payloads
2. add per-matrix weight scale handling
3. add output requantization and scale generation
4. add prefill tile support
5. connect the block into a higher-level `layer_engine`
