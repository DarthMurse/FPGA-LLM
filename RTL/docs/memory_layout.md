# BitNet FPGA RTL Memory Layout Spec

## 1. Goal

This document defines the first-pass memory organization for the FPGA design.

It covers:

- non-volatile model storage and DDR initialization
- DDR region allocation
- ternary weight packing
- scale storage
- activation layout
- KV-cache layout
- on-chip BRAM and LUTRAM buffers

## 2. Fixed Decisions

The following items are now fixed for the first implementation:

- model data is initialized from SD card or QSPI flash, not streamed from the host during inference
- activations use 8-bit symmetric integer quantization
- activation scale is stored in `fp16`
- weight scale is stored in `fp16`
- KV cache can be cleared by a hardware button signal
- host still sends prompt and decode embeddings during inference
- host still receives final hidden states and performs final RMSNorm, LM head, softmax, and sampling

## 3. Memory Hierarchy

The design uses four storage classes:

1. SD card or QSPI flash
   - stores the model image used for board bring-up
2. DDR3
   - stores runtime model weights, scales, metadata, and KV cache
3. BRAM and LUTRAM
   - stores active tiles, hidden-state buffers, temporary vectors, and partial sums
4. registers
   - store control state, counters, and small metadata

## 4. Model Initialization Flow

Model initialization is separate from inference.

Initialization flow:

1. FPGA reset is released.
2. Boot logic reads a model image from SD card or QSPI flash.
3. Boot logic copies the model image into DDR3.
4. Runtime metadata is validated.
5. Inference is enabled only after initialization completes successfully.

Important rule:

- Ethernet is a runtime control and embedding interface, not a weight-loading path.

## 5. DDR3 Address Map

Recommended logical regions:

```text
DDR_BASE + 0x0000_0000  runtime metadata region
DDR_BASE + 0x0010_0000  packed weight region
DDR_BASE + 0x0800_0000  scale region
DDR_BASE + 0x0900_0000  KV-cache region
DDR_BASE + 0x0F00_0000  debug or scratch region
```

Notes:

- exact addresses may move later
- the logical separation should remain
- all regions should be aligned for burst access

## 6. Runtime Metadata Region

This region stores small descriptors used by controllers and software tools.

Recommended contents:

- model header
- layer descriptor table
- matrix descriptor table
- quantization format IDs
- reserved debug fields

Suggested model header fields:

- magic
- layout version
- hidden size
- intermediate size
- num layers
- num heads
- num kv heads
- max context supported
- activation format ID
- activation scale format ID
- weight scale format ID

## 7. Packed Weight Region

This region stores all transformer-layer weights used by the FPGA:

- Q
- K
- V
- attention output
- MLP gate
- MLP up
- MLP down
- layer norm weights if they are stored with layer data

The embedding table and LM head are not stored here for FPGA inference.

### 7.1 Ternary coding

Use a simple 2-bit code per weight:

- `00` = `0`
- `01` = `+1`
- `10` = `-1`
- `11` = reserved

Reason:

- this is not density-optimal, but it is the safest first RTL target

### 7.2 Matrix order

Recommended logical matrix order:

- row-major by output channel
- input dimension contiguous within each row

Recommended tile order:

```text
for row_tile in 0 .. num_row_tiles-1
  for col_tile in 0 .. num_col_tiles-1
    emit tile(row_tile, col_tile)
```

### 7.3 Alignment

Recommended minimum alignment:

- matrix headers aligned to 64 bytes
- packed tile payloads aligned to 64 bytes

## 8. Scale Region

Scales are stored separately from packed ternary payloads.

Reasons:

- scales are naturally word-sized numeric values
- they are read differently from dense packed weights
- separation simplifies both packing scripts and RTL address generation

### 8.1 Scale format

Use `fp16` for:

- weight scales
- activation scales

### 8.2 Scale granularity

First-pass assumption:

- one main scale per matrix for weights
- one scale per token vector for activations

Per-channel scale support can be added later if required.

## 9. Activation Layout

## 9.1 Activation representation

Activations are quantized as symmetric signed 8-bit integers.

Representation:

- data vector: `int8`
- scale: `fp16`

Quantization rule:

```text
real_value ~= int8_value * scale
```

Symmetric means:

- zero-point is always zero

### 9.2 Per-token scaling

The activation scale is per token vector.

That means:

- prefill stores one activation scale per token in the tile
- decode stores one activation scale for the current token

### 9.3 Host payload expectation

When the host sends embeddings, the payload should include:

- quantized embedding elements
- the corresponding `fp16` scale for each token vector

When the FPGA returns a hidden state, it should return:

- quantized hidden-state elements
- the corresponding `fp16` scale

## 10. KV-Cache Region

The KV cache lives in DDR3 and is owned by the FPGA runtime.

### 10.1 Logical indexing

Use this logical order:

```text
[layer][kind][position][head][feature]
```

Where:

- `kind = 0` for `K`
- `kind = 1` for `V`

### 10.2 Dimension constants

For the current model:

- `NUM_LAYERS = 24`
- `HIDDEN_SIZE = 1536`
- `NUM_HEADS = 16`
- `HEAD_DIM = 96`

### 10.3 Element format

The exact KV element format is not fully frozen yet.

First-pass recommendation:

- store KV data in the same quantized activation format family
- keep scale metadata explicit if quantized KV is used

This decision must be finalized before implementing `kv_cache_manager`.

### 10.4 Addressing

Define:

- `MAX_CTX`
- `NUM_HEADS`
- `HEAD_DIM`
- `KV_ELEM_BYTES`

Then:

```text
layer_stride    = 2 * MAX_CTX * NUM_HEADS * HEAD_DIM * KV_ELEM_BYTES
kind_stride     =     MAX_CTX * NUM_HEADS * HEAD_DIM * KV_ELEM_BYTES
position_stride =               NUM_HEADS * HEAD_DIM * KV_ELEM_BYTES
head_stride     =                           HEAD_DIM * KV_ELEM_BYTES
feature_stride  =                                      KV_ELEM_BYTES
```

Address:

```text
kv_addr =
  KV_BASE
  + layer_idx    * layer_stride
  + kind_idx     * kind_stride
  + position_idx * position_stride
  + head_idx     * head_stride
  + feature_idx  * feature_stride
```

### 10.5 Clear behavior

The KV cache must support a hardware clear request from a board button.

First-pass behavior:

- button press asserts a `kv_clear_req`
- runtime stops accepting new inference work
- KV-cache manager invalidates the active cache state
- completion is reported through a `kv_clear_done` signal and status register

First-pass implementation note:

- logical invalidation is acceptable for bring-up
- a full physical DDR wipe is not required in the first version

## 11. On-Chip Buffer Allocation

Use explicit logical banks for:

- current hidden state
- next hidden state
- prompt tile embeddings
- Q buffer
- K buffer
- V buffer
- partial sums
- attention score scratch
- decoded weight tile scratch

Recommended placement:

- BRAM for hidden-state and vector buffers
- LUTRAM for small FIFOs and narrow staging
- registers for control metadata

## 12. Alignment and Burst Rules

Recommended first-pass rules:

- keep headers aligned to 64 bytes
- keep packed weight tiles aligned to 64 bytes
- keep scale blocks aligned to 16 bytes
- keep KV row starts aligned to 64 bytes
- prefer simple padding over complicated address math
