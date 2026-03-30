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

// Transformer-layer orchestration shell.
// Responsibilities:
// - sequence RMSNorm, linear projections, RoPE, attention, MLP, and residual steps
// - expose one layer-level interface to decode and prefill controllers

endmodule
