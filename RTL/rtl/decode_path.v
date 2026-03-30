module decode_path (
    input  wire         clk_i,
    input  wire         rst_n_i,
    input  wire         start_i,
    input  wire [7:0]   layer_idx_i,
    input  wire [15:0]  position_i,
    input  wire [15:0]  context_used_i,
    output wire         busy_o,
    output wire         done_o,

    input  wire         in_valid_i,
    output wire         in_ready_o,
    input  wire [63:0]  in_data_i,
    input  wire         in_last_i,

    output wire         out_valid_o,
    input  wire         out_ready_i,
    output wire [63:0]  out_data_o,
    output wire         out_last_o
);

// Decode datapath wrapper shell.
// Responsibilities:
// - wrap shared compute blocks for one-token decode execution
// - present one path-level interface to decode_controller

endmodule
