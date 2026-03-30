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

// RMSNorm shell.
// Responsibilities:
// - compute per-vector RMS statistics
// - apply learned weight and emit normalized output

endmodule
