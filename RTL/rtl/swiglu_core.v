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

// SwiGLU shell.
// Responsibilities:
// - consume gate and up streams
// - apply the chosen activation and elementwise product

endmodule
