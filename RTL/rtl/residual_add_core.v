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

// Residual-add shell.
// Responsibilities:
// - align two vector streams
// - add them elementwise under the chosen numeric policy

endmodule
