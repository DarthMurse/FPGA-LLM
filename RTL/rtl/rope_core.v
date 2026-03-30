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

// RoPE shell.
// Responsibilities:
// - apply rotary position embedding to Q and K streams
// - preserve stream ordering and head structure

endmodule
