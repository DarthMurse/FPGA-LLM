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

// Attention engine shell.
// Responsibilities:
// - perform decode or prefill attention score/value processing
// - request KV data and emit attended output vectors

endmodule
