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

// KV-cache manager shell.
// Responsibilities:
// - translate logical KV indices into DDR addresses
// - handle KV write and read request routing
// - implement logical cache clear behavior

endmodule
