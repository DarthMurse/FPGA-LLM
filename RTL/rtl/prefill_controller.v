module prefill_controller (
    input  wire         clk_i,
    input  wire         rst_n_i,

    input  wire         cmd_valid_i,
    output wire         cmd_ready_o,
    input  wire [31:0]  cmd_session_id_i,
    input  wire [31:0]  cmd_sequence_id_i,
    input  wire [31:0]  cmd_token_start_i,
    input  wire [31:0]  cmd_token_count_i,
    input  wire         cmd_is_last_tile_i,

    input  wire         embed_valid_i,
    output wire         embed_ready_o,
    input  wire [63:0]  embed_data_i,
    input  wire         embed_last_i,

    output wire         path_start_o,
    input  wire         path_busy_i,
    input  wire         path_done_i,

    output wire         result_valid_o,
    input  wire         result_ready_i,
    output wire [63:0]  result_data_o,
    output wire         result_last_o,

    output wire         rsp_valid_o,
    input  wire         rsp_ready_i,
    output wire [7:0]   rsp_opcode_o,
    output wire [31:0]  rsp_session_id_o,
    output wire [31:0]  rsp_sequence_id_o,
    output wire [31:0]  rsp_status_o
);

// Prefill control shell.
// Responsibilities:
// - accept prompt tile commands and embedding payloads
// - sequence prefill execution across layers and tiles
// - return final hidden-state payload or ACK response

endmodule
