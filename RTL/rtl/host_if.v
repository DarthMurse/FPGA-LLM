module host_if (
    input  wire         clk_i,
    input  wire         rst_n_i,

    output wire         cmd_valid_o,
    input  wire         cmd_ready_i,
    output wire [7:0]   cmd_opcode_o,
    output wire [31:0]  cmd_session_id_o,
    output wire [31:0]  cmd_sequence_id_o,
    output wire [31:0]  cmd_arg0_o,
    output wire [31:0]  cmd_arg1_o,
    output wire [31:0]  cmd_arg2_o,

    output wire         data_out_valid_o,
    input  wire         data_out_ready_i,
    output wire [63:0]  data_out_o,
    output wire         data_out_last_o,

    input  wire         data_in_valid_i,
    output wire         data_in_ready_o,
    input  wire [63:0]  data_in_i,
    input  wire         data_in_last_i,

    input  wire         rsp_valid_i,
    output wire         rsp_ready_o,
    input  wire [7:0]   rsp_opcode_i,
    input  wire [31:0]  rsp_session_id_i,
    input  wire [31:0]  rsp_sequence_id_i,
    input  wire [31:0]  rsp_status_i
);

// Host protocol termination shell.
// Responsibilities:
// - parse host packets into coarse commands and payload streams
// - format FPGA responses back to the host
// - buffer embedding input and hidden-state output payloads

endmodule
