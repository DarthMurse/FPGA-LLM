module ddr_if (
    input  wire         clk_i,
    input  wire         rst_n_i,

    input  wire         rd_cmd_valid_i,
    output wire         rd_cmd_ready_o,
    input  wire [31:0]  rd_addr_i,
    input  wire [15:0]  rd_len_i,
    input  wire [7:0]   rd_tag_i,

    output wire         rd_data_valid_o,
    input  wire         rd_data_ready_i,
    output wire [63:0]  rd_data_o,
    output wire         rd_data_last_o,
    output wire [7:0]   rd_data_tag_o,

    input  wire         wr_cmd_valid_i,
    output wire         wr_cmd_ready_o,
    input  wire [31:0]  wr_addr_i,
    input  wire [15:0]  wr_len_i,
    input  wire [7:0]   wr_tag_i,

    input  wire         wr_data_valid_i,
    output wire         wr_data_ready_o,
    input  wire [63:0]  wr_data_i,
    input  wire         wr_data_last_i
);

// DDR interface shell.
// Responsibilities:
// - translate generic burst read/write requests into board DDR controller traffic
// - isolate controller and datapath modules from MIG-specific details

endmodule
