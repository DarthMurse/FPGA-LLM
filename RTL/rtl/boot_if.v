module boot_if (
    input  wire        clk_i,
    input  wire        rst_n_i,
    input  wire        boot_start_i,
    output wire        boot_busy_o,
    output wire        boot_done_o,
    output wire        boot_error_o,

    output wire        ddr_wr_cmd_valid_o,
    input  wire        ddr_wr_cmd_ready_i,
    output wire [31:0] ddr_wr_addr_o,
    output wire [15:0] ddr_wr_len_o,
    output wire        ddr_wr_data_valid_o,
    input  wire        ddr_wr_data_ready_i,
    output wire [63:0] ddr_wr_data_o,
    output wire        ddr_wr_data_last_o
);

// Boot-time model initialization shell.
// Responsibilities:
// - load model image from SD card or QSPI
// - write metadata, weights, and scales into DDR
// - report boot success or failure

endmodule
