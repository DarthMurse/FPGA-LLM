module onchip_buffer_bank (
    input  wire         clk_i,
    input  wire         rst_n_i,

    input  wire         wr_valid_i,
    output wire         wr_ready_o,
    input  wire [7:0]   wr_bank_i,
    input  wire [15:0]  wr_addr_i,
    input  wire [63:0]  wr_data_i,

    input  wire         rd_valid_i,
    output wire         rd_ready_o,
    input  wire [7:0]   rd_bank_i,
    input  wire [15:0]  rd_addr_i,

    output wire         rd_data_valid_o,
    input  wire         rd_data_ready_i,
    output wire [63:0]  rd_data_o
);

// On-chip BRAM/LUTRAM buffer shell.
// Responsibilities:
// - own hidden-state, tile, Q/K/V, and scratch buffers
// - provide a clean logical-bank abstraction to controllers and layer_engine

endmodule
