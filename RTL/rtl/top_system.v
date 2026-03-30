module top_system (
    input  wire        sys_clk_i,
    input  wire        sys_rst_n_i,
    input  wire        kv_clear_btn_i,

    input  wire        eth_rx_clk_i,
    input  wire [7:0]  eth_rxd_i,
    input  wire        eth_rx_dv_i,
    output wire        eth_tx_clk_o,
    output wire [7:0]  eth_txd_o,
    output wire        eth_tx_en_o,

    output wire [13:0] ddr_addr_o,
    output wire [2:0]  ddr_ba_o,
    inout  wire [15:0] ddr_dq_io,
    inout  wire [1:0]  ddr_dqs_p_io,
    inout  wire [1:0]  ddr_dqs_n_io
);

// Board-level integration shell.
// Owns only board wiring, clocks, resets, and top-level external interfaces.
// Keep model logic out of this module.

endmodule
