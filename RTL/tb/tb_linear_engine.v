`timescale 1ns/1ps

module tb_linear_engine;

localparam integer COLS  = 8;
localparam integer ROWS  = 3;
localparam integer PAR_COLS = 4;
localparam integer IN_W  = 8;
localparam integer ACC_W = 16;

reg clk;
reg rst_n;
reg start;

reg vec_valid;
wire vec_ready;
reg signed [IN_W-1:0] vec_data;
reg vec_last;

reg row_valid;
wire row_ready;
reg [2*COLS-1:0] row_weights;
reg row_last;

wire out_valid;
reg out_ready;
wire signed [ACC_W-1:0] out_data;
wire out_last;
wire busy;
wire done;

reg signed [IN_W-1:0] test_vec [0:COLS-1];
reg [1:0] test_weights [0:ROWS-1][0:COLS-1];
reg signed [ACC_W-1:0] expected [0:ROWS-1];
reg signed [ACC_W-1:0] observed [0:ROWS-1];

integer i;
integer r;
integer c;
integer out_count;
integer hold_cycles;

linear_engine #(
    .COLS(COLS),
    .ROWS(ROWS),
    .PAR_COLS(PAR_COLS),
    .IN_W(IN_W),
    .ACC_W(ACC_W)
) dut (
    .clk_i(clk),
    .rst_n_i(rst_n),
    .start_i(start),
    .busy_o(busy),
    .done_o(done),
    .vec_valid_i(vec_valid),
    .vec_ready_o(vec_ready),
    .vec_data_i(vec_data),
    .vec_last_i(vec_last),
    .row_valid_i(row_valid),
    .row_ready_o(row_ready),
    .row_weights_i(row_weights),
    .row_last_i(row_last),
    .out_valid_o(out_valid),
    .out_ready_i(out_ready),
    .out_data_o(out_data),
    .out_last_o(out_last)
);

function [2*COLS-1:0] pack_row;
    input integer row_idx;
    integer idx;
begin
    pack_row = {2*COLS{1'b0}};
    for (idx = 0; idx < COLS; idx = idx + 1) begin
        pack_row = pack_row | (test_weights[row_idx][idx] << (idx * 2));
    end
end
endfunction

function signed [ACC_W-1:0] golden_row_sum;
    input integer row_idx;
    integer idx;
    reg signed [ACC_W-1:0] acc;
    reg signed [ACC_W-1:0] vec_ext;
begin
    acc = {ACC_W{1'b0}};
    for (idx = 0; idx < COLS; idx = idx + 1) begin
        vec_ext = {{(ACC_W-IN_W){test_vec[idx][IN_W-1]}}, test_vec[idx]};
        case (test_weights[row_idx][idx])
            2'b01: acc = acc + vec_ext;
            2'b10: acc = acc - vec_ext;
            default: acc = acc;
        endcase
    end
    golden_row_sum = acc;
end
endfunction

always #5 clk = ~clk;

initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    start = 1'b0;
    vec_valid = 1'b0;
    vec_data = '0;
    vec_last = 1'b0;
    row_valid = 1'b0;
    row_weights = '0;
    row_last = 1'b0;
    out_ready = 1'b1;
    out_count = 0;

    test_vec[0] = 8'sd3;
    test_vec[1] = -8'sd2;
    test_vec[2] = 8'sd5;
    test_vec[3] = -8'sd7;
    test_vec[4] = 8'sd1;
    test_vec[5] = 8'sd4;
    test_vec[6] = -8'sd6;
    test_vec[7] = 8'sd2;

    test_weights[0][0] = 2'b01;
    test_weights[0][1] = 2'b00;
    test_weights[0][2] = 2'b10;
    test_weights[0][3] = 2'b01;
    test_weights[0][4] = 2'b00;
    test_weights[0][5] = 2'b10;
    test_weights[0][6] = 2'b01;
    test_weights[0][7] = 2'b00;

    test_weights[1][0] = 2'b10;
    test_weights[1][1] = 2'b01;
    test_weights[1][2] = 2'b01;
    test_weights[1][3] = 2'b00;
    test_weights[1][4] = 2'b10;
    test_weights[1][5] = 2'b00;
    test_weights[1][6] = 2'b01;
    test_weights[1][7] = 2'b10;

    test_weights[2][0] = 2'b00;
    test_weights[2][1] = 2'b10;
    test_weights[2][2] = 2'b00;
    test_weights[2][3] = 2'b01;
    test_weights[2][4] = 2'b01;
    test_weights[2][5] = 2'b01;
    test_weights[2][6] = 2'b10;
    test_weights[2][7] = 2'b00;

    for (r = 0; r < ROWS; r = r + 1) begin
        expected[r] = golden_row_sum(r);
        observed[r] = {ACC_W{1'b0}};
    end

    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    @(posedge clk);

    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;

    @(posedge clk);
    if (!busy) begin
        $display("ERROR: busy_o did not assert after start");
        $finish;
    end

    fork
        begin : drive_vector
            for (i = 0; i < COLS; i = i + 1) begin
                while (!vec_ready) @(posedge clk);
                vec_valid <= 1'b1;
                vec_data <= test_vec[i];
                vec_last <= (i == COLS - 1);
                @(posedge clk);
            end
            vec_valid <= 1'b0;
            vec_last <= 1'b0;
            vec_data <= '0;
        end

        begin : drive_rows
            for (r = 0; r < ROWS; r = r + 1) begin
                while (!row_ready) @(posedge clk);
                row_valid <= 1'b1;
                row_weights <= pack_row(r);
                row_last <= (r == ROWS - 1);
                @(posedge clk);
            end
            row_valid <= 1'b0;
            row_last <= 1'b0;
            row_weights <= '0;
        end

        begin : inject_illegal_start
            while (!busy) @(posedge clk);
            @(posedge clk);
            start <= 1'b1;
            @(posedge clk);
            start <= 1'b0;
        end

        begin : apply_backpressure
            while (!out_valid) @(posedge clk);
            out_ready <= 1'b0;
            hold_cycles = 0;
            while (hold_cycles < 3) begin
                hold_cycles = hold_cycles + 1;
                @(posedge clk);
                if (!out_valid) begin
                    $display("ERROR: out_valid_o dropped during backpressure");
                    $finish;
                end
            end
            out_ready <= 1'b1;
        end
    join

    wait (done);
    @(posedge clk);

    if (busy) begin
        $display("ERROR: busy_o did not deassert after completion");
        $finish;
    end

    if (out_count != ROWS) begin
        $display("ERROR: expected %0d outputs, observed %0d", ROWS, out_count);
        $finish;
    end

    for (r = 0; r < ROWS; r = r + 1) begin
        if (observed[r] !== expected[r]) begin
            $display("ERROR: row %0d mismatch, expected %0d observed %0d", r, expected[r], observed[r]);
            $finish;
        end
    end

    $display("PASS: linear_engine produced expected ternary matrix-vector results");
    $finish;
end

always @(posedge clk) begin
    if (rst_n && out_valid && out_ready) begin
        if (out_count >= ROWS) begin
            $display("ERROR: observed too many outputs");
            $finish;
        end

        observed[out_count] <= out_data;

        if ((out_count == ROWS - 1) && !out_last) begin
            $display("ERROR: out_last_o was not asserted on final output");
            $finish;
        end

        if ((out_count != ROWS - 1) && out_last) begin
            $display("ERROR: out_last_o asserted too early");
            $finish;
        end

        out_count <= out_count + 1;
    end
end

endmodule
