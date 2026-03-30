module linear_engine #(
    parameter integer COLS = 16,
    parameter integer ROWS = 4,
    parameter integer PAR_COLS = 4,
    parameter integer IN_W = 8,
    parameter integer ACC_W = 16
) (
    input  wire                    clk_i,
    input  wire                    rst_n_i,
    input  wire                    start_i,
    output wire                    busy_o,
    output wire                    done_o,

    input  wire                    vec_valid_i,
    output wire                    vec_ready_o,
    input  wire signed [IN_W-1:0]  vec_data_i,
    input  wire                    vec_last_i,

    input  wire                    row_valid_i,
    output wire                    row_ready_o,
    input  wire [2*COLS-1:0]       row_weights_i,
    input  wire                    row_last_i,

    output wire                    out_valid_o,
    input  wire                    out_ready_i,
    output wire signed [ACC_W-1:0] out_data_o,
    output wire                    out_last_o
);

localparam [2:0] ST_IDLE     = 3'd0;
localparam [2:0] ST_LOAD_VEC = 3'd1;
localparam [2:0] ST_LOAD_ROW = 3'd2;
localparam [2:0] ST_CALC     = 3'd3;
localparam [2:0] ST_OUT      = 3'd4;

reg [2:0] state_r;
reg done_r;

reg signed [IN_W-1:0] vec_mem [0:COLS-1];
reg [2*COLS-1:0] row_mem [0:ROWS-1];
reg signed [ACC_W-1:0] result_mem [0:ROWS-1];
(* use_dsp = "yes" *) reg signed [ACC_W-1:0] partial_acc_r [0:PAR_COLS-1];

reg [15:0] vec_load_idx_r;
reg [15:0] row_load_idx_r;
reg [15:0] calc_row_idx_r;
reg [15:0] calc_col_idx_r;
reg [15:0] out_row_idx_r;

integer i;
integer lane;

function [1:0] ternary_code_at;
    input [2*COLS-1:0] row_i;
    input integer col_idx_i;
    reg [2*COLS-1:0] shifted_row;
begin
    shifted_row = row_i >> (col_idx_i * 2);
    ternary_code_at = shifted_row[1:0];
end
endfunction

function signed [ACC_W-1:0] signed_vec_ext;
    input signed [IN_W-1:0] vec_i;
begin
    signed_vec_ext = {{(ACC_W-IN_W){vec_i[IN_W-1]}}, vec_i};
end
endfunction

function signed [ACC_W-1:0] lane_contribution;
    input [2*COLS-1:0] row_i;
    input integer base_idx_i;
    input integer lane_i;
    integer col_idx;
    reg signed [ACC_W-1:0] vec_ext;
begin
    col_idx = base_idx_i + lane_i;
    lane_contribution = {ACC_W{1'b0}};

    if (col_idx < COLS) begin
        vec_ext = signed_vec_ext(vec_mem[col_idx]);
        case (ternary_code_at(row_i, col_idx))
            2'b01: lane_contribution = vec_ext;
            2'b10: lane_contribution = -vec_ext;
            default: lane_contribution = {ACC_W{1'b0}};
        endcase
    end
end
endfunction

function signed [ACC_W-1:0] reduced_row_sum;
    input [2*COLS-1:0] row_i;
    input integer base_idx_i;
    integer idx;
    reg signed [ACC_W-1:0] sum;
begin
    sum = {ACC_W{1'b0}};
    for (idx = 0; idx < PAR_COLS; idx = idx + 1) begin
        sum = sum + partial_acc_r[idx] + lane_contribution(row_i, base_idx_i, idx);
    end
    reduced_row_sum = sum;
end
endfunction

wire busy_w;
wire vec_ready_w;
wire row_ready_w;
wire out_valid_w;
wire out_last_w;
wire signed [ACC_W-1:0] out_data_w;

assign busy_w = (state_r != ST_IDLE);
assign vec_ready_w = (state_r == ST_LOAD_VEC);
assign row_ready_w = (state_r == ST_LOAD_ROW);
assign out_valid_w = (state_r == ST_OUT);
assign out_last_w = (state_r == ST_OUT) && (out_row_idx_r == ROWS - 1);
assign out_data_w = result_mem[out_row_idx_r];

assign busy_o = busy_w;
assign done_o = done_r;
assign vec_ready_o = vec_ready_w;
assign row_ready_o = row_ready_w;
assign out_valid_o = out_valid_w;
assign out_last_o = out_last_w;
assign out_data_o = out_data_w;

always @(posedge clk_i or negedge rst_n_i) begin
    if (!rst_n_i) begin
        state_r <= ST_IDLE;
        done_r <= 1'b0;
        vec_load_idx_r <= 16'd0;
        row_load_idx_r <= 16'd0;
        calc_row_idx_r <= 16'd0;
        calc_col_idx_r <= 16'd0;
        out_row_idx_r <= 16'd0;

        for (i = 0; i < COLS; i = i + 1) begin
            vec_mem[i] <= {IN_W{1'b0}};
        end

        for (i = 0; i < ROWS; i = i + 1) begin
            row_mem[i] <= {(2*COLS){1'b0}};
            result_mem[i] <= {ACC_W{1'b0}};
        end

        for (i = 0; i < PAR_COLS; i = i + 1) begin
            partial_acc_r[i] <= {ACC_W{1'b0}};
        end
    end else begin
        done_r <= 1'b0;

        case (state_r)
            ST_IDLE: begin
                vec_load_idx_r <= 16'd0;
                row_load_idx_r <= 16'd0;
                calc_row_idx_r <= 16'd0;
                calc_col_idx_r <= 16'd0;
                out_row_idx_r <= 16'd0;

                for (i = 0; i < PAR_COLS; i = i + 1) begin
                    partial_acc_r[i] <= {ACC_W{1'b0}};
                end

                if (start_i) begin
                    state_r <= ST_LOAD_VEC;
                end
            end

            ST_LOAD_VEC: begin
                if (vec_valid_i && vec_ready_w) begin
                    vec_mem[vec_load_idx_r] <= vec_data_i;

                    if (vec_load_idx_r == COLS - 1) begin
                        vec_load_idx_r <= 16'd0;
                        row_load_idx_r <= 16'd0;
                        state_r <= ST_LOAD_ROW;
                    end else begin
                        vec_load_idx_r <= vec_load_idx_r + 16'd1;
                    end
                end
            end

            ST_LOAD_ROW: begin
                if (row_valid_i && row_ready_w) begin
                    row_mem[row_load_idx_r] <= row_weights_i;

                    if (row_load_idx_r == ROWS - 1) begin
                        row_load_idx_r <= 16'd0;
                        calc_row_idx_r <= 16'd0;
                        calc_col_idx_r <= 16'd0;

                        for (i = 0; i < PAR_COLS; i = i + 1) begin
                            partial_acc_r[i] <= {ACC_W{1'b0}};
                        end

                        state_r <= ST_CALC;
                    end else begin
                        row_load_idx_r <= row_load_idx_r + 16'd1;
                    end
                end
            end

            ST_CALC: begin
                if ((calc_col_idx_r + PAR_COLS) >= COLS) begin
                    result_mem[calc_row_idx_r] <= reduced_row_sum(row_mem[calc_row_idx_r], calc_col_idx_r);
                    calc_col_idx_r <= 16'd0;

                    for (lane = 0; lane < PAR_COLS; lane = lane + 1) begin
                        partial_acc_r[lane] <= {ACC_W{1'b0}};
                    end

                    if (calc_row_idx_r == ROWS - 1) begin
                        calc_row_idx_r <= 16'd0;
                        out_row_idx_r <= 16'd0;
                        state_r <= ST_OUT;
                    end else begin
                        calc_row_idx_r <= calc_row_idx_r + 16'd1;
                    end
                end else begin
                    for (lane = 0; lane < PAR_COLS; lane = lane + 1) begin
                        partial_acc_r[lane] <= partial_acc_r[lane] + lane_contribution(
                            row_mem[calc_row_idx_r],
                            calc_col_idx_r,
                            lane
                        );
                    end

                    calc_col_idx_r <= calc_col_idx_r + PAR_COLS;
                end
            end

            ST_OUT: begin
                if (out_valid_w && out_ready_i) begin
                    if (out_row_idx_r == ROWS - 1) begin
                        state_r <= ST_IDLE;
                        done_r <= 1'b1;
                        out_row_idx_r <= 16'd0;
                    end else begin
                        out_row_idx_r <= out_row_idx_r + 16'd1;
                    end
                end
            end

            default: begin
                state_r <= ST_IDLE;
            end
        endcase
    end
end

endmodule
