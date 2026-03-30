if { $argc != 7 } {
    puts "Usage: vivado -mode batch -source scripts/synth_linear_engine.tcl -tclargs <part> <cols> <rows> <par_cols> <in_w> <acc_w> <out_dir>"
    exit 1
}

set part_name [lindex $argv 0]
set cols [lindex $argv 1]
set rows [lindex $argv 2]
set par_cols [lindex $argv 3]
set in_w [lindex $argv 4]
set acc_w [lindex $argv 5]
set out_dir [lindex $argv 6]

file mkdir $out_dir

read_verilog rtl/linear_engine.v

synth_design \
    -top linear_engine \
    -part $part_name \
    -mode out_of_context \
    -generic "COLS=$cols ROWS=$rows PAR_COLS=$par_cols IN_W=$in_w ACC_W=$acc_w"

create_clock -period 10.000 [get_ports clk_i]

report_utilization -file [file join $out_dir utilization.rpt]
report_timing_summary -file [file join $out_dir timing_summary.rpt]
write_checkpoint -force [file join $out_dir post_synth.dcp]

close_project
exit 0
