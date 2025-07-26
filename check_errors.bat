@echo off
cargo check > build_output.txt 2>&1
type build_output.txt | findstr /C:"error[E"