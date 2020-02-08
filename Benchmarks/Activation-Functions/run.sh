#!/bin/bash
g++ generate_benchmarks.cpp -o Generate_BenchMarks.out
./Generate_BenchMarks.out > result.csv
g++ generate_benchmark.cpp -o Generate_Benchmarks -framework Accelerate   
./Generate_Benchmarks > results_with_blas.csv  
python3 inference.py