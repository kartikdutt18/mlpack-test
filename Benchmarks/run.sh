#!/bin/bash
g++ generate_benchmarks.cpp -o Generate_BenchMarks.out
./Generate_BenchMarks.out > result.csv
sudo rm ./Generate_BenchMarks.out
python3 inference.py