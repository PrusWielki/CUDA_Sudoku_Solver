all: sudokugpu sudokucpu

sudokucpu: cpu_sudoku_solver.cpp
	g++ -o sudokucpu cpu_sudoku_solver.cpp
sudokugpu: 
	nvcc  -std=c++11 -o sudokugpu gpu_sudoku_solver.cu
