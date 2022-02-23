all: sudokugpu sudokucpu

sudokucpu: main.cpp
	g++ -o sudokucpu main.cpp
sudokugpu: 
	nvcc  -std=c++11 -o sudokugpu a.cu
