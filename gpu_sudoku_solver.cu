#define BOARDSIZE 9
#define SQRTBOARD 3
#include <chrono>
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <curand.h>

__device__ bool is_board_valid(const int *board);
// checks rows, columns, smaller boards in relation to a changed cell
__device__ bool is_board_valid(const int *board, int changed)
{
  int r = changed / 9;
  int c = changed % 9;

  if (changed < 0)
  {
    return is_board_valid(board);
  }

  if ((board[changed] < 1) || (board[changed] > 9))
  {
    return false;
  }

  bool visited[BOARDSIZE];
  for (int k = 0; k < BOARDSIZE; k++)
  {
    visited[k] = false;
  }
  // row validation
  for (int i = 0; i < BOARDSIZE; i++)
  {
    int val = board[r * BOARDSIZE + i];

    if (val != 0)
    {
      if (visited[val - 1])
      {
        return false;
      }
      else
      {
        visited[val - 1] = true;
      }
    }
  }

  // collumn validation
  for (int k = 0; k < BOARDSIZE; k++)
  {
    visited[k] = false;
  }
  for (int j = 0; j < BOARDSIZE; j++)
  {
    int val = board[j * BOARDSIZE + c];

    if (val != 0)
    {
      if (visited[val - 1])
      {
        return false;
      }
      else
      {
        visited[val - 1] = true;
      }
    }
  }

  int ridx = r / SQRTBOARD;
  int cidx = c / SQRTBOARD;
  // valdiation of smaller boards
  for (int k = 0; k < BOARDSIZE; k++)
  {
    visited[k] = false;
  }
  for (int i = 0; i < SQRTBOARD; i++)
  {
    for (int j = 0; j < SQRTBOARD; j++)
    {
      int val = board[(ridx * SQRTBOARD + i) * BOARDSIZE + (cidx * SQRTBOARD + j)];

      if (val != 0)
      {
        if (visited[val - 1])
        {
          return false;
        }
        else
        {
          visited[val - 1] = true;
        }
      }
    }
  }
  return true;
}

//

void sudoku_print_cpu(int *board)
{
  for (int i = 0; i < BOARDSIZE; i++)
  {
    for (int j = 0; j < BOARDSIZE; j++)
    {
      if (j % SQRTBOARD == 0)
      {
        ;
      }
      printf("%d ", board[i * BOARDSIZE + j]);
    }
    printf("\n");
  }
}

// checking if given sudoku board is valid, no repeating numbers in each row, column and 3x3 subboards
__device__ bool is_board_valid(const int *board)
{
  bool visited[BOARDSIZE];
  for (int k = 0; k < BOARDSIZE; k++)
  {
    visited[k] = false;
  }

  // First check the validiity of collumns
  for (int i = 0; i < BOARDSIZE; i++)
  {
    for (int k = 0; k < BOARDSIZE; k++)
    {
      visited[k] = false;
    }
    for (int j = 0; j < BOARDSIZE; j++)
    {
      int value = board[BOARDSIZE * j + i];
      if (value != 0)
      {
        if (visited[value - 1])
          return false;
        else
          visited[value - 1] = true;
      }
    }
  }
  // Check the validity of rows
  for (int i = 0; i < BOARDSIZE; i++)
  {
    for (int k = 0; k < BOARDSIZE; k++)
    {
      visited[k] = false;
    }
    for (int j = 0; j < BOARDSIZE; j++)
    {
      int value = board[BOARDSIZE * i + j];
      if (value != 0)
      {
        if (visited[value - 1])
          return false;
        else
          visited[value - 1] = true;
      }
    }
  }
  // Check the validity of sub-boards
  int board_dimension = SQRTBOARD;
  for (int row_index = 0; row_index < board_dimension; row_index++)
  {
    for (int collumn_index = 0; collumn_index < board_dimension; collumn_index++)
    {
      for (int k = 0; k < BOARDSIZE; k++)
      {
        visited[k] = false;
      }
      for (int i = 0; i < board_dimension; i++)
      {
        for (int j = 0; i < board_dimension; j++)
        {
          int value = board[(row_index * board_dimension + i) * BOARDSIZE + (collumn_index * board_dimension + j)];
          if (0 != value)
          {
            if (visited[value - 1])
              return false;
            else
              visited[value - 1] = true;
          }
        }
      }
    }
  }

  return true; // if the function didnt return false on the way down here then the board is valid
}
// a backtrack algorithm trying to find the solution to sudoku board, a kind of brute-force method, checking every possible value in each cell, in range 1-9, stops until the first solution is found
__global__ void backtrack(int *boards_set, int amount_of_boards, int *not_filled_cells, int *amount_nfc, int *solved_board, int *done)
{

  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int *current_board;
  int *current_not_filled_cells;
  int current_amount_nfc;

  while ((index < amount_of_boards) && (0 == *done))
  { // loop continues until we run out of boards to fill in or we find a solution
    current_board = boards_set + index * BOARDSIZE * BOARDSIZE;
    current_not_filled_cells = not_filled_cells + index * BOARDSIZE * BOARDSIZE;
    current_amount_nfc = amount_nfc[index];

    int index_of_empty = 0; // indicates at which empty cell of a board we currently are
    while ((index_of_empty >= 0) && (index_of_empty < current_amount_nfc))
    { // loops until the index goes beyond the board

      current_board[current_not_filled_cells[index_of_empty]]++; // increase the value of a empty cell by one
      if (!is_board_valid(current_board, current_not_filled_cells[index_of_empty]))
      {

        if (current_board[current_not_filled_cells[index_of_empty]] >= 9)
        {
          // if the board is ivalid and the current empty cell's value is 9, then we revert the value of current cell and go back to previous empty cell (which is not empty at the moment) and try a value higher by 1
          current_board[current_not_filled_cells[index_of_empty]] = 0;
          index_of_empty--;
        }
      }
      else
      {
        // if the board is valid, try to fill next empty cell
        index_of_empty++;
      }
    }
    if (index_of_empty == current_amount_nfc)
    {
      // if we reach the end of the board, in this case it means the last empty cell and the board is valid then we found the solution
      *done = 1;
      for (int i = 0; i < BOARDSIZE * BOARDSIZE; i++)
      {
        solved_board[i] = current_board[i];
      }
    }
    // if the solution could not be found fot the current board, take another from the boards generated by bfs
    index += gridDim.x * blockDim.x;
  }
}

// In order to parallelize searching for the valid board, first BFS is run to gather a set of possible boards that later will be passed to DFS backtracking, each thread is going to take care of different board
// thread gets a board and finds empty cell, fills it with a number from range 1-9, checks its validity, if valid saves it to a new variable boards_new, process repeates for other values in that found cell, process repeates for other cells in current board and then finally process repeats for other boards from boards_old
/*As arguments this kernel takes:
boards_old - set of boards from which new boards will be generated
baords_new where the new set of boards will be stored
amount_of_boards - number of boards in boards_old
not_filled_cells - where the indeces of empty spaces of boards_new are stored
amount_nfc - number of empty spaces in each board*/
__global__ void kernel_bfs(int *boards_old, int *boards_new, int amount_of_boards, int *board_index, int *not_filled_cells, int *amount_nfc)
{

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < amount_of_boards)
  { // repeat the process until we run out of boards
    int empty_spot_found = 0;

    for (int i = (index * BOARDSIZE * BOARDSIZE);
         (i < (index * BOARDSIZE * BOARDSIZE) + BOARDSIZE * BOARDSIZE) && (empty_spot_found == 0); i++)
    { // go through all the cells of current board

      if (boards_old[i] == 0)
      {
        empty_spot_found = 1;

        int row = (i - BOARDSIZE * BOARDSIZE * index) / BOARDSIZE;
        int collumn = (i - BOARDSIZE * BOARDSIZE * index) % BOARDSIZE; // row and column of found empty cell

        for (int attempt = 1; attempt <= BOARDSIZE; attempt++)
        { // tries to find all possible values in current cell (1-9)
          int works = 1;
          // check if a chosen value(attempt) is already in a row
          for (int c = 0; c < BOARDSIZE; c++)
          {
            if (boards_old[row * BOARDSIZE + c + BOARDSIZE * BOARDSIZE * index] == attempt)
            {
              works = 0;
            }
          }
          // check if a chosen value(attempt) is already in a collumn
          for (int r = 0; r < BOARDSIZE; r++)
          {
            if (boards_old[r * BOARDSIZE + collumn + BOARDSIZE * BOARDSIZE * index] == attempt)
            {
              works = 0;
            }
          }
          // check the smaller boards
          int n = SQRTBOARD;
          for (int r = n * (row / n); r < n; r++)
          {
            for (int c = n * (collumn / n); c < n; c++)
            {
              if (boards_old[r * BOARDSIZE + c + BOARDSIZE * BOARDSIZE * index] == attempt)
              {
                works = 0;
              }
            }
          }
          if (works == 1)
          {
            // copy the new board

            int next_board_index = atomicAdd(board_index, 1);
            int empty_index = 0;
            for (int r = 0; r < BOARDSIZE; r++)
            {
              for (int c = 0; c < BOARDSIZE; c++)
              {
                boards_new[next_board_index * BOARDSIZE * BOARDSIZE + r * BOARDSIZE + c] = boards_old[index * BOARDSIZE * BOARDSIZE + r * BOARDSIZE + c];
                if (boards_old[index * BOARDSIZE * BOARDSIZE + r * BOARDSIZE + c] == 0 && (r != row || c != collumn))
                {                                                                                               // fiunds empty cells, excluding the one we are filling right now
                  not_filled_cells[empty_index + BOARDSIZE * BOARDSIZE * next_board_index] = r * BOARDSIZE + c; // save the indexes of empty cells, later needed for backtracking

                  empty_index++;
                }
              }
            }
            amount_nfc[next_board_index] = empty_index;
            boards_new[next_board_index * BOARDSIZE * BOARDSIZE + row * BOARDSIZE + collumn] = attempt;
          }
        }
      }
    }

    index += blockDim.x * gridDim.x; // once all possible new boards are found from current board, thread moves on to the next board
  }
}

void load_from_file(char *file_name, int *board)
{
  FILE *a_file = fopen(file_name, "r");

  if (a_file == NULL)
  {
    printf("unable to load file\n");
    return;
  }

  char temp;

  for (int i = 0; i < BOARDSIZE; i++)
  {
    for (int j = 0; j < BOARDSIZE; j++)
    {
      if (!fscanf(a_file, "%c\n", &temp))
      {
        printf("File loading error!\n");
        return;
      }

      if (temp >= '1' && temp <= '9')
      {
        board[i * BOARDSIZE + j] = (int)(temp - '0');
      }
      else
      {
        board[i * BOARDSIZE + j] = 0;
      }
    }
  }
}

int main(int argc, char **argv)
{

  if (argc < 5)
  {
    printf("threads_per_block max_blocks amount_of_bfs_iterations file_with_board\n");
    return 1;
  }

  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  unsigned int maxThreads = deviceProp.maxThreadsPerBlock;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEvent_t startbfs, stopbfs;
  cudaEventCreate(&startbfs);
  cudaEventCreate(&stopbfs);

  cudaEvent_t start_memalloc, stop_memalloc;
  cudaEventCreate(&start_memalloc);
  cudaEventCreate(&stop_memalloc);

  cudaEvent_t start_reading, stop_reading;
  cudaEventCreate(&start_reading);
  cudaEventCreate(&stop_reading);

  cudaEvent_t start_copying, stop_copying;
  cudaEventCreate(&start_copying);
  cudaEventCreate(&stop_copying);

  auto begin = std::chrono::high_resolution_clock::now();
  const unsigned int threadsPerBlock = atoi(argv[1]);
  const unsigned int maxBlocks = atoi(argv[2]);
  char *file_name = argv[4];
  int iterations = atoi(argv[3]);

  int *board = new int[BOARDSIZE * BOARDSIZE];
  cudaEventRecord(start_reading);
  load_from_file(file_name, board);
  cudaEventRecord(stop_reading);

  // stores the boards generated by bfs kernel
  int *boards_new;
  // stores the previous boards, source for bfs kernel
  int *boards_old;
  // stores the location of the empty spaces in the boards
  int *not_filled_cells;
  // stores the number of empty spaces in each board
  int *amount_nfc;
  // stores next generated board
  int *board_index;

  // determines the amount of maximum boards generated from bfs
  const int max_bfs = pow(2, 24);

  cudaEventRecord(start_memalloc);
  cudaMalloc(&not_filled_cells, max_bfs * sizeof(int));
  cudaMalloc(&amount_nfc, (max_bfs / BOARDSIZE * BOARDSIZE + 1) * sizeof(int));
  cudaMalloc(&boards_new, max_bfs * sizeof(int));
  cudaMalloc(&boards_old, max_bfs * sizeof(int));
  cudaMalloc(&board_index, sizeof(int));
  cudaEventRecord(stop_memalloc);
  int total_boards = 1;

  cudaMemset(board_index, 0, sizeof(int));
  cudaMemset(boards_new, 0, max_bfs * sizeof(int));
  cudaMemset(boards_old, 0, max_bfs * sizeof(int));

  cudaEventRecord(start_copying);
  cudaMemcpy(boards_old, board, BOARDSIZE * BOARDSIZE * sizeof(int), cudaMemcpyHostToDevice);
  cudaEventRecord(stop_copying);

  kernel_bfs<<<maxBlocks, threadsPerBlock>>>(boards_old, boards_new, total_boards, board_index, not_filled_cells, amount_nfc);

  // number of boards after a call to BFS kernel
  int host_count;
  // number of iterations to run BFS for
  // altering variable
  //     int iterations = 10;

  // loop through BFS iterations to generate more boards deeper in the tree
  cudaEventRecord(startbfs);
  for (int i = 0; i < iterations; i++)
  {
    cudaMemcpy(&host_count, board_index, sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemset(board_index, 0, sizeof(int));

    if (i % 2 == 0)
    {
      kernel_bfs<<<maxBlocks, threadsPerBlock>>>(boards_new, boards_old, host_count, board_index, not_filled_cells, amount_nfc);
    }
    else
    {
      kernel_bfs<<<maxBlocks, threadsPerBlock>>>(boards_old, boards_new, host_count, board_index, not_filled_cells, amount_nfc);
    }
  }
  cudaEventRecord(stopbfs);
  cudaMemcpy(&host_count, board_index, sizeof(int), cudaMemcpyDeviceToHost);
  printf("Amount of generated boards: %d with amount of iterations: %d\n", host_count, iterations);

  // indicates when a solution is found
  int *dev_finished;
  // stores the solved board
  int *dev_solved;

  cudaMalloc(&dev_finished, sizeof(int));
  cudaMalloc(&dev_solved, BOARDSIZE * BOARDSIZE * sizeof(int));

  cudaMemset(dev_finished, 0, sizeof(int));
  cudaMemcpy(dev_solved, board, BOARDSIZE * BOARDSIZE * sizeof(int), cudaMemcpyHostToDevice);

  if (iterations % 2 == 1)
  {
    // when the number of iterations is odd send boards_old to the dfs
    boards_new = boards_old;
  }

  // int threadsPerBlock2 = (max_bfs < maxThreads*2) ? nextPow2((max_bfs + 1)/ 2) : maxThreads;
  // int maxBlocks2 = (max_bfs + (threadsPerBlock2 * 2 - 1)) / (threadsPerBlock2 * 2);

  cudaEventRecord(start);
  backtrack<<<maxBlocks, threadsPerBlock>>>(boards_new, host_count, not_filled_cells, amount_nfc, dev_solved, dev_finished);
  cudaEventRecord(stop);

  // copy the solved board back
  int *solved = new int[BOARDSIZE * BOARDSIZE];

  memset(solved, 0, BOARDSIZE * BOARDSIZE * sizeof(int));
  // sudoku_print_cpu(solved);
  cudaMemcpy(solved, dev_solved, BOARDSIZE * BOARDSIZE * sizeof(int), cudaMemcpyDeviceToHost);

  sudoku_print_cpu(solved);

  delete[] board;
  delete[] solved;

  cudaFree(not_filled_cells);
  cudaFree(amount_nfc);
  cudaFree(boards_new);
  cudaFree(boards_old);
  cudaFree(board_index);

  cudaFree(dev_finished);
  cudaFree(dev_solved);
  cudaEventSynchronize(stop);

  float millisecondscopying = 0;
  cudaEventElapsedTime(&millisecondscopying, start_copying, stop_copying);
  printf("Data Copying: %.3f seconds.\n", 0.001 * millisecondscopying);
  float millisecondsreading = 0;
  cudaEventElapsedTime(&millisecondsreading, start_reading, stop_reading);
  printf("Data Loading: %.3f seconds.\n", 0.001 * millisecondsreading);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Backtrack: %.3f seconds.\n", 0.001 * milliseconds);
  float millisecondsbfs = 0;
  cudaEventElapsedTime(&millisecondsbfs, startbfs, stopbfs);
  printf("BFS: %.3f seconds.\n", 0.001 * millisecondsbfs);
  float millisecondsmem = 0;
  cudaEventElapsedTime(&millisecondsmem, start_memalloc, stop_memalloc);
  printf("MemAlloc: %.3f seconds.\n", 0.001 * millisecondsmem);
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

  printf("Time measured(total time): %.3f seconds.\n", elapsed.count() * 1e-9);
  return 0;
}