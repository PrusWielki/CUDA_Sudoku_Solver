#include <iostream>
#include <chrono>
using namespace std;
#define BOARDSIZE 9
#define SQRTBOARD 3


//checks if the board with a new number in a given row, column would be valid
bool is_board_valid(int board[BOARDSIZE][BOARDSIZE], int row, int col, int num)
{

//row validation
    for (int i=0; i<BOARDSIZE; i++)
        if (board[row][i] == num)
            return false;

//collumn validation
    for (int i =0; i <BOARDSIZE; i++)
        if (board[i][col] == num)
            return false;

//small boards validation
    int startRow=row-row%SQRTBOARD;
    int startCol =col-col%SQRTBOARD;

    for (int i=0; i<SQRTBOARD; i++)
        for (int j =0; j<SQRTBOARD; j++)
            if (board[i+startRow][j+startCol] == num)
                return false;

    return true;
}

//backtrack algorithm to solve sudoku
bool backtrack(int board[BOARDSIZE][BOARDSIZE], int row, int col)
{
    if (row == BOARDSIZE - 1 && col == BOARDSIZE)
        return true;

    if (col==BOARDSIZE) {
        row++;
        col=0;
    }

    if (board[row][col] > 0)
        return backtrack(board, row, col + 1);

    for (int num =1;num<=BOARDSIZE;num++)
    {

        if (is_board_valid(board, row, col, num))
        {

        board[row][col]=num;
            if (backtrack(board,row,col+1))
                return true;
        }
        board[row][col] = 0;
    }
    return false;
}
void print_cpu(int arr[BOARDSIZE][BOARDSIZE])
{
    for (int i=0;i <BOARDSIZE; i++)
    {
        for (int j=0; j<BOARDSIZE;j++)
            cout << arr[i][j];
        cout << endl;
    }
}
void load_from_file(char *file_name, int board[BOARDSIZE][BOARDSIZE]) {
    FILE * a_file = fopen(file_name, "r");

    if (a_file == NULL) {
        printf("unable to load file\n");
        return;
    }

    char temp;

    for (int i = 0; i < BOARDSIZE; i++) {
        for (int j = 0; j < BOARDSIZE; j++) {
            if (!fscanf(a_file, "%c\n", &temp)) {
                printf("File loading error!\n");
                return;
            }

            if (temp >= '1' && temp <= '9') {
                board[i][j] = (int) (temp - '0');
            } else {
                board[i][j] = 0;
            }
        }
    }
}
int main(int argc, char**argv)
{
    auto begin = std::chrono::high_resolution_clock::now();
char* file_name = argv[1];
int board[BOARDSIZE][BOARDSIZE];
load_from_file(file_name,board);
print_cpu(board);
cout<<endl;
cout<<endl;
    if (backtrack(board, 0, 0))
        print_cpu(board);
    else
        return 1;
   auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);


    printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);
    return 0;
}
