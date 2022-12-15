#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <vector>

#define THREAD_NUMS 20

double func_1(double x){
        double y;
        y = x*x;
        return y;
}

int main(){
    std::vector<double> input_arr;
    std::vector<double> op_arr;

    for(int i = 0;i<100000;i++){
        input_arr[i] = i;
    }

    omp_set_num_threads(THREAD_NUMS);
    #pragma parallel for shared(op_arr)
    {
    for(int i =0;i < input_arr.size();i++)
        {
        double y_i;
        std::vector<double> inside_arr;
        y_i = func_1((double)input_arr[i]);
        inside_arr.push_back(y_i);
        };
    #pragma omp atomic
    op_arr.
    }

    printf("finish");
}

