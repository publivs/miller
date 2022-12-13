#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <math.h>

static long long counts  = 10000000;
double step;
#define THREAD_NUMS 10

int main(int argc,const char* argv[] ){
    double pi = 0.0;
    double start,time,sum;
    start = omp_get_wtime();
    step = 1.0/(double)counts;

    int unit = counts/THREAD_NUMS;

    #pragma omp parallel num_threads(THREAD_NUMS)
    {
    int id = omp_get_thread_num();
    int left = id*unit+1;
    int right = (id+1)*unit;
    double x;
    // printf("线程id:%d,数据区间%d~%d\n",id,left,right);

    #pragma omp critical
    {
    printf("线程 %d 在运行...\n",id);
    while (left <= right)
        {
        x = (left +0.5)*step;
        {
        sum+= 4.0/(1+ x*x);
        }
        left++;
        }
    }
    }
    pi += sum*step;
    time = omp_get_wtime() - start;
    printf("Pi  = %f计算耗时%f线程共%d",pi,time,THREAD_NUMS);
}