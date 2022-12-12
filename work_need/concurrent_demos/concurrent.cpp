#include <cstdio>
#include "omp.h"   //#incluce ""格式：引用非标准库的头文件，编译器从用户的工作目录开始搜索

int main(int argc, char* argv[])
{
    int nthreads, tid;
    #pragma omp parallel private(nthreads, tid) //{  花括号写在这会报错
    {
        tid = omp_get_thread_num();
        printf("Hello World from OMP thread %d\n", tid);
        if(tid == 0)
            {
            nthreads = omp_get_num_threads();
            printf("Number of threads %d\n", nthreads);
            }
    }
    return 0;
}
