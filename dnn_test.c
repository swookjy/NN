/*
  dnn_test.c

  Usage:
  $ ./dnn


*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
//#include <endian.h>
#include <string.h>
#include "cnn.h"
#include <math.h>
#include "mem_check.h"

#define DEBUG_IDXFILE 0

#define DEBUG_LAYER 0

static int clocks_starts = 0;
static size_t used_memory = 0;

/* f: function to learn */
static double f(double a[], int n)
{
    double max_val = a[0];
    double min_val = a[0];
    int i;
    
    for(i=1; i<n; i++){
        if(a[i] > max_val)
            max_val = a[i];
        if(a[i] < min_val)
            min_val = a[i];
    }
    
    return max_val - min_val;
}

/* main */
int main(int argc, char* argv[])
{
    /* Use a fixed random seed for debugging. */
    srand(0);

    /* Initialize layers. : 10x16x16x16x1 */
    Layer* linput = Layer_create_input(10, 1, 1);
    Layer* lfull1 = Layer_create_full(linput, 16, 0.1);
    Layer* lfull2 = Layer_create_full(lfull1, 16, 0.1);
    Layer* lfull3 = Layer_create_full(lfull2, 16, 0.1);
    Layer* loutput = Layer_create_full(lfull3, 1, 0.1);

    fprintf(stderr, "training...\n");
    double rate = 0.01;      /* learning rate : 각 반복에서 weight를 얼마나 update할지 */
    double etotal = 0;      /* total error : 예측과 실제 결과 간의 차이. 이 값을 최소화해야함*/
    int nepoch = 100000;        /* epoch : 전체 학습 데이터셋을 몇 번 반복해서 학습할지 */
    
    reset_timer(clocks_starts);
    
    /* Run the network */
    for (int i = 0; i < nepoch; i++) {
        double x[10];   //input
        double y[1];    //output
        double t[1];    //target(목표 출력값)
        for (int j=0 ; j<10; j++){
            x[j] = rnd();
        }
        
        t[0] = f1(x, 10);
        Layer_setInputs(linput, x);     //입력 & feed forward to last layer
        Layer_getOutputs(loutput, y);   //output layer에서 출력값을 가져와 y에 저장
        Layer_learnOutputs(loutput, t); //backpropagation(역전파) algorithm : error = z(target)-y(output)
        etotal += Layer_getErrorTotal(loutput);
        
        fprintf(stderr, "i=%d, x=[%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f], y=[%.4f], t=[%.4f], etotal=%.4f\n",
                i, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], y[0], t[0], etotal);
        Layer_update(loutput, rate);    //backpropagation을 통해 구한 weight, bias의 변화율을 이용해 새로운 값을 계산, linput까지 재귀적 수행
    }

    /* Check time & memory */
    show_elapsed_time_in_sec(clocks_starts);
    printf("\nUsed Memory : %ld bytes\n\n", used_memory_in_bytes(used_memory));

    /* Training finished. */
    Layer_dump(linput, stdout);
    Layer_dump(lfull1, stdout);
    Layer_dump(lfull2, stdout);
    Layer_dump(lfull3, stdout);
    Layer_dump(loutput, stdout);


    /* test */
    
    fprintf(stderr, "testing...\n");
    int ncorrect = 0;
    int ntests = 2000;
    for (int i = 0; i < ntests; i++) {
        double x[10];       // INPUT
        double y[1];        // OUTPUT
        for (int j = 0; j < 10; j++) {
            x[j] = rand();
        }
        Layer_setInputs(linput, x);
        Layer_getOutputs(loutput, y);
        
        /* Pick the most probable label. */
/*        int mj = -1;
        for (int j = 0; j < 10; j++) {
            if (mj < 0 || y[mj] < y[j]) {
                mj = j;
            }
        }
        if (mj == label) {
            ncorrect++;
        }
    }
    fprintf(stderr, "ntests=%d, ncorrect=%d\n", ntests, ncorrect);
*/

    Layer_destroy(linput);
    Layer_destroy(lfull1);
    Layer_destroy(lfull2);
    Layer_destroy(lfull3);
    Layer_destroy(loutput);

    return 0;
}