/*
  dnn_test.c

  Usage:
  $ ./dnn


*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <endian.h>
#include <string.h>
#include <math.h>
#include "cnn.h"
#include "mem_check.h"

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

/*
  cnn.c
  Convolutional Neural Network in C.
*/


#define DEBUG_LAYER 0


/*  Misc. functions
 */

/* rnd(): uniform random [0.0, 1.0] */
static inline double rnd()
{
    return ((double)rand() / RAND_MAX);
}

/* nrnd(): normal random (std=1.0) */
static inline double nrnd()
{
    return (rnd()+rnd()+rnd()+rnd()-2.0) * 1.724; /* std=1.0 */
}

#if 0
/* sigmoid(x): sigmoid function */
static inline double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}
/* sigmoid_d(y): sigmoid gradient */
static inline double sigmoid_g(double y)
{
    return y * (1.0 - y);
}
#endif

#if 0
/* tanh(x): hyperbolic tangent */
static inline double tanh(double x)
{
    return 2.0 / (1.0 + exp(-2*x)) - 1.0;
}
#endif
/* tanh_g(y): hyperbolic tangent gradient */
static inline double tanh_g(double y)
{
    return 1.0 - y*y;
}

/* relu(x): ReLU */
static inline double relu(double x)
{
    return (0 < x)? x : 0;
}
/* relu_g(y): ReLU gradient */
static inline double relu_g(double y)
{
    return (0 < y)? 1 : 0;
}


/*  Layer
 */

/* Layer_create(lprev, ltype, depth, width, height, nbiases, nweights)
   Creates a Layer object for internal use.
*/
static Layer* Layer_create(
    Layer* lprev, LayerType ltype,
    int depth, int width, int height,
    int nbiases, int nweights)
{
    Layer* self = (Layer*)calloc_c(1, sizeof(Layer), &used_memory);
    if (self == NULL) return NULL;

    self->lprev = lprev;
    self->lnext = NULL;
    self->ltype = ltype;
    self->lid = 0;
    if (lprev != NULL) {
        assert (lprev->lnext == NULL);
        lprev->lnext = self;
        self->lid = lprev->lid+1;
    }
    self->depth = depth;
    self->width = width;
    self->height = height;

    /* Nnodes: number of outputs. */
    self->nnodes = depth * width * height;
    self->outputs = (double*)calloc_c(self->nnodes, sizeof(double), &used_memory);
    self->gradients = (double*)calloc_c(self->nnodes, sizeof(double), &used_memory);
    self->errors = (double*)calloc_c(self->nnodes, sizeof(double), &used_memory);

    self->nbiases = nbiases;
    self->biases = (double*)calloc_c(self->nbiases, sizeof(double), &used_memory);
    self->u_biases = (double*)calloc_c(self->nbiases, sizeof(double), &used_memory);

    self->nweights = nweights;
    self->weights = (double*)calloc_c(self->nweights, sizeof(double), &used_memory);
    self->u_weights = (double*)calloc_c(self->nweights, sizeof(double), &used_memory);

    return self;
}

/* Layer_destroy(self)
   Releases the memory.
*/
void Layer_destroy(Layer* self)
{
    assert (self != NULL);

    free(self->outputs);
    free(self->gradients);
    free(self->errors);

    free(self->biases);
    free(self->u_biases);
    free(self->weights);
    free(self->u_weights);

    free(self);
}

/* Layer_dump(self, fp)
   Shows the debug output.
*/
void Layer_dump(const Layer* self, FILE* fp)
{
    assert (self != NULL);
    Layer* lprev = self->lprev;
    fprintf(fp, "Layer%d ", self->lid);
    if (lprev != NULL) {
        fprintf(fp, "(lprev=Layer%d) ", lprev->lid);
    }
    fprintf(fp, "shape=(%d,%d,%d), nodes=%d\n",
            self->depth, self->width, self->height, self->nnodes);
    {
        int i = 0;
        for (int z = 0; z < self->depth; z++) {
            fprintf(fp, "  %d:\n", z);
            for (int y = 0; y < self->height; y++) {
                fprintf(fp, "    [");
                for (int x = 0; x < self->width; x++) {
                    fprintf(fp, " %.4f", self->outputs[i++]);
                }
                fprintf(fp, "]\n");
            }
        }
    }

    switch (self->ltype) {
    case LAYER_FULL:
        /* Fully connected layer. */
        assert (lprev != NULL);
        fprintf(fp, "  biases = [");
        for (int i = 0; i < self->nnodes; i++) {
            fprintf(fp, " %.4f", self->biases[i]);
        }
        fprintf(fp, "]\n");
        fprintf(fp, "  weights = [\n");
        {
            int k = 0;
            for (int i = 0; i < self->nnodes; i++) {
                fprintf(fp, "    [");
                for (int j = 0; j < lprev->nnodes; j++) {
                    fprintf(fp, " %.4f", self->weights[k++]);
                }
                fprintf(fp, "]\n");
            }
        }
        fprintf(fp, "  ]\n");
        break;

    case LAYER_CONV:
        /* Convolutional layer. */
        assert (lprev != NULL);
        fprintf(fp, "  stride=%d, kernsize=%d\n",
                self->conv.stride, self->conv.kernsize);
        {
            int k = 0;
            for (int z = 0; z < self->depth; z++) {
                fprintf(fp, "  %d: bias=%.4f, weights = [", z, self->biases[z]);
                for (int j = 0; j < lprev->depth * self->conv.kernsize * self->conv.kernsize; j++) {
                    fprintf(fp, " %.4f", self->weights[k++]);
                }
                fprintf(fp, "]\n");
            }
        }
        break;

    default:
        break;
    }
}

/* Layer_feedForw_full(self)
   Performs feed forward updates.
*/
static void Layer_feedForw_full(Layer* self)
{
    assert (self->ltype == LAYER_FULL);
    assert (self->lprev != NULL);
    Layer* lprev = self->lprev;

    int k = 0;
    for (int i = 0; i < self->nnodes; i++) {
        /* Compute Y = (W * X + B) without activation function. */
        double x = self->biases[i];
        for (int j = 0; j < lprev->nnodes; j++) {
            x += (lprev->outputs[j] * self->weights[k++]);
        }
        self->outputs[i] = x;
    }

    if (self->lnext == NULL) {
        /* Last layer - use Softmax. */
        double m = -1;
        for (int i = 0; i < self->nnodes; i++) {
            double x = self->outputs[i];
            if (m < x) { m = x; }
        }
        double t = 0;
        for (int i = 0; i < self->nnodes; i++) {
            double x = self->outputs[i];
            double y = exp(x-m);
            self->outputs[i] = y;
            t += y;
        }
        for (int i = 0; i < self->nnodes; i++) {
            self->outputs[i] /= t;
            /* This isn't right, but set the same value to all the gradients. */
            self->gradients[i] = 1;
        }
    } else {
        /* Otherwise, use Tanh. */
        for (int i = 0; i < self->nnodes; i++) {
            double x = self->outputs[i];
            double y = tanh(x);
            self->outputs[i] = y;
            self->gradients[i] = tanh_g(y);
        }
    }

#if DEBUG_LAYER
    fprintf(stderr, "Layer_feedForw_full(Layer%d):\n", self->lid);
    fprintf(stderr, "  outputs = [");
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(stderr, " %.4f", self->outputs[i]);
    }
    fprintf(stderr, "]\n  gradients = [");
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(stderr, " %.4f", self->gradients[i]);
    }
    fprintf(stderr, "]\n");
#endif
}

static void Layer_feedBack_full(Layer* self)
{
    assert (self->ltype == LAYER_FULL);
    assert (self->lprev != NULL);
    Layer* lprev = self->lprev;

    /* Clear errors. */
    for (int j = 0; j < lprev->nnodes; j++) {
        lprev->errors[j] = 0;
    }

    int k = 0;
    for (int i = 0; i < self->nnodes; i++) {
        /* Computer the weight/bias updates. */
        double dnet = self->errors[i] * self->gradients[i];
        for (int j = 0; j < lprev->nnodes; j++) {
            /* Propagate the errors to the previous layer. */
            lprev->errors[j] += self->weights[k] * dnet;
            self->u_weights[k] += dnet * lprev->outputs[j];
            k++;
        }
        self->u_biases[i] += dnet;
    }

#if DEBUG_LAYER
    fprintf(stderr, "Layer_feedBack_full(Layer%d):\n", self->lid);
    for (int i = 0; i < self->nnodes; i++) {
        double dnet = self->errors[i] * self->gradients[i];
        fprintf(stderr, "  dnet = %.4f, dw = [", dnet);
        for (int j = 0; j < lprev->nnodes; j++) {
            double dw = dnet * lprev->outputs[j];
            fprintf(stderr, " %.4f", dw);
        }
        fprintf(stderr, "]\n");
    }
#endif
}

/* Layer_feedForw_conv(self)
   Performs feed forward updates.
*/
static void Layer_feedForw_conv(Layer* self)
{
    assert (self->ltype == LAYER_CONV);
    assert (self->lprev != NULL);
    Layer* lprev = self->lprev;

    int kernsize = self->conv.kernsize;
    int i = 0;
    for (int z1 = 0; z1 < self->depth; z1++) {
        /* z1: dst matrix */
        /* qbase: kernel matrix base index */
        int qbase = z1 * lprev->depth * kernsize * kernsize;
        for (int y1 = 0; y1 < self->height; y1++) {
            int y0 = self->conv.stride * y1 - self->conv.padding;
            for (int x1 = 0; x1 < self->width; x1++) {
                int x0 = self->conv.stride * x1 - self->conv.padding;
                /* Compute the kernel at (x1,y1) */
                /* (x0,y0): src pixel */
                double v = self->biases[z1];
                for (int z0 = 0; z0 < lprev->depth; z0++) {
                    /* z0: src matrix */
                    /* pbase: src matrix base index */
                    int pbase = z0 * lprev->width * lprev->height;
                    for (int dy = 0; dy < kernsize; dy++) {
                        int y = y0+dy;
                        if (0 <= y && y < lprev->height) {
                            int p = pbase + y*lprev->width;
                            int q = qbase + dy*kernsize;
                            for (int dx = 0; dx < kernsize; dx++) {
                                int x = x0+dx;
                                if (0 <= x && x < lprev->width) {
                                    v += lprev->outputs[p+x] * self->weights[q+dx];
                                }
                            }
                        }
                    }
                }
                /* Apply the activation function. */
                v = relu(v);
                self->outputs[i] = v;
                self->gradients[i] = relu_g(v);
                i++;
            }
        }
    }
    assert (i == self->nnodes);

#if DEBUG_LAYER
    fprintf(stderr, "Layer_feedForw_conv(Layer%d):\n", self->lid);
    fprintf(stderr, "  outputs = [");
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(stderr, " %.4f", self->outputs[i]);
    }
    fprintf(stderr, "]\n  gradients = [");
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(stderr, " %.4f", self->gradients[i]);
    }
    fprintf(stderr, "]\n");
#endif
}

static void Layer_feedBack_conv(Layer* self)
{
    assert (self->ltype == LAYER_CONV);
    assert (self->lprev != NULL);
    Layer* lprev = self->lprev;

    /* Clear errors. */
    for (int j = 0; j < lprev->nnodes; j++) {
        lprev->errors[j] = 0;
    }

    int kernsize = self->conv.kernsize;
    int i = 0;
    for (int z1 = 0; z1 < self->depth; z1++) {
        /* z1: dst matrix */
        /* qbase: kernel matrix base index */
        int qbase = z1 * lprev->depth * kernsize * kernsize;
        for (int y1 = 0; y1 < self->height; y1++) {
            int y0 = self->conv.stride * y1 - self->conv.padding;
            for (int x1 = 0; x1 < self->width; x1++) {
                int x0 = self->conv.stride * x1 - self->conv.padding;
                /* Compute the kernel at (x1,y1) */
                /* (x0,y0): src pixel */
                double dnet = self->errors[i] * self->gradients[i];
                for (int z0 = 0; z0 < lprev->depth; z0++) {
                    /* z0: src matrix */
                    /* pbase: src matrix base index */
                    int pbase = z0 * lprev->width * lprev->height;
                    for (int dy = 0; dy < kernsize; dy++) {
                        int y = y0+dy;
                        if (0 <= y && y < lprev->height) {
                            int p = pbase + y*lprev->width;
                            int q = qbase + dy*kernsize;
                            for (int dx = 0; dx < kernsize; dx++) {
                                int x = x0+dx;
                                if (0 <= x && x < lprev->width) {
                                    lprev->errors[p+x] += self->weights[q+dx] * dnet;
                                    self->u_weights[q+dx] += dnet * lprev->outputs[p+x];
                                }
                            }
                        }
                    }
                }
                self->u_biases[z1] += dnet;
                i++;
            }
        }
    }
    assert (i == self->nnodes);

#if DEBUG_LAYER
    fprintf(stderr, "Layer_feedBack_conv(Layer%d):\n", self->lid);
    for (int i = 0; i < self->nnodes; i++) {
        double dnet = self->errors[i] * self->gradients[i];
        fprintf(stderr, "  dnet=%.4f, dw=[", dnet);
        for (int j = 0; j < lprev->nnodes; j++) {
            double dw = dnet * lprev->outputs[j];
            fprintf(stderr, " %.4f", dw);
        }
        fprintf(stderr, "]\n");
    }
#endif
}

/* Layer_setInputs(self, values)
   Sets the input values.
*/
void Layer_setInputs(Layer* self, const double* values)
{
    assert (self != NULL);
    assert (self->ltype == LAYER_INPUT);
    assert (self->lprev == NULL);

#if DEBUG_LAYER
    fprintf(stderr, "Layer_setInputs(Layer%d): values = [", self->lid);
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(stderr, " %.4f", values[i]);
    }
    fprintf(stderr, "]\n");
#endif

    /* Set the values as the outputs. */
    for (int i = 0; i < self->nnodes; i++) {
        self->outputs[i] = values[i];
    }

    /* Start feed forwarding. */
    Layer* layer = self->lnext;
    while (layer != NULL) {
        switch (layer->ltype) {
        case LAYER_FULL:
            Layer_feedForw_full(layer);
            break;
        case LAYER_CONV:
            Layer_feedForw_conv(layer);
            break;
        default:
            break;
        }
        layer = layer->lnext;
    }
}

/* Layer_getOutputs(self, outputs)
   Gets the output values.
*/
void Layer_getOutputs(const Layer* self, double* outputs)
{
    assert (self != NULL);
    for (int i = 0; i < self->nnodes; i++) {
        outputs[i] = self->outputs[i];
    }
}

/* Layer_getErrorTotal(self)
   Gets the error total.
*/
double Layer_getErrorTotal(const Layer* self)
{
    assert (self != NULL);
    double total = 0;
    for (int i = 0; i < self->nnodes; i++) {
        double e = self->errors[i];
        total += e*e;
    }
    return (total / self->nnodes);
}

/* Layer_learnOutputs(self, values)
   Learns the output values.
*/
void Layer_learnOutputs(Layer* self, const double* values)
{
    assert (self != NULL);
    assert (self->ltype != LAYER_INPUT);
    assert (self->lprev != NULL);
    for (int i = 0; i < self->nnodes; i++) {
        self->errors[i] = (self->outputs[i] - values[i]);
    }

#if DEBUG_LAYER
    fprintf(stderr, "Layer_learnOutputs(Layer%d): errors = [", self->lid);
    for (int i = 0; i < self->nnodes; i++) {
        fprintf(stderr, " %.4f", self->errors[i]);
    }
    fprintf(stderr, "]\n");
#endif

    /* Start backpropagation. */
    Layer* layer = self;
    while (layer != NULL) {
        switch (layer->ltype) {
        case LAYER_FULL:
            Layer_feedBack_full(layer);
            break;
        case LAYER_CONV:
            Layer_feedBack_conv(layer);
            break;
        default:
            break;
        }
        layer = layer->lprev;
    }
}

/* Layer_update(self, rate)
   Updates the weights.
*/
void Layer_update(Layer* self, double rate)
{
    for (int i = 0; i < self->nbiases; i++) {
        self->biases[i] -= rate * self->u_biases[i];
        self->u_biases[i] = 0;
    }
    for (int i = 0; i < self->nweights; i++) {
        self->weights[i] -= rate * self->u_weights[i];
        self->u_weights[i] = 0;
    }
    if (self->lprev != NULL) {
        Layer_update(self->lprev, rate);
    }
}

/* Layer_create_input(depth, width, height)
   Creates an input Layer with size (depth x weight x height).
*/
Layer* Layer_create_input(int depth, int width, int height)
{
    return Layer_create(
        NULL, LAYER_INPUT, depth, width, height, 0, 0);
}

/* Layer_create_full(lprev, nnodes, std)
   Creates a fully-connected Layer.
*/
Layer* Layer_create_full(Layer* lprev, int nnodes, double std)
{
    assert (lprev != NULL);
    Layer* self = Layer_create(
        lprev, LAYER_FULL, nnodes, 1, 1,
        nnodes, nnodes * lprev->nnodes);
    assert (self != NULL);

    for (int i = 0; i < self->nweights; i++) {
        self->weights[i] = std * nrnd();
    }

#if DEBUG_LAYER
    Layer_dump(self, stderr);
#endif
    return self;
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
    double rate = 0.0001;      /* learning rate : 각 반복에서 weight를 얼마나 update할지 */
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
        
        t[0] = f(x, 10);
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
    printf("epoch : %d\n", nepoch);
    printf("\nUsed Memory : %ld bytes\n\n", used_memory_in_bytes(used_memory));

    /* Training finished. */
    //Layer_dump(linput, stdout);
    //Layer_dump(lfull1, stdout);
    //Layer_dump(lfull2, stdout);
    //Layer_dump(lfull3, stdout);
    //Layer_dump(loutput, stdout);


    /* test */
/*    fprintf(stderr, "testing...\n");
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

        int label = 1;
        
        
        int mj = -1;
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