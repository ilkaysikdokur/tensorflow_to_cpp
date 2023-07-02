#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <random>
#include <vector>
#include <cmath>

using namespace std;

//Layer 0: Input
#define IMAGEX 32
#define IMAGEY 32
#define IMAGECHANNEL 3
#define EPOCH 1
#define BATCHSIZE 2
#define CLASSSIZE 10
#define NTRAIN 50000
#define NTEST 10000

#define PRODUCTX0 32
#define PRODUCTY0 32
#define PRODUCTZ0 3
//Layer 1: Conv2D
#define KERNELX1 5
#define KERNELY1 5
#define FILTERAMOUNT1 2
#define PADDING1 0
#define STRIDE1 1
#define FILTERAMOUNTPREV1 3
#define PRODUCTX1 28
#define PRODUCTY1 28
#define PRODUCTZ1 2

//Layer 2: MaxPool2D
#define POOLX2 2
#define POOLY2 2
#define PRODUCTX2 14
#define PRODUCTY2 14
#define PRODUCTZ2 2

//Layer 3: Conv2D
#define KERNELX3 5
#define KERNELY3 5
#define FILTERAMOUNT3 2
#define PADDING3 0
#define STRIDE3 1
#define FILTERAMOUNTPREV3 2
#define PRODUCTX3 10
#define PRODUCTY3 10
#define PRODUCTZ3 2

//Layer 4: MaxPool2D
#define POOLX4 2
#define POOLY4 2
#define PRODUCTX4 5
#define PRODUCTY4 5
#define PRODUCTZ4 2

//Layer 5: Flatten
#define LENGTH5 50

//Layer 6: Dense
#define LENGTH6 64
#define LENGTHPREV6 50

//Layer 7: Dense
#define LENGTH7 10
#define LENGTHPREV7 64


//Optimizer Constants
#define ETA 0.01

//Data Type Definitions
typedef double type_weight;
typedef int type_index;


//Weight and Bias Arrays
extern type_weight F1[KERNELX1][KERNELY1][FILTERAMOUNTPREV1][FILTERAMOUNT1];
extern type_weight BF1[FILTERAMOUNT1];
extern type_weight F3[KERNELX3][KERNELY3][FILTERAMOUNTPREV3][FILTERAMOUNT3];
extern type_weight BF3[FILTERAMOUNT3];
extern type_weight W6[LENGTHPREV6][LENGTH6];
extern type_weight BW6[BATCHSIZE][LENGTH6];
extern type_weight W7[LENGTHPREV7][LENGTH7];
extern type_weight BW7[BATCHSIZE][LENGTH7];


//Other Function Definition


//Training Function Definition
void train(type_weight I[BATCHSIZE][1+IMAGEX*IMAGEY*IMAGECHANNEL], type_weight POUT[BATCHSIZE][CLASSSIZE], bool isInference);