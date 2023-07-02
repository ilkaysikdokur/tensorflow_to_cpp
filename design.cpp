#include "design.hpp"


//Weight and Bias Arrays
type_weight F1[KERNELX1][KERNELY1][FILTERAMOUNTPREV1][FILTERAMOUNT1] = {0};
type_weight BF1[FILTERAMOUNT1] = {0};
type_weight F3[KERNELX3][KERNELY3][FILTERAMOUNTPREV3][FILTERAMOUNT3] = {0};
type_weight BF3[FILTERAMOUNT3] = {0};
type_weight W6[LENGTHPREV6][LENGTH6] = {0};
type_weight BW6[BATCHSIZE][LENGTH6] = {0};
type_weight W7[LENGTHPREV7][LENGTH7] = {0};
type_weight BW7[BATCHSIZE][LENGTH7] = {0};


//Training Function
void train(type_weight I[BATCHSIZE][1+IMAGEX*IMAGEY*IMAGECHANNEL], type_weight POUT[BATCHSIZE][CLASSSIZE], bool isInference){

	//Indices
	type_index i, j, k, l, m, n, p;


	//Products
	type_weight P1[BATCHSIZE][PRODUCTX1][PRODUCTY1][PRODUCTZ1] = {0};
	type_weight P2[BATCHSIZE][PRODUCTX2][PRODUCTY2][PRODUCTZ2] = {0};
	type_index IndexP2[BATCHSIZE][PRODUCTX2][PRODUCTY2][PRODUCTZ2][2] = {0};
	type_weight P3[BATCHSIZE][PRODUCTX3][PRODUCTY3][PRODUCTZ3] = {0};
	type_weight P4[BATCHSIZE][PRODUCTX4][PRODUCTY4][PRODUCTZ4] = {0};
	type_index IndexP4[BATCHSIZE][PRODUCTX4][PRODUCTY4][PRODUCTZ4][2] = {0};
	type_weight P5[BATCHSIZE][LENGTH5] = {0};
	type_weight P6[BATCHSIZE][LENGTH6] = {0};
	type_weight P7[BATCHSIZE][LENGTH7] = {0};
	type_weight IArr[BATCHSIZE][1+PRODUCTX0*PRODUCTY0*PRODUCTZ0] = {0};

	for (i=0 ; i<BATCHSIZE ; ++i){
		for (j=0 ; j<1+PRODUCTX0*PRODUCTY0*PRODUCTZ0 ; ++j){
			IArr[i][j] = I[i][j];
		}
	}


	//Actual Class Array
	type_index outActual[BATCHSIZE][CLASSSIZE] = {0};
	for(i=0 ; i<BATCHSIZE ; ++i){
		for(j=0 ; j<CLASSSIZE ; ++j){
			if (IArr[i][0] == j) outActual[i][j] = 1;
		}
	}



	//Forward Pass

	//Layer 1: Conv2D
	for (i=0 ; i<BATCHSIZE ; ++i){
		for (j=0 ; j<FILTERAMOUNTPREV1 ; ++j){
			for (k=0 ; k<PRODUCTX1 ; ++k){
				for (l=0 ; l<PRODUCTY1 ; ++l){
					for (m=0 ; m<PRODUCTZ1 ; ++m){
						for (n=0 ; n<KERNELX1 ; ++n){
							for (p=0 ; p<KERNELY1 ; ++p){
								if ((isInference == true && i==0)||(isInference == false)){
									P1[i][k][l][m] += IArr[i][1 + j*IMAGEX*IMAGEY + (k-PADDING1+n)*IMAGEX + (l-PADDING1+p)] * F1[n][p][j][m];
									if (j==FILTERAMOUNTPREV1-1 &&  n==KERNELX1-1 && p==KERNELY1-1)
										P1[i][k][l][m] += BF1[m];
									if (j==FILTERAMOUNTPREV1-1 && n==KERNELX1-1 && p==KERNELY1-1 && P1[i][k][l][m]<0)
										P1[i][k][l][m] = 0;
								}
							}
						}
					}
				}
			}
		}
	}
	//Layer 2: MaxPool2D
	for (i=0 ; i<BATCHSIZE ; ++i){
		for (j=0 ; j<PRODUCTX2 ; ++j){
			for (k=0 ; k<PRODUCTY2 ; ++k){
				for (l=0 ; l<PRODUCTZ2 ; ++l){
					for (n=0 ; n<POOLX2 ; ++n){
						for (p=0 ; p<POOLY2 ; ++p){
							if ((isInference == true && i==0)||(isInference == false)){
								if (P2[i][j][k][l] < P1[i][j*POOLX2+n][k*POOLY2+p][l]){
									P2[i][j][k][l] = P1[i][j*POOLX2+n][k*POOLY2+p][l];
									IndexP2[i][j][k][l][0] = j*POOLX2+n;
									IndexP2[i][j][k][l][1] = k*POOLY2+p;
								}
							}
						}
					}
				}
			}
		}
	}
	//Layer 3: Conv2D
	for (i=0 ; i<BATCHSIZE ; ++i){
		for (j=0 ; j<FILTERAMOUNTPREV3 ; ++j){
			for (k=0 ; k<PRODUCTX3 ; ++k){
				for (l=0 ; l<PRODUCTY3 ; ++l){
					for (m=0 ; m<PRODUCTZ3 ; ++m){
						for (n=0 ; n<KERNELX3 ; ++n){
							for (p=0 ; p<KERNELY3 ; ++p){
								if ((isInference == true && i==0)||(isInference == false)){
									P3[i][k][l][m] += P2[i][k-PADDING3+n][l-PADDING3+p][m] * F3[n][p][j][m];
									if (j==FILTERAMOUNTPREV3-1 && n==KERNELX3-1 && p==KERNELY3-1)
										P3[i][k][l][m] += BF3[m];
									if (j==FILTERAMOUNTPREV3-1 && n==KERNELX3-1 && p==KERNELY3-1 && P3[i][k][l][m]<0)
										P3[i][k][l][m] = 0;
								}
							}
						}
					}
				}
			}
		}
	}
	//Layer 4: MaxPool2D
	for (i=0 ; i<BATCHSIZE ; ++i){
		for (j=0 ; j<PRODUCTX4 ; ++j){
			for (k=0 ; k<PRODUCTY4 ; ++k){
				for (l=0 ; l<PRODUCTZ4 ; ++l){
					for (n=0 ; n<POOLX4 ; ++n){
						for (p=0 ; p<POOLY4 ; ++p){
							if ((isInference == true && i==0)||(isInference == false)){
								if (P4[i][j][k][l] < P3[i][j*POOLX4+n][k*POOLY4+p][l]){
									P4[i][j][k][l] = P3[i][j*POOLX4+n][k*POOLY4+p][l];
									IndexP4[i][j][k][l][0] = j*POOLX4+n;
									IndexP4[i][j][k][l][1] = k*POOLY4+p;
								}
							}
						}
					}
				}
			}
		}
	}
	//Layer 5: Flatten
	for (i=0 ; i<BATCHSIZE ; ++i){
		for (j=0 ; j<PRODUCTX4 ; ++j){
			for (k=0 ; k<PRODUCTY4 ; ++k){
				for (l=0 ; l<PRODUCTZ4 ; ++l){
				if ((isInference == true && i==0)||(isInference == false)){
						P5[i][l*PRODUCTY4*PRODUCTX4 + j*PRODUCTX4 + k] = P4[i][j][k][l];
					}
				}
			}
		}
	}
	//Layer 6: Dense
	for (i=0 ; i<BATCHSIZE ; ++i){
		for (j=0 ; j<LENGTH6 ; ++j){
			for (k=0 ; k<LENGTHPREV6 ; ++k){
				if ((isInference == true && i==0)||(isInference == false)){
					P6[i][j] += P5[i][k] * W6[k][j];
					if (k==LENGTHPREV6-1)
						P6[i][j] += BW6[i][j];
					if (k==LENGTHPREV6-1 && P6[i][j]<0)
						P6[i][j] = 0;
				}
			}
		}
	}
	//Layer 7: Dense
	for (i=0 ; i<BATCHSIZE ; ++i){
		for (j=0 ; j<LENGTH7 ; ++j){
			for (k=0 ; k<LENGTHPREV7 ; ++k){
				if ((isInference == true && i==0)||(isInference == false)){
					P7[i][j] += P6[i][k] * W7[k][j];
					if (k==LENGTHPREV7-1)
						P7[i][j] += BW7[i][j];
					if (k==LENGTHPREV7-1)
						POUT[i][j] = P7[i][j];
				}
			}
		}
	}


	if (isInference == true) return;

	//Backward Pass

	//Backward Dense and Flatten Storage Arrays
	type_weight BDFSA7[BATCHSIZE][LENGTH7] = {0};
	type_weight BDFSA6[BATCHSIZE][LENGTH6] = {0};
	type_weight BDFSA5[BATCHSIZE][LENGTH5] = {0};

	//Backward Dense and Flatten Loss Gradient Arrays
	type_weight BDFLGA7[LENGTHPREV7][LENGTH7] = {0};
	type_weight BDFLGA6[LENGTHPREV6][LENGTH6] = {0};

	//Backward Conv2D Kernel and Bias Weight Loss Gradient Arrays
	type_weight BKLGA3[KERNELX3][KERNELY3][FILTERAMOUNTPREV3][FILTERAMOUNT3] = {0};
	type_weight BKBLGA3[FILTERAMOUNT3] = {0};
	type_weight BKLGA1[KERNELX1][KERNELY1][FILTERAMOUNTPREV1][FILTERAMOUNT1] = {0};
	type_weight BKBLGA1[FILTERAMOUNT1] = {0};

	//Backward Conv2D MaxPool2D AvgPool2D Product Loss Gradient Arrays
	type_weight BCMLGA4[BATCHSIZE][PRODUCTX4][PRODUCTY4][PRODUCTZ4] = {0};
	type_weight BCMLGA3[BATCHSIZE][PRODUCTX3][PRODUCTY3][PRODUCTZ3] = {0};
	type_weight BCMLGA2[BATCHSIZE][PRODUCTX2][PRODUCTY2][PRODUCTZ2] = {0};
	type_weight BCMLGA1[BATCHSIZE][PRODUCTX1][PRODUCTY1][PRODUCTZ1] = {0};

	//Backward Loss Gradient Calculation
	for (i=0 ; i<BATCHSIZE ; ++i){
		for (j=0 ; j<LENGTH7 ; ++j){
			BDFSA7[i][j] = P7[i][j] - outActual[i][j];
		}
	}
	for (k=0 ; k<LENGTH7 ; ++k){
		for (j=0 ; j<LENGTH6 ; ++j){
			for (i=0 ; i<BATCHSIZE ; ++i){
				BDFSA6[i][j] += BDFSA7[i][k]*W7[j][k];
				BDFLGA7[j][k] += P6[i][j]*BDFSA7[i][k];
				if (i == BATCHSIZE-1){
					W7[j][k] -= ETA*BDFLGA7[j][k]/BATCHSIZE;
				}
				if (j == LENGTH6-1){
					BW7[i][k] -= ETA*BDFSA7[i][k]/BATCHSIZE;
				}
			}
		}
	}
	for (k=0 ; k<LENGTH6 ; ++k){
		for (j=0 ; j<LENGTH5 ; ++j){
			for (i=0 ; i<BATCHSIZE ; ++i){
				BDFSA5[i][j] += BDFSA6[i][k]*W6[j][k];
				BDFLGA6[j][k] += P5[i][j]*BDFSA6[i][k];
				if (i == BATCHSIZE-1){
					W6[j][k] -= ETA*BDFLGA6[j][k]/BATCHSIZE;
				}
				if (j == LENGTH5-1){
					BW6[i][k] -= ETA*BDFSA6[i][k]/BATCHSIZE;
				}
			}
		}
	}
	for (i=0 ; i<BATCHSIZE ; ++i){
		for (j=0 ; j<PRODUCTZ4 ; ++j){
			for (k=0 ; k<PRODUCTX4 ; ++k){
				for (l=0 ; l<PRODUCTY4 ; ++l){
					BCMLGA4[i][k][l][j] = BDFSA5[i][j*PRODUCTX4*PRODUCTY4+k*PRODUCTX4+l];
				}
			}
		}
	}
	for (i=0 ; i<BATCHSIZE ; ++i){
		for (j=0 ; j<PRODUCTZ4 ; ++j){
			for (k=0 ; k<PRODUCTX4 ; ++k){
				for (l=0 ; l<PRODUCTY4 ; ++l){
					BCMLGA3[i][IndexP4[i][k][l][j][0]][IndexP4[i][k][l][j][1]][j] = BCMLGA4[i][k][l][j];
				}
			}
		}
	}
	for (i=0 ; i<BATCHSIZE ; ++i){
		for (j=0 ; j<PRODUCTZ2 ; ++j){
			for (k=0 ; k<KERNELX3 ; ++k){
				for (l=0 ; l<KERNELY3 ; ++l){
					for (m=0 ; m<PRODUCTZ3 ; ++m){
						for (n=0 ; n<PRODUCTX3 ; ++n){
							for (p=0 ; p<PRODUCTY3 ; ++p){
								BKLGA3[k][l][j][m] += P3[i][k-PADDING3+n][l-PADDING3+p][j] * BCMLGA3[i][n][p][m];
								if (j==PRODUCTZ2-1 && k==KERNELX3-1 && l==KERNELY3-1)
									BKBLGA3[m] += BCMLGA3[i][n][p][m];
								if (i==BATCHSIZE-1 &&  n==PRODUCTX3-1 && p==PRODUCTY3-1){
									F3[k][l][j][m] -= ETA*BKLGA3[k][l][j][m]/BATCHSIZE;
									if (i==BATCHSIZE-1 &&  n==PRODUCTX3-1 && p==PRODUCTY3-1 && l==KERNELY3-1 && k==KERNELX3-1 && j==PRODUCTZ2-1){
										BF3[m] -= ETA*BKBLGA3[m]/BATCHSIZE;
									}
								}
							}
						}
					}
				}
			}
		}
	}
	for (i=0 ; i<BATCHSIZE ; ++i){
		for (j=0 ; j<PRODUCTZ2 ; ++j){
			for (k=-1 ; k<PRODUCTX2+1 ; ++k){
				for (l=-1 ; l<PRODUCTY2+1 ; ++l){
					for (m=0 ; m<FILTERAMOUNT3 ; ++m){
						for (n=KERNELX3-1 ; n>=0 ; --n){
							for (p=KERNELY3-1 ; p>=0 ; --p){
								if (k>=0 && k<PRODUCTX2 && l>=0 && l<PRODUCTY2)
									BCMLGA2[i][k][l][j] += BCMLGA3[i][k-PADDING3+n][l-PADDING3+p][m] * F3[n][p][j][m];
							}
						}
					}
				}
			}
		}
	}
	for (i=0 ; i<BATCHSIZE ; ++i){
		for (j=0 ; j<PRODUCTZ2 ; ++j){
			for (k=0 ; k<PRODUCTX2 ; ++k){
				for (l=0 ; l<PRODUCTY2 ; ++l){
					BCMLGA1[i][IndexP2[i][k][l][j][0]][IndexP2[i][k][l][j][1]][j] = BCMLGA2[i][k][l][j];
				}
			}
		}
	}
	for (i=0 ; i<BATCHSIZE ; ++i){
		for (j=0 ; j<PRODUCTZ0 ; ++j){
			for (k=0 ; k<KERNELX1 ; ++k){
				for (l=0 ; l<KERNELY1 ; ++l){
					for (m=0 ; m<PRODUCTZ1 ; ++m){
						for (n=0 ; n<PRODUCTX1 ; ++n){
							for (p=0 ; p<PRODUCTY1 ; ++p){
								BKLGA1[k][l][j][m] += IArr[i][1 + j*IMAGEX*IMAGEY + (k-PADDING1+n)*IMAGEX + (l-PADDING1+p)] * BCMLGA1[i][n][p][m];
								if (j==PRODUCTZ0-1 && k==KERNELX1-1 && l==KERNELY1-1)
									BKBLGA1[m] += BCMLGA1[i][n][p][m];
								if (i==BATCHSIZE-1 &&  n==PRODUCTX1-1 && p==PRODUCTY1-1){
									F1[k][l][j][m] -= ETA*BKLGA1[k][l][j][m]/BATCHSIZE;
									if (i==BATCHSIZE-1 &&  n==PRODUCTX1-1 && p==PRODUCTY1-1 && l==KERNELY1-1 && k==KERNELX1-1 && j==PRODUCTZ0-1){
										BF1[m] -= ETA*BKBLGA1[m]/BATCHSIZE;
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

