#include "design.hpp"


int main(){

	int i, j, k, l;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<double> d(0, 0.1);

	for (i=0 ; i<KERNELX1 ; ++i){
		for (j=0 ; j<KERNELY1 ; ++j){
			for (k=0 ; k<FILTERAMOUNTPREV1 ; ++k){
				for (l=0 ; l<FILTERAMOUNT1 ; ++l){
					F1[i][j][k][l] = d(gen);
					if(i==0 && j==0 && k==0)
						BF1[l]= d(gen);
				}
			}
		}
	}
	for (i=0 ; i<KERNELX3 ; ++i){
		for (j=0 ; j<KERNELY3 ; ++j){
			for (k=0 ; k<FILTERAMOUNTPREV3 ; ++k){
				for (l=0 ; l<FILTERAMOUNT3 ; ++l){
					F3[i][j][k][l] = d(gen);
					if(i==0 && j==0 && k==0)
						BF3[l]= d(gen);
				}
			}
		}
	}
	for (i=0 ; i<LENGTHPREV6 ; ++i){
		for (j=0 ; j<LENGTH6 ; ++j){
			for (k=0 ; k<BATCHSIZE ; ++k){
				if(k==0)
					W6[i][j] = d(gen);
				if(i==0)
					BW6[k][j] = d(gen);
			}
		}
	}
	for (i=0 ; i<LENGTHPREV7 ; ++i){
		for (j=0 ; j<LENGTH7 ; ++j){
			for (k=0 ; k<BATCHSIZE ; ++k){
				if(k==0)
					W7[i][j] = d(gen);
				if(i==0)
					BW7[k][j] = d(gen);
			}
		}
	}

	type_index epochIndex, nIndex;

	type_weight I[BATCHSIZE][1+IMAGEX*IMAGEY*IMAGECHANNEL];
	type_weight POUT[BATCHSIZE][CLASSSIZE];
	char imageRow[1+4*IMAGEX*IMAGEY*IMAGECHANNEL];

	long rowBeginningPositions[NTRAIN] = {0};

	FILE* f_train = fopen("mnist_train.csv", "r+");

	for (nIndex = 1; nIndex <= NTRAIN; ++nIndex) {
		rowBeginningPositions[nIndex - 1] = ftell(f_train);
		fscanf(f_train, "%s\n", imageRow);
	}

	fclose(f_train);

	for (epochIndex = 0; epochIndex < EPOCH ; ++epochIndex){
		FILE* f_train = fopen("mnist_train.csv","r+");

		vector<int> indices(NTRAIN);
		for (i = 0; i < NTRAIN; ++i) indices[i] = i;
		shuffle(indices.begin(), indices.end(), default_random_engine(time(NULL)));

		for (nIndex = 1; nIndex <= NTRAIN ; ++nIndex){
			fseek(f_train, rowBeginningPositions[indices[nIndex-1]], SEEK_SET);

			fscanf(f_train, "%s\n", imageRow);
			char* token;
			int ind = 0;
			token = strtok(imageRow, ",");
			I[(nIndex%BATCHSIZE-1+BATCHSIZE)%BATCHSIZE][ind] = atoi(token);
			ind += 1;
			token = strtok(NULL, ",");
			while (token != NULL){
				I[(nIndex%BATCHSIZE-1+BATCHSIZE)%BATCHSIZE][ind] = atoi(token)/255.0;
				ind += 1;
				token = strtok(NULL, ",");
			}

			if (nIndex%BATCHSIZE != 0)
				continue;

			train(I, POUT, false);

			if (nIndex % 1000 == 0){
				printf("Epoch: %d, nIndex: %d\n", epochIndex+1, nIndex);
			}
		}

		//Testing
		printf("TESTING\n");
		double trueCnt = 0.0, falseCnt = 0.0;


		FILE* f_test = fopen("mnist_test.csv", "r+");

		for (nIndex = 1; nIndex <= NTEST ; ++nIndex){
			fscanf(f_test, "%s\n", imageRow);
			char* token;
			int classImage;

			int ind = 0;
			token = strtok(imageRow, ",");
			I[0][ind] = atoi(token);
			ind += 1;
			token = strtok(NULL, ",");
			while (token != NULL){
				I[0][ind] = atoi(token)/255.0;
				ind += 1;
				token = strtok(NULL, ",");
			}

			train(I, POUT, true);

			double val = 0;
			classImage = I[0][0];

			int valInd = 0;

			for (i=0 ; i<CLASSSIZE ; i++){
				if (POUT[0][i]>=val){
					val = POUT[0][i];
					valInd = i;
				}
			}

			if (valInd == classImage)
				trueCnt += 1;
			else
				falseCnt += 1;
		}
		printf("Accuracy: %f\n\n", (1.0*trueCnt)/(trueCnt + falseCnt));


		fclose(f_test);
		fclose(f_train);

	}

	return 0;

}

