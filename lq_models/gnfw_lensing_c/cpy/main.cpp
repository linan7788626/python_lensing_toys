#include <stdlib.h>
#include <iostream>

using namespace std;

//uncomment the following line if you know the data size
//#define SIZE_KNOWN

#ifdef SIZE_KNOWN
const int data_size = 100;
double data[data_size];
#endif

int main(int argc, char * argv[]){

	// if you know the incoming data size then you could do something like
#ifdef SIZE_KNOWN
	// read in data
	for(int i=0:i<data_size;i++){
		data[i] = atof(argv[i+1]); //data start from 0
	}

#else
	for(int i=1;i<argc;i++){
		//argv[0] is not parameter but the name of the program!! so start with 1
		double value = atof(argv[i]);
		value += 1; // do something with the incoming data;
		cout<<value<<" "; //divide output with a space
	}
#endif

#ifdef SIZE_KNOWN
	// process data
	for(int i=0:i<data_size;i++){
		data[i] +=1;  
	}

	// output data
	for(int i=0:i<data_size;i++){
		cout<<data[i<<" ";
	}	
#endif

	return 0;
}