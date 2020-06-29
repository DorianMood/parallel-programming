#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>


float distance(float* first, float* second);

void knnSerial(float* coords, float* newCoords, int* classes, int numClasses, int numSamples, int numNewSamples, int k);
