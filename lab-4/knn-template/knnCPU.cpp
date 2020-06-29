#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include <vector>
#include <algorithm>
#include <map>

#include "base.h"

float distance(float* first, float* second)
{
    float sum = 0;
    for (int i = 0; i < DIMENSION; i++)
    {
        sum += pow(first[i] - second[i], 2);
    }
    return sqrt(sum);
}

void knnSerial(float* coords, float* newCoords, int* classes, int numClasses, int numSamples, int numNewSamples, int k)
{
    // Connect distances to classes for convinience
    std::pair<float, int> points[numSamples];

    // Iterate over points to classify
    for (int i = 0; i < numNewSamples; i++)
    {
        for (int j = 0; j < numSamples; j++)
        {
            points[j].second = classes[j];
        }

        // Check all points
        for (int j = 0; j < numSamples; j++)
        {
            float dist = distance(&coords[DIMENSION * j], &newCoords[DIMENSION * i]);
            points[j].first = dist;
        }
        std::sort(points, points + numSamples);

        printf("\n");
        for (int j = 0; j < numSamples; j++)
        {
            printf("[distance: %f class: %d]\n", points[j].first, points[j].second);
        }

        std::map<int, int> frequencies;
        for (int j = 0; j < k; j++)
        {
            frequencies[points[j].second]++;
        }
        int m = 1;
        int currentClass = points[0].second;
        for (auto freq : frequencies)
        {
            printf(" {class: %d frequency: %d} ", freq.first, freq.second);
            if (freq.second > m)
            {
                m = freq.second;
                currentClass = freq.first;
            }
        }
        printf("\n");
        classes[numSamples + i] = currentClass;
    }
}
