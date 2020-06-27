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
    for (int i = 0; i < numSamples; i++)
    {
        points[i].second = classes[i];
    }

    // Iterate over points to classify
    for (int i = 0; i < numNewSamples; i++)
    {
        // Check all points
        for (int j = 0; j < numSamples; j++)
        {
            float dist = distance(&coords[DIMENSION * j], &newCoords[DIMENSION * i]);
            points[j].first = dist;
        }
        std::sort(points, points + numSamples);
        std::map<int, int> frequencies;
        for (int j = 0; j < k; j++)
        {
            frequencies[points[numSamples - j - 1].second]++;
        }
        int m = 0;
        int currentClass = classes[0];
        for (auto freq : frequencies)
        {
            if (freq.second > m)
            {
                m = freq.second;
                currentClass = freq.first;
            }
        }
        classes[numSamples + i] = currentClass;
    }
}
