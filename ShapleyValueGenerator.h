#pragma once
#include <vector>
#include <random>

using namespace std;

static class ShapleyValueGenerator
{
    //int Factorial(int num);

public:

    int numOfAgents;

    vector<double> CreateRandomCharacterTable(default_random_engine& re);

private:

    vector<vector<int>> permutations;

    void GenerateAllPermutations(vector<int>& v, int size);

    vector<vector<int>> GenerateAllCombinations();

    // hey dont forget that v1 isnt sorted here, so if you use this for something where v1 has to be an unsorted list, uhh, just remember that this is why its not working. also v1 and v2 is a reference, not a copy.
    bool DoVectorsContainSameElements(vector<int>& v1, vector<int> v2);

public:

    vector<double> CalculateShapleyValues(vector<double>& characterTable);

};
