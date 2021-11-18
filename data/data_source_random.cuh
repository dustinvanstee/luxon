#ifndef LUXON_DATA_SOURCE_RANDOM_CUH
#define LUXON_DATA_SOURCE_RANDOM_CUH

#include <vector>
#include <iostream>

#include "../common.cuh"
#include "../data/idataSource.cuh"

#define RAND_BLOCK_SIZE 1776     //Size of the Data in a random block
#define RAND_BLOCK_COUNT 1024    //Number of Messages in the Flow

struct Block {
    char buffer[RAND_BLOCK_SIZE];
    int  bufferSize;
} typedef block_t;

class RandomData : public IDataSource {
public:
    RandomData() {

        this->dataSourceType = eDataSourceType::RANDOM;

        for (int i = 0; i < RAND_BLOCK_COUNT; ++i) {
            block_t b;
            b.bufferSize = RAND_BLOCK_SIZE;
            for (int j = 0; j < RAND_BLOCK_SIZE; ++j) {
                int r = (uint8_t)((rand() % 256) + 1);
                b.buffer[j] = r;
            }
            randomBlocks.push_back(b);
        }
    }// Constructor declaration

    //Creates a random set of updates for the market data instruments, and puts them in a flow for sending.
    std::vector<block_t> createRandomUpdate(int numMsg)
    {
        std::vector<block_t> update;
        for(int j = 0; j < numMsg; j++)
        {
            block_t b = randomBlocks[(uint8_t)(rand()%RAND_BLOCK_COUNT)];
            update.push_back(b);
        }
        return update;
    }

private:
    std::vector<block_t> randomBlocks;

};


#endif //LUXON_FINANCE_MSG_CUH
