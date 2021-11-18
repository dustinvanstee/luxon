//
// Created by alex on 7/15/20.
//

#ifndef LUXON_FINANCE_MSG_CUH
#define LUXON_FINANCE_MSG_CUH

#include <vector>
#include <iostream>

#include "../common.cuh"
#include "../data/idataSource.cuh"

#define NUMBER_INSTRUMENTS 10

struct Instrument {
    std::string  symbol; //TODO: Change these to char or fixed length string.
    std::string  exchange;
    int     bid;
    int     ask;
    int     volume;
} typedef instrument;

class MarketData : public IDataSource {
public:
    MarketData() {

        this->dataSourceType = eDataSourceType::FINANCE;

        this->marketData= {
                {"AAPL", "NYSE", 0, 0, 0},
                {"MSFT", "NYSE", 0, 0, 0},
                {"AMZN", "NYSE", 0, 0, 0},
                {"FB", "NYSE", 0, 0, 0},
                {"GOOGL", "NYSE", 0, 0, 0},
                {"GOOG", "NYSE", 0, 0, 0},
                {"TSLA", "NYSE", 0, 0, 0},
                {"NVDA", "NYSE", 0, 0, 0},
                {"BRK.B", "NYSE", 0, 0, 0},
                {"JPM", "NYSE", 0, 0, 0},
        };

    }; // Constructor declaration

    //Creates a random set of updates for the market data instruments, and puts them in a flow for sending.
    std::vector<instrument> createRandomUpdate(int numMsg)
    {
        std::vector<instrument> update;
        for(int j = 0; j < numMsg; j++)
        {
            instrument i = marketData[(uint8_t)(rand()%NUMBER_INSTRUMENTS)];
            i.bid = (uint8_t)((rand()%256)+1);
            i.ask = (uint8_t)((rand()%256)+1);
            i.volume++;
            update.push_back(i);
        }
        return update;
    }

private:
    std::vector<instrument> marketData;

};


#endif //LUXON_FINANCE_MSG_CUH
