//
// ChangeLog 
//  121721 DV added random walk to finance instruments

#ifndef LUXON_FINANCE_MSG_CUH
#define LUXON_FINANCE_MSG_CUH

#include <vector>
#include <iostream>

#include "../common.cuh"
#include "../data/idataSource.cuh"
#include <random>

#define NUMBER_INSTRUMENTS 10

struct Instrument {
    std::string  symbol; //TODO: Change these to char or fixed length string.
    std::string  exchange;
    int     bid;
    int     ask;
    int     volume;
    int     pkt_num;
} typedef instrument;

class MarketData : public IDataSource {
public:
    double mu {0.01};
    double sigma {0.001};
    double dt = {1.0};
    MarketData() {

        this->dataSourceType = eDataSourceType::FINANCE;

        this->marketData= {
                {"AAPL", "NYSE", 100, 101, 9999, -1},
                {"MSFT", "NYSE", 100, 101, 9999, -1},
                {"AMZN", "NYSE", 100, 101, 9999, -1},
                {"FB",   "NYSE", 100, 101, 9999, -1},
                {"GOOGL","NYSE", 100, 101, 9999, -1},
                {"GOOG", "NYSE", 100, 101, 9999, -1},
                {"TSLA", "NYSE", 100, 101, 9999, -1},
                {"NVDA", "NYSE", 100, 101, 9999, -1},
                {"BRK.B","NYSE", 100, 101, 9999, -1},
                {"JPM",  "NYSE", 100, 101, 9999, -1},
        };

    }; // Constructor declaration

    //Creates a random set of updates for the market data instruments, and puts them in a flow for sending.
    std::vector<instrument> createRandomUpdate(int numMsg)
    {
        std::vector<instrument> update;
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0,1.0);

        for(int j = 0; j < numMsg; j++)
        {
            instrument i = marketData[(uint8_t)(rand()%NUMBER_INSTRUMENTS)];
            int dS = mu * i.bid * dt + sigma * i.bid * distribution(generator) * sqrt(dt);
            i.bid = i.bid + dS;
            i.ask = i.bid + 0.01;
            i.volume++;
            i.pkt_num = j;
            update.push_back(i);
        }
        return update;
    }

private:
    std::vector<instrument> marketData;

};


#endif //LUXON_FINANCE_MSG_CUH
