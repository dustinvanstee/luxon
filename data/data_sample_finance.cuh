#ifndef LUXON_DATA_SOURCE_FINANCE_CUH
#define LUXON_DATA_SOURCE_FINANCE_CUH

#include <vector>
#include <iostream>
#include <string>

#include "../common.cuh"
#include "../data/idataSource.cuh"
#include <random>

// Base size is 40B ..
// Add dummy field to increase ?? 
struct Instrument {
    char    symbol[5]; //TODO: Change these to char or fixed length string.
    char    exchange[5];
    double     bid;
    double     ask;
    int     volume;
    int     pkt_num;
} typedef instrument;

class MarketData : public IDataSource {
public:
    double mu {0.01};
    double sigma {0.001};
    double dt = {1.0};
    int numInstruments;
    MarketData() {
        npt("Calling MarketData Constructor %c\n",':');
        this->dataSourceType = eDataSourceType::FINANCE;
        std::vector<std::string> stks = {"AAPL", "GOOG"};
        this->numInstruments = stks.size();

        for(auto &s: stks) {
            //Instrument i = {"AAPL", "NYSE", 100, 101, 9999, -1};
            //Instrument i = {s.c_str(), "NYSE", 100, 101, 9999, -1};
            Instrument i = {"TEMP", "NYSE", 100, 101, 9999, -1};
            std::copy(s.begin(),s.end(), i.symbol);
            this->marketData.push_back(i);
        }
        //i = {"AAPL", "NYSE", 100, 101, 9999, -1};
        //this->marketData.push_back(i);

        //this->marketData= {
        //        {"AAPL", "NYSE", 100, 101, 9999, -1},
        //        {"MSFT", "NYSE", 100, 101, 9999, -1},
        //        {"AMZN", "NYSE", 100, 101, 9999, -1},
        //        {"FB__",   "NYSE", 100, 101, 9999, -1},
        //        {"GOOL","NYSE", 100, 101, 9999, -1},
        //        {"GOOG", "NYSE", 100, 101, 9999, -1},
        //        {"TSLA", "NYSE", 100, 101, 9999, -1},
        //        {"NVDA", "NYSE", 100, 101, 9999, -1},
        //        {"BRKB","NYSE", 100, 101, 9999, -1},
        //        {"JPM_", "NYSE", 100, 101, 9999, -1},
        //};

    }; // Constructor declaration

    //Creates a random set of updates for the market data instruments, and puts them in a flow for sending.
    std::vector<instrument> createRandomUpdate(int numMsg)
    {
        std::vector<instrument> update;
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0,1.0);
        npt("Adding %d messages in Random Update\n", numMsg);
        for(int j = 0; j < numMsg; j++)
        {
            instrument i = marketData[(uint8_t)(rand()%this->numInstruments)];
            double dS = mu * i.bid * dt + sigma * i.bid * distribution(generator) * sqrt(dt);
            i.bid = i.bid + dS;
            i.ask = i.bid + 0.01;
            i.volume++;
            i.pkt_num = j;
            update.push_back(i);
        }
        pt("Added %d MarketData updates", (int) update.size());
        return update;
    }

    void summarizeMessage(Message* m) {
        // 1. cast the data, and print
        Instrument *ins = new(Instrument);
        memcpy(ins, m, sizeof(Instrument));
        //std::fprintf(std::cerr, "Instument: %s %s %d", ins->symbol, ins->exchange, ins->bid);
        std::cerr <<  "Instrument: " << ins->symbol << ins->exchange << ins->bid << std::endl;
        //TODO: Not formating correctly, Dustin can you get it how oyu want it.
    }


private:
    std::vector<instrument> marketData;

};


#endif //LUXON_DATA_SOURCE_FINANCE_CUH
