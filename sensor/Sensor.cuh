#ifndef LUXON_SENSOR_CUH
#define LUXON_SENSOR_CUH

#include <vector>
#include <iostream>
#include <pcap.h>

#include "../common.cuh"
#include "../data/idataSource.cuh"
#include "../transport/itransport.cuh"

typedef struct
{
   int msgCount;
   Message* msgBlk;
} flow;

class Sensor {
public:
    Sensor(ITransport*, eDataSourceType); // Constructor declaration

    //Flow Creation Functions
    int createRandomFlow(flow &f, int numMsg);
    int createFinanceFlow(flow &f, int numMsg);
    int createPCAPFlow(flow &f, std::string fileName); //TODO: Need to create a data source class for pcap, this can solve overrun issue.

    int getFlowByteLength(flow &f);                    //Length of the flow in Bytes
    int getFlowMsgCount(flow &f);                      //Number of Messages in the Flow
    int getFlowMsgAvgSize(flow &f);                    //Average Size of a Message in the flow

    //Flow display
    void printFlow(flow &f);
    int sendFlow(flow &f);

private:
    IDataSource*        dataSource;
    ITransport*         transport;

};


#endif //LUXON_SENSOR_CUH
