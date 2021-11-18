#ifndef LUXON_SENSOR_CUH
#define LUXON_SENSOR_CUH

#include <vector>
#include <iostream>
#include <pcap.h>

#include "../common.cuh"
#include "../data/idataSource.cuh"
#include "../transport/itransport.cuh"

class Sensor {
public:
    Sensor(ITransport*, eDataSourceType); // Constructor declaration

    //Flow Creation Functions
    int createRandomFlow(int numMsg);
    int createFinanceFlow(int numMsg);
    int createPCAPFlow(std::string fileName); //TODO: Need to create a data source class for pcap

    int getFlowByteLength();                    //Length of the flow in Bytes
    int getFlowMsgCount();                      //Number of Messages in the Flow
    int getFlowMsgAvgSize();                    //Average Size of a Message in the flow

    //Flow display
    void printFlow();
    int sendFlow();

private:
    std::vector<Message*> flow; //This is the Flow the sensor will send.
    IDataSource* dataSource;
    ITransport* transport;

};


#endif //SENSORSIM_SENSOR_CUH
