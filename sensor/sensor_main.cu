#include <iostream>
#include <unistd.h>
#include <chrono>
#include "math.h"
#include "Sensor.cuh"

void PrintUsage()
{
    cout << "usage: sensorSim [ -s data_source] [-t transport_mode] [-d msg_payload_size] [-n num_pkts] [-b msg_blk_size] [-l local-addr] mcast-addr" << endl;
    cout << "\t mcast-addr            : multicast group where sensor publishes data (required argument)" << endl;
    cout << "\t[-s data_source]       : a data source, can be PCAP, RANDOM, FINANCE, PAT (default: RANDOM)" << endl;
    cout << "\t[-t transport mode]    : transport mode, PRINT, UDP, RDMA_UD, NONE (default: PRINT)" << endl;
    cout << "\t[-d msg_payload_size]  : size of the individual messages in the message buffer array (default : 100Bytes)" << endl;
    cout << "\t[-n num_pkts]          : Number of packets to send across the transport. (default:1024)" << endl;
    cout << "\t[-b msg_blk_size]      : Number of messages in the message buffer. (default:1024)" << endl;
    cout << "\t[-l ip-addr]           : local ip addresss to bind. (default: bind to first address)" << endl;
}

int main(int argc,char *argv[], char *envp[]) {
     /*
     * Parsing the command line and validating the input
     */
    int op;
    string fileName;
    eTransportType transportType {eTransportType::PRINT};
    eDataSourceType dataSourceType {eDataSourceType::RANDOM};
    string mcastAddr;
    string srcAddr;
    char hostBuffer[256];
    ITransport* transport;
    IDataSource* dataSource;
    int msg_payload_size = 100; //bytes
    int num_pkts = 1024;        // number of individual message packets to send.
    int msg_blk_size= 1024;     // number of individual message packets in a message block.

    
    while ((op = getopt(argc, argv, "s:t:d:n:b:l:")) != -1) {
        switch (op) {
            case 's':
                dataSourceType = dataSource->strToDataSourceType(optarg);
                break;
            case 't':
                transportType = transport->strToTransportType(optarg);
                break;
            case 'd':
                msg_payload_size = atoi(optarg);
                break;
            case 'n':
                num_pkts = atoi(optarg);
                break;
            case 'b':
                msg_blk_size = atoi(optarg);
                break;
            case 'l':
                srcAddr = optarg;
                break;
            default:
                PrintUsage();
                return -1;
        }
    }

    if(argc <= optind)
    {
        PrintUsage();
        return -1;
    }
    else
    {
        mcastAddr = argv[optind++];
    }
    gethostname(hostBuffer, sizeof(hostBuffer));
    cout << "********  ********  ********  ********  ********  ********" << endl;
    cout << "Sensor Simulator - Read Data Source and sends data buffers to target" << endl;
    cout << "********  ********  ********  ********  ********  ********" << endl;
    cout << "Running on " << hostBuffer <<endl;
    cout << "Local Address: " << (srcAddr.empty() ? "Default" : srcAddr) << endl;
    cout << "Mcast Group Address: " << mcastAddr << endl;
    cout << "Source: " << dataSource->DataSourceTypeToStr(dataSourceType) << endl;
    cout << "Transport Mode: " << transport->TransportTypeToStr(transportType) << endl <<endl;

    //Create the Transport
    switch(transportType) {
        case eTransportType::UDP :
            transport = new UdpTransport(srcAddr, mcastAddr, eTransportRole::SENSOR);
            break;
        case eTransportType::RDMA_UD :
            transport = new RdmaUdTransport(srcAddr, mcastAddr, eTransportRole::SENSOR);
            break;
        case eTransportType::PRINT :
            transport = new PrintTransport();
            break;
        case eTransportType::NONE :
            transport = new NoneTransport();
            break;
        default :
            cout << "No or Invalid Transport Specified" << endl;
            PrintUsage();
            return -1;
    }

    //Create the Sensor
    Sensor s = Sensor(transport, dataSourceType);

    MessageBlk msgBlk;

    //Create the update flow based on the data source.
    switch(dataSourceType) {
        case eDataSourceType::PCAP:
            s.createPCAPFlow(msgBlk, fileName);
            break;
        case eDataSourceType::RANDOM:
            s.createRandomFlow(msgBlk);
            break;
        case eDataSourceType::PAT:
            s.createPatternFlow(msgBlk);
            break;
        case eDataSourceType::FINANCE:
            s.createFinanceFlow(msgBlk);
            break;
        default :
            cout << "No valid data source" << endl;
            PrintUsage();
            return -1;
    }
    npt("Sensor Statistics %c", ':');
    cout << "Sensor Flow has " << s.getFlowMsgCount(msgBlk) << " messages w/ avg size " << s.getFlowMsgAvgSize(msgBlk) << endl;
    cout << "Sensor Flow total size is " << s.getFlowByteLength(msgBlk) << " bytes " << endl;

    long long sentMessages = 0;
    int flowLength = s.getFlowMsgCount(msgBlk);

    timer t_runTime;

    int num_iters = (int) ceil(float(num_pkts) / float(msg_blk_size));
    pt("Sensor Flow sending %d individual message packets in chunks of %d\n", num_pkts, msg_blk_size);
    pt("Sensor Flow will send %d Message Blocks\n", num_iters);
    int i = 1; //First Iteration
    t_runTime.start();
    do {
        if (0 != s.sendFlow(msgBlk))
        {
            cout << "Transport Error Sending sensor Flow - Exiting" << endl;
            return -1;
        }
        sentMessages += flowLength;
    } while (i++ < num_iters);
    t_runTime.stop();
    cerr << "\rSent " << sentMessages << " messages\t Time: " << t_runTime.usec_elapsed() << "usec"  << endl;

    cerr << "Rate " << (sentMessages/t_runTime.usec_elapsed()) * 1000 << " Messages Per Second" << endl;


    return 0;
}
