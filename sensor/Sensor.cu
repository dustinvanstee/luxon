//
// Created by alex on 7/15/20.
//

#include "Sensor.cuh"

#include "../data/data_sample_finance.cuh"
#include "../data/data_source_random.cuh"

Sensor::Sensor(ITransport* t, eDataSourceType dst) {
    transport = t;

    switch(dst) {
        case eDataSourceType::PCAP:
            break;
        case eDataSourceType::RANDOM :
            dataSource = new RandomData();
            break;
        case eDataSourceType::FINANCE :
            dataSource = new MarketData();
            break;
    }
}

int Sensor::createPCAPFlow(std::string fileName)
{
    if(dataSource->getType() != eDataSourceType::PCAP) {
        return -1;
    }
    //Get the Message count
    pcap_t *handle;
    char errbuf[PCAP_ERRBUF_SIZE];
    struct pcap_pkthdr *header;
    const u_char *data;

    handle = pcap_open_offline(fileName.c_str(), errbuf);

    if (handle == nullptr) {
        std::cout << "Couldn't open pcap file "<< fileName << ":" << errbuf << std::endl;
        return(2);
    }

    //Create the Flow, allocated the memory
    Message* m;

    //double lastMsgSec = 0, deltaSec = 0;
    double lastMsgUsec = 0, deltaUSec = 0;
    int i = 0;
    while (int returnValue = pcap_next_ex(handle, &header, &data) >= 0) {

        // Set the size of the Message in bytes
        DEBUG(printf("Packet size: %d bytes\n", header->caplen));

        // Set an interval time since last Message
        //printf("Epoch Time: %l:%l seconds\n", header->ts.tv_sec, header->ts.tv_usec);

        //deltaSec = (header->ts.tv_sec) - lastMsgSec; //TODO: Calculating Message interval factor in > 1 Second delays
        deltaUSec = (header->ts.tv_usec) - lastMsgUsec;

        m = transport->createMessage();
        m->seqNumber = i++;
        m->interval = deltaUSec;
        m->bufferSize = header->caplen;
        memcpy(m->buffer, data, header->caplen);

        std::cout << "Adding to flow: ";
        transport->printMessage(m, 0);
        std::cout << std::endl;

        flow.push_back(m);
    }

    return 0;
}

int Sensor::createRandomFlow(int numMsg) {
    if(dataSource->getType() != eDataSourceType::RANDOM) {
        return -1;
    }

    RandomData* rd = (RandomData *) dataSource;

    std::vector<block_t> updates = rd->createRandomUpdate(numMsg);

    int i = 0;
    for (auto &block_t : updates)
    {
        Message* m = NULL;
        m = transport->createMessage();
        m->seqNumber = i;
        m->interval = 100;
        m->bufferSize = sizeof(block_t);
        memcpy(&m->buffer, &block_t, sizeof(block_t));
        flow.push_back(m);
        i++;
    }

    return 0;
}

int Sensor::createFinanceFlow(int numMsg) {
    if(dataSource->getType() != eDataSourceType::FINANCE) {
        return -1;
    }
    MarketData* md = (MarketData *) dataSource;

    std::vector<instrument> updates = md->createRandomUpdate(numMsg);

    int i = 0;
    for (auto &instrument : updates)
    {
        Message* m = NULL;
        m = transport->createMessage();
        m->seqNumber = i;
        m->interval = 100;
        m->bufferSize = sizeof(instrument);
        memcpy(&m->buffer, &instrument, sizeof(instrument));
        flow.push_back(m);
        i++;
    }

    return 0;
}

void Sensor::printFlow() {
    // loop through the messages and print as hexidecimal representations of octets
    for (int i=0; (i < flow.size() ) ; i++) {
      transport->printMessage(flow[i], 32);
      printf("\n\n ");
    }

    return;
}

int Sensor::getFlowMsgCount() {
    return flow.size();
}

int Sensor::getFlowByteLength() {
    int sumOfLengths = 0;
    for (auto& n : flow)
        sumOfLengths += n->bufferSize;

    return sumOfLengths;
}

int Sensor::getFlowMsgAvgSize() {
    int sumOfLengths = 0;
    for (auto& n : flow)
        sumOfLengths += n->bufferSize;

    return sumOfLengths/flow.size();
}

int Sensor::sendFlow() {
    for(int i = 0; i < flow.size() ; i++)
    {
        if(0 != transport->push(flow[i]))
        {
            return -1;
        }
    }
    return 0;
}


