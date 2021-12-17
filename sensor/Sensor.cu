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

int Sensor::createPCAPFlow(flow &f, std::string fileName)
{
    f.msgBlk = NULL;
    f.msgCount = 0;

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
    if(!transport->createMessageBlock(f.msgBlk, eMsgBlkLocation::HOST)){
        return -1;
    }

    //double lastMsgSec = 0, deltaSec = 0;
    double lastMsgUsec = 0, deltaUSec = 0;
    int i = 0;
    // TODO : add checker PCAP > msg_blk_size
    while (int returnValue = pcap_next_ex(handle, &header, &data) >= 0) {

        // Set the size of the Message in bytes
        DEBUG(printf("Packet size: %d bytes\n", header->caplen));

        // Set an interval time since last Message
        //printf("Epoch Time: %l:%l seconds\n", header->ts.tv_sec, header->ts.tv_usec);

        //deltaSec = (header->ts.tv_sec) - lastMsgSec; //TODO: Calculating Message interval factor in > 1 Second delays
        deltaUSec = (header->ts.tv_usec) - lastMsgUsec;

        f.msgBlk[i].seqNumber = i;
        f.msgBlk[i].interval = deltaUSec;
        f.msgBlk[i].bufferSize = header->caplen;
        memcpy(f.msgBlk[i].buffer, data, header->caplen);

        std::cout << "Adding to flow: ";
        transport->printMessage(&f.msgBlk[i], 0);
        std::cout << std::endl;

        i++;
    }

    return 0;
}

int Sensor::createRandomFlow(flow &f, int numMsg) {
    f.msgBlk = NULL;
    f.msgCount = 0;

    if(dataSource->getType() != eDataSourceType::RANDOM) {
        return -1;
    }

    RandomData* rd = (RandomData *) dataSource;

    std::vector<block_t> updates = rd->createRandomUpdate(numMsg);

    transport->createMessageBlock(f.msgBlk, eMsgBlkLocation::HOST);

    int i = 0;
    for (auto &block_t : updates)
    {
        f.msgBlk[i].seqNumber = i;
        f.msgBlk[i].interval = 100;
        f.msgBlk[i].bufferSize = sizeof(block_t);
        memcpy(&f.msgBlk[i].buffer, &block_t, sizeof(block_t));
        i++;
    }
    f.msgCount = i;
    return 0;
}

int Sensor::createFinanceFlow(flow &f, int numMsg) {

    f.msgBlk = NULL;
    f.msgCount = 0;

    if(dataSource->getType() != eDataSourceType::FINANCE) {
        return -1;
    }
    MarketData* md = (MarketData *) dataSource;

    std::vector<instrument> updates = md->createRandomUpdate(numMsg);

    transport->createMessageBlock(f.msgBlk, eMsgBlkLocation::HOST);

    int i = 0;
    for (auto &instrument : updates)
    {
        f.msgBlk[i].seqNumber = i;
        f.msgBlk[i].interval = 100;
        f.msgBlk[i].bufferSize = sizeof(instrument);
        memcpy(&f.msgBlk[i].buffer, &instrument, sizeof(instrument));
        i++;
    }
    f.msgCount = i;
    return 0;
}

void Sensor::printFlow(flow &f) {
    // loop through the messages and print as hexidecimal representations of octets
    for (int i=0; (i < f.msgCount ) ; i++) {
      transport->printMessage(&f.msgBlk[i], 32);
      printf("\n\n ");
    }

    return;
}

int Sensor::getFlowMsgCount(flow &f) {
    return f.msgCount;
}

int Sensor::getFlowByteLength(flow &f) {
    int sumOfLengths = 0;
    for (int i=0; (i < f.msgCount ) ; i++) {
        sumOfLengths += f.msgBlk->bufferSize;
    }

    return sumOfLengths;
}

int Sensor::getFlowMsgAvgSize(flow &f) {
    int sumOfLengths = 0;
    for (int i=0; (i < f.msgCount ) ; i++) {
        sumOfLengths += f.msgBlk->bufferSize;
    }

    return sumOfLengths/f.msgCount;
}

int Sensor::sendFlow(flow &f) {
    for (int i=0; (i < f.msgCount ) ; i++) {
        if(0 != transport->push(&f.msgBlk[i]))
        {
            return -1;
        }
    }
    return 0;
}


