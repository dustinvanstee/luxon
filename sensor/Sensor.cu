#include "Sensor.cuh"


Sensor::Sensor(ITransport* t, eDataSourceType dst) {
    transport = t;

    switch(dst) {
        case eDataSourceType::PCAP :
            break;
        case eDataSourceType::RANDOM :
            dataSource = new RandomData();
            break;
        case eDataSourceType::FINANCE :
            dataSource = new MarketData();
            break;
        case eDataSourceType::PAT :
            dataSource = new PatternData();
            break;
    }
}

int Sensor::createPCAPFlow(MessageBlk &mb, std::string fileName)
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
    if(!transport->createMessageBlock(&mb, eMsgBlkLocation::HOST)){
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

        mb.messages[i].seqNumber = i;
        mb.messages[i].interval = deltaUSec;
        mb.messages[i].bufferSize = header->caplen;
        memcpy(mb.messages[i].buffer, data, header->caplen);

        std::cout << "Adding to flow: ";
        transport->printMessage(&mb.messages[i], 0);
        std::cout << std::endl;

        i++;
    }

    return 0;
}

int Sensor::createRandomFlow(MessageBlk &mb) {
    mb.msgCount = 0;

    if(dataSource->getType() != eDataSourceType::RANDOM) {
        return -1;
    }

    RandomData* rd = (RandomData *) dataSource;

    std::vector<randomBlock_t> updates = rd->createRandomUpdate(mb.msgBlkSize);

    transport->createMessageBlock(&mb, eMsgBlkLocation::HOST);

    int i = 0;
    for (auto &block_t : updates)
    {
        mb.messages[i].seqNumber = i;
        mb.messages[i].interval = 100;
        mb.messages[i].bufferSize = min(MSG_MAX_SIZE, static_cast<int>(sizeof(block_t)));
        memcpy(& mb.messages[i].buffer, &block_t, min(MSG_MAX_SIZE, static_cast<int>(sizeof(block_t))));
        i++;
    }
    mb.msgCount = i;
    return 0;
}

int Sensor::createPatternFlow(MessageBlk &mb, int msg_blk_size) {
    mb.msgCount = 0;

    if(dataSource->getType() != eDataSourceType::PAT) {
        return -1;
    }

    PatternData* rd = (PatternData *) dataSource;

    std::vector<patternBlock_t> updates = rd->createPatternUpdate(msg_blk_size);

    transport->createMessageBlock(&mb, eMsgBlkLocation::HOST);

    int i = 0;
    for (auto &patternBlock_t : updates)
    {
        mb.messages[i].seqNumber = i;
        mb.messages[i].interval = 100;
        mb.messages[i].bufferSize = min(MSG_MAX_SIZE, static_cast<int>(sizeof(patternBlock_t)));
        memcpy(& mb.messages[i].buffer, &patternBlock_t, min(MSG_MAX_SIZE, static_cast<int>(sizeof(patternBlock_t))));
        i++;
    }
    mb.msgCount = i;
    return 0;
}

int Sensor::createFinanceFlow(MessageBlk &mb, int msg_blk_size) {

    mb.msgCount = 0;

    if(dataSource->getType() != eDataSourceType::FINANCE) {
        return -1;
    }
    MarketData* md = (MarketData *) dataSource;

    std::vector<instrument> updates = md->createRandomUpdate(msg_blk_size);

    transport->createMessageBlock(&mb, eMsgBlkLocation::HOST);

    int i = 0;
    for (auto &instrument : updates)
    {
        mb.messages[i].seqNumber = i;
        mb.messages[i].interval = 100;
        mb.messages[i].bufferSize = sizeof(instrument);
        memcpy(& mb.messages[i].buffer, &instrument, sizeof(instrument));
        i++;
    }
    mb.msgCount = i;
    return 0;
}

void Sensor::printFlow(MessageBlk &mb) {
    // loop through the messages and print as hexidecimal representations of octets
    for (int i=0; (i < mb.msgCount ) ; i++) {
      transport->printMessage(&mb.messages[i], 32);
      printf("\n\n ");
    }

    return;
}

int Sensor::getFlowMsgCount(MessageBlk &mb) {
    return mb.msgCount;
}

int Sensor::getFlowByteLength(MessageBlk &mb) {
    int sumOfLengths = 0;
    for (int i=0; (i < mb.msgCount ) ; i++) {
        sumOfLengths += mb.messages[i].bufferSize;
    }

    return sumOfLengths;
}

int Sensor::getFlowMsgAvgSize(MessageBlk &mb) {
    int sumOfLengths = 0;
    for (int i=0; (i < mb.msgCount ) ; i++) {
        sumOfLengths += mb.messages[i].bufferSize;
    }
    return sumOfLengths/mb.msgCount;
}

int Sensor::sendFlow(MessageBlk &mb) {
    if (0 != transport->push(&mb, mb.msgCount)) {
        return -1;
    }
    return 0;
}


