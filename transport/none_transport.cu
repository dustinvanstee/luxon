#include <cstdio>
#include <arpa/inet.h>
#include <iostream>

#include "none_transport.cuh"

NoneTransport::NoneTransport()
{
    this->transportType = eTransportType::NONE;
}

int NoneTransport::push(MessageBlk* m, int numMsg)
{
    push_counter++;
    return 0;
}

int NoneTransport::pop(MessageBlk* m, int numReqMsg, int& numRetMsg)
{
    pop_counter++;
    return 0;
}

int NoneTransport::createMessageBlock(MessageBlk* msgBlk, eMsgBlkLocation dest)
{
    return createMessageBlockHelper(msgBlk, dest);
}

int NoneTransport::freeMessageBlock(MessageBlk* msgBlk, eMsgBlkLocation dest)
{
    return freeMessageBlockHelper(msgBlk, dest);
}


