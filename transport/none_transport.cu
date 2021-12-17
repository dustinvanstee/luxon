#include <cstdio>
#include <arpa/inet.h>
#include <iostream>

#include "none_transport.cuh"

NoneTransport::NoneTransport()
{
    this->transportType = eTransportType::NONE;
}

int NoneTransport::push(Message* m)
{
    return 0;
}

int NoneTransport::pop(Message* m, int numReqMsg, int& numRetMsg)
{
    return 0;
}

int NoneTransport::createMessageBlock(Message* &msgBlk, eMsgBlkLocation dest)
{
    return createMessageBlockHelper(msgBlk, dest);
}

int NoneTransport::freeMessageBlock(Message* msgBlk, eMsgBlkLocation dest)
{
    return freeMessageBlockHelper(msgBlk, dest);
}


