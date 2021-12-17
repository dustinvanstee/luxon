#include <cstdio>
#include <arpa/inet.h>
#include <iostream>

#include "print_transport.cuh"

PrintTransport::PrintTransport()
{
    this->transportType = eTransportType::PRINT;
}

int PrintTransport::push(Message* m)
{
    printMessage(m, 32);
    return 0;
}

int PrintTransport::pop(Message* msgBlk, int numReqMsg, int& numRetMsg)
{
    numRetMsg = numReqMsg;
    return 0;
}

int PrintTransport::createMessageBlock(Message* &msgBlk, eMsgBlkLocation dest)
{
    return createMessageBlockHelper(msgBlk, dest);
}

int PrintTransport::freeMessageBlock(Message* msgBlk, eMsgBlkLocation dest)
{
    return freeMessageBlockHelper(msgBlk, dest);
}



