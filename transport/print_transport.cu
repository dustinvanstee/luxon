#include <cstdio>
#include <arpa/inet.h>
#include <iostream>

#include "print_transport.cuh"

PrintTransport::PrintTransport()
{
    this->transportType = eTransportType::PRINT;
}

int PrintTransport::push(MessageBlk* m, int msg_blk_size)
{
    for(int i = 0; i < msg_blk_size; i++)
    {
        printMessage(&m->messages[i], 32);
    }
    return 0;
}

int PrintTransport::pop(MessageBlk* msgBlk, int numReqMsg, int& numRetMsg)
{
    numRetMsg = numReqMsg;
    return 0;
}

int PrintTransport::createMessageBlock(MessageBlk* msgBlk, eMsgBlkLocation dest)
{
    return createMessageBlockHelper(msgBlk, dest);
}

int PrintTransport::freeMessageBlock(MessageBlk* msgBlk, eMsgBlkLocation dest)
{
    return freeMessageBlockHelper(msgBlk, dest);
}



