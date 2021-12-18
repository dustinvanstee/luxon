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
    int i = 0; //Some stuff to avoid getting this whole stack call optimized out.
    i++;
    return 0;
}

int NoneTransport::pop(MessageBlk* m, int numReqMsg, int& numRetMsg)
{
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


