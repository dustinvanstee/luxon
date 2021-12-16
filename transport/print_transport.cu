#include <cstdio>
#include <arpa/inet.h>
#include <iostream>

#include "print_transport.cuh"

#define PATTERN 0xFEED

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

int PrintTransport::freeMessage(Message* m)
{
    free(m);
    return 0;
}


