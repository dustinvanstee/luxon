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

int NoneTransport::freeMessage(Message* m)
{
    free(m);
    return 0;
}


