#include <cstdio>
#include <arpa/inet.h>
#include <iostream>

#include "none_transport.cuh"

NoneTransport::NoneTransport()
{
    this->transportType = eTransportType::NONE;
}

int NoneTransport::push(Message** m, int count)
{
    return 0;
}

int NoneTransport::pop(Message** m, int numReqMsg, int& numRetMsg, eTransportDest dest)
{
    return 0;
}

Message* NoneTransport::createMessage() {
    std::size_t t = sizeof(Message);
    auto* m = static_cast<Message*>(malloc(t));
    return m;
}

int NoneTransport::freeMessage(Message* m)
{
    free(m);
    return 0;
}


