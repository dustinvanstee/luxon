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

int PrintTransport::pop(Message** m, int numReqMsg, int& numRetMsg, eTransportDest dest)
{
    int recvlen;                        // num bytes received

    for(int i = 0; i < numReqMsg; i++) {
        recvlen = MSG_MAX_SIZE;

        if (recvlen > 0) {
            m[i] = createMessage();
            m[i]->seqNumber = i;
            m[i]->interval = 0;
            m[i]->bufferSize = recvlen;
            memset(&m[i]->buffer, PATTERN, MSG_MAX_SIZE);
            numRetMsg = numRetMsg + 1;
        }
    }

    return 0;
}
int PrintTransport::pop(Message* m, int numReqMsg, int& numRetMsg, eTransportDest dest){
    npt("new pop implementation%d\n",0);
    return 0;
}

int PrintTransport::pop(int a){
    npt("new pop implementation %d\n",a );
    return 0;
}
Message* PrintTransport::createMessage() {
    std::size_t t = sizeof(Message);
    auto* m = static_cast<Message*>(malloc(t));
    return m;
}

int PrintTransport::freeMessage(Message* m)
{
    free(m);
    return 0;
}


