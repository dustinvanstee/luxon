#include <cstdio>
#include <arpa/inet.h>
#include <iostream>

#include "udp_transport.cuh"

UdpTransport::UdpTransport(string localAddr, string mcastAddr, eTransportRole role) {
    this->transportType = eTransportType::RDMA_UD;

    s_localAddr = localAddr;
    s_mcastAddr = mcastAddr;
    n_mcastPort = 6655; //TODO: does this matter?
    n_localPort = 6655; //TODO: does this matter?

    bzero( &g_localAddr, sizeof( g_localAddr ) );
    bzero( &g_mcastAddr, sizeof( g_mcastAddr ) );

    // Creating socket file descriptor
       if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        cerr << "ERROR UdpTransport - Failed to create socket " << errno << endl;
        exit(EXIT_FAILURE);
    }
    DEBUG("Created local UDP socket: " << sockfd << endl);

    if(role == eTransportRole::SENSOR)
    {
        memset((char *) &g_mcastAddr, 0, sizeof(g_mcastAddr));
        g_mcastAddr.sin_family = AF_INET;
        g_mcastAddr.sin_addr.s_addr = inet_addr(s_mcastAddr.c_str());
        g_mcastAddr.sin_port = htons(n_mcastPort);

        char loopch=0;

        if (setsockopt(sockfd, IPPROTO_IP, IP_MULTICAST_LOOP,
                (char *)&loopch,
                sizeof(loopch)) < 0) {
            cerr << "ERROR UdpTransport - disable loop back failed set opt " << errno << endl;
            close(sockfd);
            exit(EXIT_FAILURE);
        }

        inet_pton(AF_INET, s_localAddr.c_str(), &this->g_localAddr.sin_addr);
        if (setsockopt(sockfd, IPPROTO_IP, IP_MULTICAST_IF,
                       (char *)&g_localAddr,
                       sizeof(g_localAddr)) < 0) {
            cerr << "ERROR UdpTransport - Setting up local interface " << strerror(errno) << endl;
            exit(1);
        }

    } else if(role == eTransportRole::PROCESSOR) {

        {
            int reuse=1;

            if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR,
                           (char *)&reuse, sizeof(reuse)) < 0) {
                cerr << "ERROR UdpTransport - set option to allow socket reuse " << errno << endl;
                close(sockfd);
                exit(EXIT_FAILURE);
            }
        }

        //Create the SockAddr for the Local System
        memset(&g_localAddr, 0, sizeof(g_localAddr));
        g_localAddr.sin_family = AF_INET;
        g_localAddr.sin_port = htons(n_localPort);
        g_localAddr.sin_addr.s_addr = INADDR_ANY;

        cout << "Bind the socket local address: " << s_localAddr << endl;
        if (bind(sockfd, (const struct sockaddr *) &g_localAddr,
                 sizeof(g_localAddr)) < 0) {
            cerr << "ERROR UdpTransport - failed to bind to local socket " << errno << endl;
            exit(EXIT_FAILURE);
        }

        if (inet_pton(AF_INET, s_mcastAddr.c_str(), &mcastGroup.imr_multiaddr.s_addr) != 1) {
            cerr << "ERROR inet pton can't convert address " << errno << endl;
            close(sockfd);
            exit(EXIT_FAILURE);
        }
        if (inet_pton(AF_INET, s_localAddr.c_str(), &mcastGroup.imr_interface.s_addr) != 1) {
            cerr << "ERROR inet pton can't convert address " << errno << endl;
            close(sockfd);
            exit(EXIT_FAILURE);
        }

        if (setsockopt(sockfd, IPPROTO_IP, IP_ADD_MEMBERSHIP,
                       (char *)&mcastGroup, sizeof(mcastGroup)) < 0) {
            cerr << "ERROR UdpTransport - couldn't join mcast group " << errno << endl;
            close(sockfd);
            exit(EXIT_FAILURE);
        }


    }

}

int UdpTransport::push(MessageBlk* msgBlk, int numMsg)
{
    for(int i = 0; i < numMsg; i++) {
        if (sendto(sockfd, (const char *) msgBlk->messages[i].buffer, msgBlk->messages[i].bufferSize, 0,
                   (const struct sockaddr *) &this->g_mcastAddr,
                   sizeof(this->g_mcastAddr)) <= 0) {
            cerr << "ERROR UdpTransport Push - failed sendto operation " << errno << endl;
            close(sockfd);
            exit(EXIT_FAILURE);
        }
        DEBUG("To " << inet_ntoa(g_mcastAddr.sin_addr) << endl);
#if DEBUG_BUILD > 0
        printMessage(&msgBlk->messages[i], 32);
#endif
    }
    return 0;
}

int UdpTransport::pop(MessageBlk* msgBlk, int numReqMsg, int& numRetMsg)
{
    int rc;
    sockaddr *name = NULL;
    socklen_t *namelen = NULL;
    uint8_t buffer[MSG_MAX_SIZE];    // receive buffer

    npt("waiting on socket %d\n",this->n_localPort);

    for(int i = 0; i < numReqMsg; i++)
    {
        rc = recvfrom(this->sockfd, &buffer, MSG_MAX_SIZE, 0, name, namelen);

        if (rc > 0) {
            msgBlk->messages[i].seqNumber = i;
            msgBlk->messages[i].interval = 0;
            msgBlk->messages[i].bufferSize = rc;
            memcpy(msgBlk->messages[i].buffer,buffer,rc); //TODO: smarter way than a copy?
            numRetMsg = numRetMsg + 1;
        } else if(rc == -1) {
            cerr << "ERROR UdpTransport Pop - failed mcast socket read " << errno << endl;
            close(sockfd);
            exit(EXIT_FAILURE);
        }


    }

    return 0;
}

int UdpTransport::createMessageBlock(MessageBlk* msgBlk, eMsgBlkLocation dest)
{
    return createMessageBlockHelper(msgBlk, dest);
}

int UdpTransport::freeMessageBlock(MessageBlk* msgBlk, eMsgBlkLocation dest)
{
    return freeMessageBlockHelper(msgBlk, dest);
}