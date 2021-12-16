//
// Created by alex on 8/7/20.
//

#ifndef LUXON_ITRANSPORT_CUH
#define LUXON_ITRANSPORT_CUH

#include <netinet/in.h>

enum class eTransportDest {HOST, DEVICE};
enum class eTransportType {UDP, RDMA_UD, PRINT, NONE, UNKNOWN};
enum class eTransportRole {SENSOR, PROCESSOR};

typedef struct
{
    int interval; //Number of us since last Message
    int bufferSize; //Size in bytes of the Message
    int seqNumber; //Position of the Message in the flow
    uint8_t buffer[MSG_MAX_SIZE];
} Message;


class ITransport {

public:

    /*
     * Interface Methods
     */
    virtual int push(Message* msgBlk) = 0;
    virtual int pop(Message* msgBlk, int numReqMsg, int& numRetMsg) = 0;
    virtual int freeMessage(Message* msgBlk) = 0;


    /*
    * Interface Statics
    */
    static eTransportType strToTransportType(const std::string& str)
    {
        if(str == "UDP") return eTransportType::UDP;
        else if(str == "RDMA_UD") return eTransportType::RDMA_UD;
        else if(str == "PRINT") return eTransportType::PRINT;
        else if(str == "NONE") return eTransportType::NONE;
        else return eTransportType::UNKNOWN;
    }

    static std::string TransportTypeToStr(eTransportType t)
    {
        switch(t)
        {
            case eTransportType::UDP:
                return "UDP Multicast";
            case eTransportType::PRINT:
                return "A Debug Mode - Prints the message on each push/pop";
            case eTransportType::NONE:
                return "A Debug Mode - Does nothing on push and pop";
            case eTransportType::RDMA_UD:
                return "RDMA Unreliable Datagram";
            default:
                return "transport unknown";
        }
    }

    static void printMessage(Message* m, int byteCount) {
        std::cout << "[Message Seq #: " << m->seqNumber << "\tsize: " << m->bufferSize << "\tintreval: " << m->interval << "]";

        int lastByte = m->bufferSize;

        if (byteCount != 0 && byteCount > 0 && byteCount <= m->bufferSize)
            lastByte = byteCount;

        for (int j = 0; (j < lastByte); j++) {
            // Start printing on the next after every 16 octets
            if ((j % 16) == 0)
                std::cout << std::endl;

            // Print each octet as hex (x), make sure there is always two characters (.2).
            //cout << std::setfill('0') << std::setw(2) << hex << (0xff & (unsigned int)buffer[j]) << " ";
            printf("%02hhX ", m->buffer[j]);
        }
        std::cout << std::endl;
    }

    /*
     * Accessor Methods
     */
    eTransportType getType()
    {
        return this->transportType;
    }

    std::string printType() {
       return this->TransportTypeToStr(this->transportType);
    }
    
    int createMessageBlock(Message* msgBlk, eTransportDest dest) { 
        std::size_t msgSize = sizeof(Message);
        if(dest == eTransportDest::HOST){
            msgBlk = static_cast<Message*>(malloc( msgSize*MSG_BLOCK_SIZE));
        } else {
            //TODO : add code for device selection
            CUDA_CHECK( cudaMallocManaged((void **)&msgBlk, msgSize*MSG_BLOCK_SIZE));
        }
     return 0;
    }
protected:
    //All Transports will use basic IPoX as a control plane to establish a connection.
    std::string                 s_mcastAddr;
    int                         n_mcastPort;
    std::string                 s_localAddr;
    int                         n_localPort;
    struct sockaddr_in			g_localAddr;
    struct sockaddr_in			g_mcastAddr;
    int                         sockfd;

    eTransportType              transportType;
};



#endif //LUXON_ITRANSPORT_CUH
