#ifndef LUXON_ITRANSPORT_CUH
#define LUXON_ITRANSPORT_CUH

#include <netinet/in.h>

#define PATTERN 0xFEED

enum class eMsgBlkLocation {HOST, DEVICE};
enum class eTransportType {UDP, RDMA_UD, PRINT, NONE, UNKNOWN};
enum class eTransportRole {SENSOR, PROCESSOR};

typedef struct
{
    int interval; //Number of us since last Message
    int bufferSize; //Size in bytes of the Message
    int seqNumber; //Position of the Message in the flow
    uint8_t buffer[MSG_MAX_SIZE];
} Message;

typedef struct
{
    int blockId; //used to map to other resources maintained by the transport.
    int msgCount;
    eMsgBlkLocation memLocation;
    Message* messages;
} MessageBlk;


class ITransport {

public:

    /*
     * Interface Methods
     */
    virtual int push(MessageBlk* msgBlk, int numMsg) = 0;
    virtual int pop(MessageBlk* msgBlk, int numReqMsg, int& numRetMsg) = 0;
    virtual int createMessageBlock(MessageBlk* msgBlk, eMsgBlkLocation dest) = 0;
    virtual int freeMessageBlock(MessageBlk* msgBlk, eMsgBlkLocation dest) = 0;

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

    static std::string toString(MessageBlk *m) {
        std::string rv;
        rv =  "blockId    :" + std::to_string(m->blockId) + "\n";
        rv += "msgCount   :" + std::to_string(m->msgCount) + "\n";
        //rv += "memLocation:" + std::to_string(static_cast<int>( m->memLocation)) + "\n";
        rv += "interval   :" + std::to_string(m->messages->interval) + "\n";
        rv += "bufferSize :" + std::to_string(m->messages->bufferSize) + "\n";
        rv += "seqNumber  :" + std::to_string(m->messages->seqNumber) + "\n";

        char bufferp[10];
        for(int i=0; i<5; i++) {
            sprintf(&bufferp[i*2], "%02hhX ", m->messages->buffer[i]);
        }
        rv += "Buffer[0:10]: " + std::string(bufferp) + "\n";
        
        return rv;
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

    int createMessageBlockHelper(MessageBlk* &msgBlk, eMsgBlkLocation dest) {
        npt("Debug Version %d", 4);
        std::size_t msgSize = sizeof(Message);
        msgBlk->msgCount = MSG_MAX_SIZE;
        msgBlk->blockId = 0;
        msgBlk->memLocation = dest;

        if (dest == eMsgBlkLocation::HOST) {
            msgBlk->messages = static_cast<Message *>(malloc(msgSize * MSG_BLOCK_SIZE));
        } else {
            //TODO : add code for device selection
            CUDA_CHECK(cudaMallocManaged((void **) &msgBlk->messages, msgSize * MSG_BLOCK_SIZE));
        }

        for (int i = 0; i < MSG_BLOCK_SIZE; i++) {
            msgBlk->messages[i].seqNumber = i;
            msgBlk->messages[i].interval = 0;
            msgBlk->messages[i].bufferSize = MSG_MAX_SIZE;
            memset(msgBlk->messages[i].buffer, PATTERN, MSG_MAX_SIZE);
        }
        return 0;
    }

    int freeMessageBlockHelper(MessageBlk* msgBlk, eMsgBlkLocation dest)
    {
        if (dest == eMsgBlkLocation::HOST) {
            free(msgBlk);
        } else {
            //TODO : add code for device selection
            CUDA_CHECK(cudaFree(msgBlk->messages));
        }
        return 0;
    }
};



#endif //LUXON_ITRANSPORT_CUH
