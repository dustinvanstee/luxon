#include "Processor.cuh"
#include "../data/data_sample_finance.cuh"

//inline cudaError_t chceckCuda(cudaError_t result)
//{
//    if (result != cudaSuccess) {
//        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
//        assert(result == cudaSuccess);
//    }
//    return result;
//}

__global__ void gpu_count_zeros(Message* flow, int* sum, int flowLength)
{
    int indx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = indx; i < flowLength; i += stride)
    {
        for(int j = 0; j < flow[i].bufferSize; j++)
        {
            if(flow[i].buffer[j] == 0)
            {
                sum[i] += 1;
                //cout << "found a zero at msg[" << i << "] byte[" << j << "]" << endl;
            }
        }
    }
}




void cpu_count_zeros(Message* flow, int& sum, int flowLength)
{
    for(int i = 0; i < flowLength; i++)
    {
        for(int j = 0; j < flow[i].bufferSize; j++)
        {
            if(flow[i].buffer[j] == 0)
            {
                sum += 1;
                //cout << "found a zero at msg[" << i << "] byte[" << j << "]" << endl;
            }
        }
    }
}

void Processor::initializeMsgBlk() {
   this->msgBlkPtr=  static_cast<MessageBlk *>(malloc(sizeof(MessageBlk))); 
   if(transport->createMessageBlock(msgBlkPtr, eMsgBlkLocation::DEVICE)){
        // print error messge
        fprintf(stderr, "Processor::initializeMsgBlk(): memory allocation error\n");
        exit(EXIT_FAILURE); 
    } 
}


Processor::Processor(ITransport* t,  eDataSourceType dataSourceType) {
    this->transport = t;
    this->dataSourceType = dataSourceType;
}

int Processor::procCountZerosGPU(int minMessageToProcess) {
    timer t;

    int deviceId;
    int numberOfSMs;

    CUDA_CHECK( cudaGetDevice(&deviceId));
    CUDA_CHECK( cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId));

    size_t threadsPerBlock;
    size_t numberOfBlocks;

    threadsPerBlock = 256;
    numberOfBlocks = 32 * numberOfSMs;

    int msgCountReturned = 0;
    int processedMessages = 0;
    int sum =0;

    this->initializeMsgBlk();

    int* blockSum;   //Array with sum of zeros for this message
    size_t sumArraySize = MSG_BLOCK_SIZE * sizeof(int);
    CUDA_CHECK( cudaMallocManaged(&blockSum, sumArraySize));
   // cout << "Processing on GPU using " <<  numberOfBlocks << " blocks with " << threadsPerBlock << " threads per block" << endl;
    t.start();
    while (processedMessages < minMessageToProcess) {

        if (0 != transport->pop(msgBlkPtr, MSG_BLOCK_SIZE, msgCountReturned)) {
            exit(EXIT_FAILURE);
        }

        CUDA_CHECK(cudaMemPrefetchAsync(this->msgBlkPtr->messages, MSG_BLOCK_SIZE*sizeof(Message), deviceId));

        if(msgCountReturned > 0) //If there are new messages process them
        {
            gpu_count_zeros <<< numberOfBlocks, threadsPerBlock >>>(this->msgBlkPtr->messages, blockSum, msgCountReturned);

            CUDA_CHECK( cudaGetLastError() );
            CUDA_CHECK( cudaDeviceSynchronize() ); //Wait for GPU threads to complete

            CUDA_CHECK(cudaMemPrefetchAsync(blockSum, sumArraySize, cudaCpuDeviceId));

            for(int k = 0; k < msgCountReturned; k++)
            {
                sum += blockSum[k]; //Add all the counts to the accumulator
                blockSum[k] = 0;
            }

            processedMessages += msgCountReturned;
            npt("GPU Loop processing complete.  Processed %d messages.\n", processedMessages );

        }
        //m.clear();
        msgCountReturned=0;
    }
    t.stop();
   
    CUDA_CHECK( cudaFree(blockSum));

    pt( "\n Processing Completed%c \n", ':');
    std::cout << "\t processed " << processedMessages << " in " << t.usec_elapsed() << " usec" << std::endl;
    std::cout << "\t total zero's in messages = " << sum << std::endl;
    return 0;
}

int Processor::procCountZerosCPU(int minMessageToProcess) {
    timer t;

    int msgCountReturned = 0;
    int sum = 0;
    int processedMessages = 0;

    //Intitialize the receive buffer
    MessageBlk msgBlk;
    MessageBlk* pmsgBlk = &msgBlk; //This is hacky, probably better to refactor createMessageblcok to pass by reference.

    if(!transport->createMessageBlock(pmsgBlk, eMsgBlkLocation::HOST)){
        // print error messge
    }
    t.start();
    while (processedMessages < minMessageToProcess) {

        if (0 != transport->pop(pmsgBlk, MSG_BLOCK_SIZE, msgCountReturned)) {
            exit(EXIT_FAILURE);
        }

        if(msgCountReturned > 0) //If there are new messages process them
        {
            std::cerr << "\rProcessed " << processedMessages << " messages";
            cpu_count_zeros(msgBlk.messages, sum, msgCountReturned);
            processedMessages += msgCountReturned;
        }
        msgCountReturned=0;

    }
    t.stop();
    //Free the receive buffer
    for(int i = 0; i < MSG_BLOCK_SIZE; i++)
    {
        transport->freeMessageBlock(pmsgBlk,eMsgBlkLocation::HOST) ;
    }

    std::cout << "\nProcessing Completed: " << std::endl;
    std::cout << "\t processed " << processedMessages << " in " << t.usec_elapsed() << " usec" << std::endl;
    std::cout << "\t total zero's in messages = " << sum << std::endl;
    return 0;
}

void Processor::procDropMsg(int minMessageToProcess) {
    timer t;

    MessageBlk msgBlk;
    MessageBlk* pmsgBlk = &msgBlk; //This is hacky, probably better to refactor createMessageblcok to pass by reference.

    if(!transport->createMessageBlock(pmsgBlk, eMsgBlkLocation::HOST)){
        exit(EXIT_FAILURE);
    }

    int msgCountReturned = 0;
    int processedMessages = 0;

    t.start();
    while (processedMessages < minMessageToProcess) {

        if (0 != transport->pop(pmsgBlk, MSG_BLOCK_SIZE, msgCountReturned)) {
            exit(EXIT_FAILURE);
        }

        if(msgCountReturned > 0) //If there are new messages process them
        {
            std::cerr << "\rProcessed " << processedMessages << " messages";
            processedMessages += msgCountReturned;
        }
        msgCountReturned=0;

    }
    t.stop();

    std::cout << "\nProcessing Completed: " << std::endl;
    std::cout << "\t processed " << processedMessages << " in " << t.usec_elapsed() << " usec" << std::endl;
    exit(EXIT_SUCCESS);
}

int Processor::procPrintMessages(int minMessageToProcess) {

    MessageBlk msgBlk;
    MessageBlk* pmsgBlk = &msgBlk; //This is hacky, probably better to refactor createMessageblcok to pass by reference.
    if(!transport->createMessageBlock(pmsgBlk, eMsgBlkLocation::HOST)){
        exit(EXIT_FAILURE);
    }
    
    int processedCount = 0;
    int messagesReturned = 0;

    do {

        if (0 != transport->pop(pmsgBlk, MSG_BLOCK_SIZE, messagesReturned)) {
            exit(EXIT_FAILURE);
        }

        processedCount += messagesReturned;

        std::cout << "Printing first bytes of " << min(messagesReturned,minMessageToProcess) << " messages" << std::endl;
        for(int i = 0; i<min(messagesReturned,minMessageToProcess); i++)
        {
            transport->printMessage(&msgBlk.messages[i], 32);
            std::cout << std::endl;
        }
    } while (processedCount < minMessageToProcess);


    //Simple process (i.e. print)
    std::cout << "Processing Completed: found " << processedCount << " messages" << std::endl;
    exit(EXIT_SUCCESS);
}

int Processor::freeMemory() {
    //TODO : could refactor freeMessageBlock just to take message block ptr, as the structure contains memlocation..
    int rv {0};
    eMsgBlkLocation location {this->msgBlkPtr->memLocation};
    this->transport->freeMessageBlock(this->msgBlkPtr, location); 
    return rv;
}

// Goal of this function is to render buffer in fairly agnostic way using data source implementations...
void Processor::summarizeBuffer() {
    /* 
    1. get point to current message buffer
    2. send buffer to data class to render buffer in a formatted way.  Good for checking and making sure things work properly
    */

    // msgBlkPtr is my message buffer
    //if finance 
    if(this->dataSourceType == eDataSourceType::FINANCE) {
        MarketData marketData;
        marketData.summarizeBuffer(this->msgBlkPtr);
    } else {
        fprintf(stderr, "summarizeBuffer Not implemented for %s!\n", IDataSource::DataSourceTypeToStr(this->dataSourceType).c_str());
        exit(EXIT_FAILURE);
    }
}
