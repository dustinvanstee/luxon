#include "Processor.cuh"
#include "../data/data_sample_finance.cuh"
#include "../data/data_source_random.cuh"
#include "../data/data_source_pattern.cuh"

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

Processor::Processor(ITransport* t,  eDataSourceType dataSourceType, eMsgBlkLocation msgBlockLocation ) {
    this->transport = t;
    this->dataSourceType = dataSourceType;

    switch(dataSourceType) {
        case eDataSourceType::PCAP :
            break;
        case eDataSourceType::RANDOM :
            dataSource = new RandomData();
            break;
        case eDataSourceType::FINANCE :
            dataSource = new MarketData();
            break;
        case eDataSourceType::PAT :
            dataSource = new PatternData();
            break;
    }

    //Allocate teh Message Block based on the transport and the target for processing
    if(0 != transport->createMessageBlock(&this->msgBlk, msgBlockLocation)){
        // TODO: Error handling
    }
}

Processor::~Processor() {
    //Free the Message Block Memory
    if(0 != transport->freeMessageBlock(&this->msgBlk, this->msgBlockLocation))
    {
        // TODO: Error handling
    }
}

int Processor::procCountZerosGPU(int minMessageToProcess) {
    timer t;

    int deviceId;
    int numberOfSMs;

    CUDA_CHECK(cudaGetDevice(&deviceId));
    CUDA_CHECK(cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId));

    size_t threadsPerBlock;
    size_t numberOfBlocks;

    threadsPerBlock = 256;
    numberOfBlocks = 32 * numberOfSMs;

    int msgCountReturned = 0;
    int processedMessages = 0;
    int sum = 0;

    int *blockSum;   //Array with sum of zeros for this message
    size_t sumArraySize = MSG_BLOCK_SIZE * sizeof(int);
    CUDA_CHECK(cudaMallocManaged(&blockSum, sumArraySize));
    // cout << "Processing on GPU using " <<  numberOfBlocks << " blocks with " << threadsPerBlock << " threads per block" << endl;
    t.start();
    while (processedMessages < minMessageToProcess) {

        if (0 != transport->pop(&this->msgBlk, MSG_BLOCK_SIZE, msgCountReturned)) {
            exit(EXIT_FAILURE);
        }

       CUDA_CHECK(cudaMemPrefetchAsync(&msgBlk.messages, MSG_BLOCK_SIZE * sizeof(Message), deviceId));

        if (msgCountReturned > 0) //If there are new messages process them
        {
            gpu_count_zeros <<< numberOfBlocks, threadsPerBlock >>>(msgBlk.messages, blockSum, msgCountReturned);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize()); //Wait for GPU threads to complete

            CUDA_CHECK(cudaMemPrefetchAsync(blockSum, sumArraySize, cudaCpuDeviceId));

            for (int k = 0; k < msgCountReturned; k++) {
                sum += blockSum[k]; //Add all the counts to the accumulator
                blockSum[k] = 0;
            }

            processedMessages += msgCountReturned;
            npt("GPU Loop processing complete.  Processed %d messages.\n", processedMessages);

        }
        //m.clear();
        msgCountReturned = 0;
    }
    t.stop();

    CUDA_CHECK(cudaFree(blockSum));

    pt("\n Processing Completed%c \n", ':');
    std::cout << "\t processed " << processedMessages << " in " << t.usec_elapsed() << " usec" << std::endl;
    std::cout << "\t total zero's in messages = " << sum << std::endl;

    return 0;
}

int Processor::procCountZerosCPU(int minMessageToProcess) {
    timer t;

    int msgCountReturned = 0;
    int sum = 0;
    int processedMessages = 0;

    t.start();
    while (processedMessages < minMessageToProcess) {

        if (0 != transport->pop(&this->msgBlk, MSG_BLOCK_SIZE, msgCountReturned)) {
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


    std::cout << "\nProcessing Completed: " << std::endl;
    std::cout << "\t processed " << processedMessages << " in " << t.usec_elapsed() << " usec" << std::endl;
    std::cout << "\t total zero's in messages = " << sum << std::endl;
    return 0;
}

void Processor::procDropMsg(int minMessageToProcess) {
    timer t;

    int msgCountReturned = 0;
    int processedMessages = 0;

    t.start();
    while (processedMessages < minMessageToProcess) {

        if (0 != transport->pop(&this->msgBlk, MSG_BLOCK_SIZE, msgCountReturned)) {
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
    int processedCount = 0;
    int messagesReturned = 0;

    do {

        if (0 != transport->pop(&this->msgBlk, MSG_BLOCK_SIZE, messagesReturned)) {
            exit(EXIT_FAILURE);
        }

        processedCount += messagesReturned;

        std::cout << "Printing first bytes of " << min(messagesReturned,minMessageToProcess) << " messages" << std::endl;
        for(int i = 0; i<min(messagesReturned,minMessageToProcess); i++)
        {
            transport->printMessage(&msgBlk.messages[i], 32);
            std::cout << std::endl;
            this->dataSource->summarizeMessage(&msgBlk.messages[i]);
        }
    } while (processedCount < minMessageToProcess);


    //Simple process (i.e. print)
    std::cout << "Processing Completed: found " << processedCount << " messages" << std::endl;
    exit(EXIT_SUCCESS);
}
