//
// Created by alex on 7/15/20.
//

#include "Processor.cuh"

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__ void gpu_count_zeros(Message** flow, int* sum, int flowLength)
{
    int indx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = indx; i < flowLength; i += stride)
    {
        for(int j = 0; j < flow[i]->bufferSize; j++)
        {
            if(flow[i]->buffer[j] == 0)
            {
                sum[i] += 1;
                //cout << "found a zero at msg[" << i << "] byte[" << j << "]" << endl;
            }
        }
    }
}


void cpu_count_zeros(Message** flow, int& sum, int flowLength)
{
    for(int i = 0; i < flowLength; i++)
    {
        for(int j = 0; j < flow[i]->bufferSize; j++)
        {
            if(flow[i]->buffer[j] == 0)
            {
                sum += 1;
                //cout << "found a zero at msg[" << i << "] byte[" << j << "]" << endl;
            }
        }
    }
}


Processor::Processor(ITransport* t) {
    transport = t;
}

void Processor::procCountZerosGPU(int minMessageToProcess) {
    timer t;

    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    size_t threadsPerBlock;
    size_t numberOfBlocks;

    threadsPerBlock = 256;
    numberOfBlocks = 32 * numberOfSMs;

    int msgCountReturned = 0;
    int processedMessages = 0;
    int sum =0;

    Message* m[MSG_BLOCK_SIZE];//Create array that is max message block size
    uint8_t * d; //The data we will store.
    size_t msgBlockSize = MSG_BLOCK_SIZE * sizeof(Message);
    size_t msgDataSize = MSG_MAX_SIZE * MSG_BLOCK_SIZE;
    checkCuda( cudaMallocManaged(m, msgBlockSize));
    checkCuda( cudaMallocManaged(&d, msgDataSize));

    int* blockSum;   //Array with sum of zeros for this message
    size_t sumArraySize = MSG_BLOCK_SIZE * sizeof(int);
    checkCuda( cudaMallocManaged(&blockSum, sumArraySize));
   // cout << "Processing on GPU using " <<  numberOfBlocks << " blocks with " << threadsPerBlock << " threads per block" << endl;

    while (processedMessages < minMessageToProcess) {

        if (0 != transport->pop(m, MSG_BLOCK_SIZE, msgCountReturned, eTransportDest::DEVICE)) {
            exit(EXIT_FAILURE);
        }

        cudaMemPrefetchAsync(m, msgBlockSize, deviceId);

        if(msgCountReturned > 0) //If there are new messages process them
        {
            std::cerr << "\rProcessed " << processedMessages << " messages";
            gpu_count_zeros <<< threadsPerBlock, numberOfBlocks >>>(m, blockSum, msgCountReturned);

            checkCuda( cudaGetLastError() );
            checkCuda( cudaDeviceSynchronize() ); //Wait for GPU threads to complete

            cudaMemPrefetchAsync(blockSum, sumArraySize, cudaCpuDeviceId);

            for(int k = 0; k < msgCountReturned; k++)
            {
                sum += blockSum[k]; //Add all the counts to the accumulator
                blockSum[k] = 0;
            }

            processedMessages += msgCountReturned;
        }
        //m.clear();
        msgCountReturned=0;

    }

    checkCuda( cudaFree(m));
    checkCuda( cudaFree(blockSum));

    std::cout << "\n Processing Completed: " << std::endl;
    std::cout << "\t processed " << processedMessages << " in " << t.seconds_elapsed() << " sec" << std::endl;
    std::cout << "\t total zero's in messages = " << sum << std::endl;

    exit(EXIT_SUCCESS);
}

int Processor::procCountZerosCPU(int minMessageToProcess) {
    timer t;

    Message* m[MSG_BLOCK_SIZE];
    int msgCountReturned = 0;
    int sum = 0;
    int processedMessages = 0;

    //Intitialize the receive buffer
    for(int i = 0; i < MSG_BLOCK_SIZE; i++)
    {
        m[i] = transport->createMessage();
    }

    while (processedMessages < minMessageToProcess) {

        if (0 != transport->pop(m, MSG_BLOCK_SIZE, msgCountReturned, eTransportDest::HOST)) {
            exit(EXIT_FAILURE);
        }

        if(msgCountReturned > 0) //If there are new messages process them
        {
            std::cerr << "\rProcessed " << processedMessages << " messages";
            cpu_count_zeros(m, sum, msgCountReturned);
            processedMessages += msgCountReturned;
        }
        msgCountReturned=0;

    }

    //Free the receive buffer
    for(int i = 0; i < MSG_BLOCK_SIZE; i++)
    {
        transport->freeMessage(m[i]);
    }

    std::cout << "\nProcessing Completed: " << std::endl;
    std::cout << "\t processed " << processedMessages << " in " << t.seconds_elapsed() << " sec" << std::endl;
    std::cout << "\t total zero's in messages = " << sum << std::endl;
    exit(EXIT_SUCCESS);
}

void Processor::procDropMsg(int minMessageToProcess) {
    timer t;

    Message* m[MSG_BLOCK_SIZE];
    int msgCountReturned = 0;
    int processedMessages = 0;

    //Intitialize the receive buffer
    for(int i = 0; i < MSG_BLOCK_SIZE; i++)
    {
        m[i] = transport->createMessage();
    }

    t.start();
    while (processedMessages < minMessageToProcess) {

        if (0 != transport->pop(m, MSG_BLOCK_SIZE, msgCountReturned, eTransportDest::HOST)) {
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

    //Free the receive buffer
    for(int i = 0; i < MSG_BLOCK_SIZE; i++)
    {
        transport->freeMessage(m[i]);
    }

    std::cout << "\nProcessing Completed: " << std::endl;
    std::cout << "\t processed " << processedMessages << " in " << t.seconds_elapsed() << " sec" << std::endl;
    exit(EXIT_SUCCESS);
}

int Processor::procPrintMessages(int minMessageToProcess) {
    Message* m[MSG_BLOCK_SIZE];
    int processedCount = 0;
    int r = 0;

    //Intitialize the receive buffer
    for(int i = 0; i < MSG_BLOCK_SIZE; i++)
    {
        m[i] = transport->createMessage();
    }

    do {

        if (0 != transport->pop(m, MSG_BLOCK_SIZE, r, eTransportDest::HOST)) {
            exit(EXIT_FAILURE);
        }

        processedCount += r;

        std::cout << "Printing first bytes of " << min(r,minMessageToProcess) << " messages" << std::endl;
        for(int i = 0; i<min(r,minMessageToProcess); i++)
        {
            transport->printMessage(m[i], 32);
            std::cout << std::endl;
        }
    } while (processedCount < minMessageToProcess);

    //Free the receive buffer
    for(int i = 0; i < MSG_BLOCK_SIZE; i++)
    {
        transport->freeMessage(m[i]);
    }

    //Simple process (i.e. print)
    std::cout << "Processing Completed: found " << processedCount << " messages" << std::endl;
    exit(EXIT_SUCCESS);
}
