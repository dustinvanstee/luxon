#ifndef LUXON_COMMON_CUH
#define LUXON_COMMON_CUH

#include <iostream>
#include <chrono>

//#define DEBUG_BUILD

#ifdef DEBUG_BUILD
#define DEBUG(x) std::cerr << x
#define DEBUG_DETAIL(x) x
#else
#  define DEBUG(x) do {} while (0)
#  define DEBUG_DETAIL(x) do {} while (0)
#endif

#define PRINT_UPDATE_DELAY 1    //Used with timer

#define MSG_MAX_SIZE 2000       //Max size of a message must be > RAND_FLOW_MSG_SIZE or max size message from pcap
#define MSG_BLOCK_SIZE 10     //Number of messages to process in parallel

class timer
{
    typedef std::chrono::steady_clock clock ;
    typedef std::chrono::seconds seconds ;

public:
    void reset() {
        start_time = clock::now() ;
    }
    void start() {
        start_time = clock::now() ;
    }
    void stop() {
        stop_time = clock::now() ;
    }

    unsigned long long seconds_elapsed() const
    {
        return std::chrono::duration_cast<seconds>( stop_time - start_time ).count() ;
    }

    unsigned long long usec_elapsed() const
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>( stop_time - start_time ).count() ;
    }

private:
    clock::time_point start_time = clock::now() ;
    clock::time_point stop_time = clock::now() ;
};


#endif //LUXON_COMMON_CUH
