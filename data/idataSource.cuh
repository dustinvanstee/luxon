#ifndef LUXON_IDATASOURCE_CUH
#define LUXON_IDATASOURCE_CUH
#include "../transport/itransport.cuh"

enum class eDataSourceType {PCAP, RANDOM, FINANCE, PAT, UNKNOWN};

class IDataSource {

public:

    /*
    * Interface Statics
    */
    static eDataSourceType strToDataSourceType(const std::string& str)
    {
        if(str == "PCAP") return eDataSourceType::PCAP;
        else if(str == "RANDOM") return eDataSourceType::RANDOM;
        else if(str == "PATTERN") return eDataSourceType::PAT;
        else if(str == "FINANCE") return eDataSourceType::FINANCE;
        else return eDataSourceType::UNKNOWN;
    }

    static std::string DataSourceTypeToStr(eDataSourceType t)
    {
        switch(t)
        {
            case eDataSourceType::PCAP:
                return "PCAP: Packet payloads parced from a PCAP File";
            case eDataSourceType::RANDOM:
                return "RANDOM: Randomly generated bytes";
            case eDataSourceType::PAT:
                return "PATTERN: Pattern of bytes 0xFA";
            case eDataSourceType::FINANCE:
                return "FINANCE: Sample financial instrument struct";
            default:
                return "data source unknown";
        }
    }

   /*
   * Virtual Interface for Data buffer - print message represenation.
   */
    virtual void summarizeMessage(Message* m) = 0;

    /*
     * Accessor Methods
     */
    eDataSourceType getType()
    {
        return this->dataSourceType;
    }

    std::string printType() {
       return this->DataSourceTypeToStr(this->dataSourceType);
    }

protected:
    eDataSourceType              dataSourceType;
};



#endif //LUXON_IDATASOURCE_CUH
