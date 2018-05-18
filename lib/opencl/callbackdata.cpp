#include "callbackdata.h"

callBackData::callBackData()
{

}

callBackData::callBackData(cl_uint param_NumCommand)
{
    listCommands = (clCommand**)malloc(param_NumCommand * sizeof (class clCommand *));
}

callBackData::~callBackData()
{
    free(listCommands);
}


