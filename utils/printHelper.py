from utils.Logger import logger
import logging

##
# @brief print means value, std value and item names
#
# @param meanList: example [1.1, 2.2, 1.2]
# @param stdList:  example [0.1, 0.15, 0.001]
# @param itemList: example ['acc', 'pre', 'recall', 'fsc']
# @param debugLevel: example logging.INFO
#
# @return 
def printMeanStd(meanList, stdList, itemList, debugLevel=logging.INFO):
    # select logging function
    loggingFunc = None
    if (debugLevel == logging.INFO):
        loggingFunc = logger.info
    elif (debugLevel == logging.DEBUG):
        loggingFunc = logger.debug
    else:
        raise NotImplementedError("Not supported logging level.")

    lengthSet = set()
    lengthSet.add(len(meanList))
    lengthSet.add(len(stdList))
    lengthSet.add(len(itemList))
    if not len(lengthSet) == 1:
        logger.error("Inconsisten list lengths when printing statistics.")
        exit(1)

    for mean, std, item in zip(meanList, stdList, itemList):
        loggingFunc("%-6s: %f +- %f", item, mean, std)


##
# @brief print means value and item names
#
# @param meanList: example [1.1, 2.2, 1.2]
# @param itemList: example ['acc', 'pre', 'recall', 'fsc']
# @param debugLevel: example logging.INFO
#
# @return 
def printMean(meanList, itemList, debugLevel=logging.INFO):
    # select logging function
    loggingFunc = None
    if (debugLevel == logging.INFO):
        loggingFunc = logger.info
    elif (debugLevel == logging.DEBUG):
        loggingFunc = logger.debug
    else:
        raise NotImplementedError("Not supported logging level.")

    lengthSet = set()
    lengthSet.add(len(meanList))
    lengthSet.add(len(itemList))
    if not len(lengthSet) == 1:
        logger.error("Inconsisten list lengths when printing statistics.")
        exit(1)

    for mean, item in zip(meanList, itemList):
        loggingFunc("%-6s: %f", item, mean)
