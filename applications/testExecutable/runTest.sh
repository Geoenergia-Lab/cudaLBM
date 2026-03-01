make install
cd D3Q27
# testExecutable
# source cleanCase.sh
clear
testExecutable -GPU 0,1
# fieldConvert -fileType vts -latestTime
cd ../
