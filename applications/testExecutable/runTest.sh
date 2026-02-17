make install
cd D3Q27
source cleanCase.sh
testExecutable -GPU 0,1
fieldConvert -fileType vts -latestTime
cd ../
