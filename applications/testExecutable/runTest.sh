make install
cd D3Q27
source cleanCase.sh
testExecutable -GPU 0,1
fieldCalculate -calculationType containsNaN
cd ../
