# csc-506-program1
ECE/CSC 506: Architecture of Parallel Computers

How to run:-

1. Plot 1

cd plot1
nano plot1.cu //to make changes to the value of OPT
//OPT==0 is base kernel
//OPT==1 is optimized kernel
make
make run > plot1.base.txt //for OPT==0
make clean

make
make run > plot1.opt.txt //for OPT==1

2. Plot 2
kindly follow the same structure as plot1 but in plot2 folder
