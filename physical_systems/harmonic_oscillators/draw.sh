#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate pytorch
iters=2000
lr=0.01
ADJ=False
npoints=100
scale=1000
for lrdecay in 0.999 #0.99 0.995 0.998 0.999 1 # 0.995 # 0.99 0.995 0.999
do
    touch ./${lr}_$lrdecay/comp.txt
    echo "type method noise lba mean std" > ./${lr}_$lrdecay/comp.txt
    for lba in 0 0.0001 0.0002 0.0005 0.0008 0.001 0.002 0.005 0.008 0.01 0.02 0.05 0.08 0.1 0.2 0.5 0.8
    do
        for std in 0.0 0.01 0.03 0.05 0.1 0.3 0.5
        do  
            if [ -d "./${lr}_$lrdecay/lba_$lba/" ] 
            then
                echo ./${lr}_$lrdecay/lba_$lba/$std/
                python draw.py --lba $lba --scale $scale --niters $iters --noise $std  --prefix ./${lr}_$lrdecay/lba_$lba/ --lr $lr --lrdecay $lrdecay >> ./${lr}_$lrdecay/comp.txt &
            fi
        done
    done
    for job in `jobs -p`; do wait ${job}; done
    python gen_table.py --lr $lr --lrdecay $lrdecay
done