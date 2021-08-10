#!/bin/bash
iters=2000
lr=0.01
ADJ=False
lrdecay=0.999
std=0.0
lba=0.8

python make_data.py

python oscillators_node_o.py --niters $iters --experiment_no 1 --seed 40 --noise $std --lrdecay $lrdecay --lr $lr 
# python oscillators_node_o.py --niters $iters --experiment_no 2 --seed 41 --noise $std --lrdecay $lrdecay --lr $lr &
# python oscillators_node_o.py --niters $iters --experiment_no 3 --seed 42 --noise $std --lrdecay $lrdecay --lr $lr &

# python oscillators_node-lstm.py --lrdecay $lrdecay --lba $lba --lr $lr --adjoint $ADJ --noise $std --niters $iters --experiment_no 1 --seed 40 &
# python oscillators_node-lstm.py --lrdecay $lrdecay --lba $lba --lr $lr --adjoint $ADJ --noise $std --niters $iters --experiment_no 2 --seed 41 &
# python oscillators_node-lstm.py --lrdecay $lrdecay --lba $lba --lr $lr --adjoint $ADJ --noise $std --niters $iters --experiment_no 3 --seed 42 &

# for job in `jobs -p`; do wait ${job}; done

# scale=1000
# python draw.py --lba $lba --scale $scale --niters $iters --noise $std  --prefix ./${lr}_$lrdecay/lba_$lba/ --lr $lr --lrdecay $lrdecay