iters=2000
lr=0.1
ADJ=False
npoints=100
seed=40
std=0.0
lrdecay=0.995
lba=0.08

# vanilla NODE
python spiral.py --niters $iters --viz --experiment_no 1 --seed $((seed+0)) --adjoint $ADJ --noise $std --lrdecay $lrdecay --lr $lr &
python spiral.py --niters $iters --viz --experiment_no 2 --seed $((seed+1)) --adjoint $ADJ --noise $std --lrdecay $lrdecay --lr $lr &
python spiral.py --niters $iters --viz --experiment_no 3 --seed $((seed+2)) --adjoint $ADJ --noise $std --lrdecay $lrdecay --lr $lr &

# NDO-NODE
python spiral.py --lrdecay $lrdecay --lstm --viz --lba $lba --lr $lr --adjoint $ADJ --noise $std --niters $iters --experiment_no 1 --seed $((seed+0)) &
python spiral.py --lrdecay $lrdecay --lstm --viz --lba $lba --lr $lr --adjoint $ADJ --noise $std --niters $iters --experiment_no 2 --seed $((seed+1)) &
python spiral.py --lrdecay $lrdecay --lstm --viz --lba $lba --lr $lr --adjoint $ADJ --noise $std --niters $iters --experiment_no 3 --seed $((seed+2)) &

for job in `jobs -p`; do wait ${job}; done

# plot loss curve
scale=100
iters=100
python draw.py --lba $lba --scale $scale --niters $iters --noise $std  --prefix ./${lr}_$lrdecay/lba_$lba/ --lr $lr --lrdecay $lrdecay