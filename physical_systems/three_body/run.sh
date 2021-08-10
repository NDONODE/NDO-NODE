#!/bin/bash
iters=100
lr=0.1
ADJ=False
npoints=100
std=0.000
lba=0.0008

python ./three_body_problem.py --experiment_no 1 --seed 40  --niters $iters --adjoint $ADJ --npoints $npoints --noise $std &
python ./three_body_problem.py --experiment_no 2 --seed 41  --niters $iters --adjoint $ADJ --npoints $npoints --noise $std &
python ./three_body_problem.py --experiment_no 3 --seed 42  --niters $iters --adjoint $ADJ --npoints $npoints --noise $std &

python ./three_body_problem_basis_alone.py --experiment_no 1 --seed 40 --lr $lr --lba $lba --adjoint $ADJ --niters $iters --npoints $npoints --noise $std &
python ./three_body_problem_basis_alone.py --experiment_no 2 --seed 41 --lr $lr --lba $lba --adjoint $ADJ --niters $iters --npoints $npoints --noise $std &
python ./three_body_problem_basis_alone.py --experiment_no 3 --seed 42 --lr $lr --lba $lba --adjoint $ADJ --niters $iters --npoints $npoints --noise $std &

for job in `jobs -p`; do wait ${job}; done

python draw.py --lba $lba --noise $std --prefix ./$npoints/$lba/ 