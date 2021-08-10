expn=1
stif=1000

# steer
python stiff_sin.py --viz --data_size 120 --version steer --min_length 0.001 --stiffness_ratio $stif --niters 8000 &

# vanilla
python stiff_sin.py --viz --data_size 120 --version standard --stiffness_ratio $stif --niters 8000 &

# NDO-NODE
lba=0.4
python stiff_sin.py --viz --data_size 120 --version standard --stiffness_ratio $stif --niters 8000 --lstm --lba $lba --experiment_no $expn &

# RNODE
lba=0.001
python stiff_sin.py --viz --data_size 120 --version standard --stiffness_ratio $stif --niters 8000 --lstm --L2 --lba $lba --experiment_no $expn &


for job in `jobs -p`; do wait ${job}; done
