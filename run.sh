python train_save.py --dataset cifar --model cnn --num_users 10 --epochs 30 --dirichlet 1.0 --seed 4

python main_Exact_Shapley.py --num_users 10 --epochs 30 --model cnn --dataset cifar --local_ep 1 --seed 4

python main_CS_Shapley_parallel_final.py --num_users 10 --epochs 30 --model cnn --dataset cifar --local_ep 1 --seed 4 --selection_ratio 0.9 --overlook_ratio 0.8

# python main_CS_Shapley_parallel_final.py --num_users 10 --epochs 30 --model cnn --dataset cifar --local_ep 1 --seed 4 --selection_ratio 0.9 --overlook_ratio 0.6 # &> /dev/null &

python main_CS_Shapley_parallel_final.py --num_users 10 --epochs 30 --model cnn --dataset cifar --local_ep 1 --seed 4 --selection_ratio 0.8 --overlook_ratio 0.9 # &> /dev/null &

# python main_CS_Shapley_parallel_final.py --num_users 10 --epochs 30 --model cnn --dataset cifar --local_ep 1 --seed 4 --selection_ratio 0.8 --overlook_ratio 0.8

python main_CS_Shapley_parallel_final.py --num_users 10 --epochs 30 --model cnn --dataset cifar --local_ep 1 --seed 4 --selection_ratio 0.7 --overlook_ratio 0.8 # &> /dev/null &

# python main_CS_Shapley_parallel_final.py --num_users 10 --epochs 30 --model cnn --dataset cifar --local_ep 1 --seed 4 --selection_ratio 0.7 --overlook_ratio 0.7

python main_GTG_Shapley.py --num_users 10 --epochs 30 --model cnn --dataset cifar --local_ep 1 --seed 4
