import os

# Define the values of eps
#Try different values (10−1, 10−2 and 10−3) of ϵ for the FGSM attack using the script run xp.py. Observe the outputs in the experiments/cnn folderboth visually and in terms of accuracy.
## To do 14
eps_values = [0.1, 0.01, 0.001]

# Loop through each value and execute the script
for eps in eps_values:
    print(f"Running evaluate.py with epsilon={eps}")
    command = f"python evaluate.py --path base --model resnet18 --attack fgsm --epsilon {eps}"
    os.system(command)
