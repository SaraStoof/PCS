'''
This script runs speedup tests on our DLA implementation, comparing performance with and 
without Numba to evaluate its impact on execution speed.
'''

import subprocess

base_command = ["python3", "../dla_3d.py"]

num_sims = ['1', '2', '5', '10', '100']
batch_sizes = ['100', '1000']
no_hits_maxs = ['1', '2', '5', '10', '50']

with open("results/result_normal.txt", "w") as f:
    f.writelines("num_sims, batch_size, no_hits_max, mold_cov_3d, mold_cov_surface\n")

    for n in num_sims:
        for b in batch_sizes:
            # no_hits_max is how long the particle loops without touching the cluster before breaking the loop
            for h in no_hits_maxs:
                result = subprocess.run(
                    [base_command[0], base_command[1], n, b, h], capture_output=True, text=True)
                f.writelines(result.stdout)
