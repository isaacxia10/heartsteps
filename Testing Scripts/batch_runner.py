import os
import sys


assert len(sys.argv) == 5

sim_count = int(sys.argv[1])
num_jobs = int(sys.argv[2])
job_num = int(sys.argv[3])
script_name = sys.argv[4]


for i in range(sim_count):
    if i%num_jobs == job_num:
        print(i)

os.system("python " + script_name +" " + str(i))