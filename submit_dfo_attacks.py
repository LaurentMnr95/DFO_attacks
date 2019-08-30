import submitit
from test_imagnet import main

# ["RandomSearch", "BPRotationInvariantDE", "NaiveTBPSA", "DE", "CMA", "TwoPointsDE", "OnePointDE", "PSO", "AlmostRotationInvariantDE", "RotationInvariantDE"]
optimizers = ["cGA", "PBIL"]
#optimizers += ["MultiCMA", "MultiScaleCMA","DiagonalCMA", "OnePlusOne", "CauchyOnePlusOne","bandits"]
#optimizers= ["CauchyOnePlusOne"]
#optimizers = ["CMA", "DiagonalCMA","bandits"]
epsilons = [0.05, 0.03, 0.01]
prior_sizes = [20, 50]
for optimizer in optimizers:
    for eps in epsilons:
        for s in prior_sizes:
            executor = submitit.AutoExecutor(folder="logs/log_DFO_imagenet/log_"+optimizer+"_"+str(eps) +
                                             '_'+str(s))  # submission interface (logs are dumped in the folder)
            executor.update_parameters(gpus_per_node=1, timeout_min=4320, partition="uninterrupted")  # timeout in min
            job = executor.submit(main, optimizer_DFO=optimizer,
                                  epsilon=eps,
                                  prior_size=s,
                                  max_budget=10000,
                                  outfile="results_dfo_att.txt"
                                  )
            print(job.job_id)  # ID of your job
