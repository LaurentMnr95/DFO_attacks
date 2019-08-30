import submitit
from random_noise import main

# optimizers = ["RandomSearch", "BPRotationInvariantDE", "NaiveTBPSA", "DE", "CMA", "TwoPointsDE", "OnePointDE", "PSO", "AlmostRotationInvariantDE", "RotationInvariantDE"]
# optimizers += ["MultiCMA", "MultiScaleCMA","DiagonalCMA", "OnePlusOne", "CauchyOnePlusOne"]
# optimizers= ["CauchyOnePlusOne"]
save_file = "different_nets.txt"
CLASSIFIERS = ["inception_v3", "resnet50", "vgg16_bn"]
epsilons = [0.01, 0.03, 0.05, 0.1]
prior_sizes = [1, 5, 10, 15, 20, 25, 35, 50, 75, 100, 120, 150, 175, 200, 224]
for eps in epsilons:
    for s in prior_sizes:
        for c in CLASSIFIERS:
            # submission interface (logs are dumped in the folder)
            executor = submitit.AutoExecutor(folder='logs/noises')
            executor.update_parameters(gpus_per_node=2, timeout_min=4320, partition="uninterrupted")  # timeout in min
            job = executor.submit(main, classifier=c, epsilon=eps, s=s, mode="noise", save_file=save_file)
            print(job.job_id)  # ID of your job
