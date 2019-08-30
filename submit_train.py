import submitit
from train_mixup import main


executor = submitit.AutoExecutor(folder="log_train_cifar_mixup")  # submission interface (logs are dumped in the folder)
executor.update_parameters(gpus_per_node=4, timeout_min=4320, partition="dev")#"uninterrupted")  # timeout in min
job = executor.submit(main)  # will compute add(5, 7)
print(job.job_id)  # ID of your job


