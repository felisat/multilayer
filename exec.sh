#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=felix.sattler@hhi.fraunhofer.de
#SBATCH --output=out/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1


hyperparameters=' [{
	"dataset" : ["EMNIST"], 
	"net" : ["ConvNet"],
	
	"n_clients" : [100],
	"dirichlet_alpha" : [100.0, 0.1],
	"x_transform" : ["rotation", null],
	"y_transform" : ["shift", null],

	"communication_rounds" : [1000],
	"local_epochs" : [1],
	"participation_rate" : [0.1],
	"batch_size" : [128],
	"layers" : [".*weight.*", ".*bias.*"],

	"pretrained" : [null],
	"save_model" : [null],
	"log_frequency" : [-100],
	"log_path" : ["layers_compare_big/"],
	"job_id" : [['$SLURM_JOB_ID']]}]'



if [[ "$HOSTNAME" == *"vca"* ]]; then # Cluster
	echo $hyperparameters
	source "/etc/slurm/local_job_dir.sh"

	export SINGULARITY_BINDPATH="$LOCAL_DATA:/data,$LOCAL_JOB_DIR:/mnt/output,./code:/opt/code,./checkpoints:/opt/checkpoints,./results:/opt/small_files,$HOME/in_ram_data:/opt/in_ram_data"
	singularity exec --nv $HOME/base_images/pytorch15.sif python -u /opt/code/federated_learning.py --hp="$hyperparameters"

	mkdir -p results
	cp -r ${LOCAL_JOB_DIR}/. ${SLURM_SUBMIT_DIR}/results	


else # Local
	source activate base
	python -u code/federated_learning.py --hp="$hyperparameters"




fi






