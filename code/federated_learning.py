import os, argparse, json, copy, time
from tqdm import tqdm
import torch, torchvision
from torchvision import datasets, transforms
import numpy as np

import data_utils as data 
import models 
import experiment_manager as xpm
from fl_devices import Client, Server

if "vca" in os.popen('hostname').read().rstrip(): # Runs on Cluster
  CODE_PATH = "/opt/code/"
  CHECKPOINT_PATH = "/opt/checkpoints/"
  RESULTS_PATH = "/opt/small_files/"
  DATA_PATH = "/opt/in_ram_data/"
else:
  CODE_PATH = ""
  RESULTS_PATH = "/home/sattler/Workspace/PyTorch/multilayer/results/"
  DATA_PATH = "/home/sattler/Data/PyTorch/"
  CHECKPOINT_PATH = "/home/sattler/Workspace/PyTorch/multilayer/checkpoints/"

np.set_printoptions(precision=4, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("--schedule", default="main", type=str)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=None, type=int)
parser.add_argument("--reverse_order", default=False, type=bool)
parser.add_argument("--hp", default=None, type=str)
args = parser.parse_args()



def run_experiment(xp, xp_count, n_experiments):

    print(xp)
    hp = xp.hyperparameters

    model_fn, optimizer_fn = models.get_model(hp["net"])
    client_data, server_data = data.get_data(hp["dataset"], n_clients=hp["n_clients"], alpha=hp["dirichlet_alpha"], path=DATA_PATH)

    for i, d in enumerate(client_data):
        d.subset_transform = data.get_x_transform(hp["x_transform"], i)
        d.label_transform = data.get_y_transform(hp["y_transform"], i)

    clients = [Client(model_fn, optimizer_fn, subset, hp["batch_size"], layers=hp["layers"], idnum=i) for i, subset in enumerate(client_data)]
    server = Server(model_fn, server_data, layers=hp["layers"])
    server.load_model(path=CHECKPOINT_PATH, name=hp["pretrained"])

    # print model
    models.print_model(server.model)
    xp.log({"shared_layers" : list(server.W.keys())})

    # Start Distributed Training Process
    print("Start Distributed Training..\n")
    t1 = time.time()
    for c_round in range(1, hp["communication_rounds"]+1):

        participating_clients = server.select_clients(clients, hp["participation_rate"])

        accs = []
        for i, client in enumerate(clients):
            client.synchronize_with_server(server)
            accs += [client.evaluate()["accuracy"]]
        xp.log({"client_accuracies" : accs}, printout=False)
        xp.log({"mean_accuracy" : np.mean(accs)})    

        for client in tqdm(participating_clients):
            train_stats = client.compute_weight_update(hp["local_epochs"])  

        for i, client in enumerate(clients):
            accs += [client.evaluate()["accuracy"]]
        xp.log({"post_client_accuracies" : accs}, printout=False)
        xp.log({"post_mean_accuracy" : np.mean(accs)}) 
      
        server.aggregate_weight_updates(clients)
        for client in participating_clients:
          xp.log(client.compute_server_angle(server), printout=False)


        # Logging
        if xp.is_log_round(c_round):
            print("Experiment: {} ({}/{})".format(args.schedule, xp_count+1, n_experiments))   

            xp.log({'communication_round' : c_round, 'epochs' : c_round*hp['local_epochs']})

            # Evaluate  
            #xp.log({"client_train_{}".format(key) : value for key, value in train_stats.items()})
            #for i, client in enumerate(clients):
            #    xp.log({"client_{}_val_{}".format(i, key) : value for key, value in client.evaluate().items()})

            # Save results to Disk
            try:
                xp.save_to_disc(path=RESULTS_PATH, name=hp['log_path'])
            except:
                print("Saving results Failed!")

            # Timing
            e = int((time.time()-t1)/c_round*(hp['communication_rounds']-c_round))
            print("Remaining Time (approx.):", '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60), 
                    "[{:.2f}%]\n".format(c_round/hp['communication_rounds']*100))


    xp.log(server.compute_pairwise_angles_layerwise(clients), printout=False)
    xp.save_to_disc(path=RESULTS_PATH, name=hp['log_path'])

    # Save model to disk
    server.save_model(path=CHECKPOINT_PATH, name=hp["save_model"])

    # Delete objects to free up GPU memory
    del server; clients.clear()
    torch.cuda.empty_cache()


def run():

  if args.hp:
    experiments_raw = json.loads(args.hp)
  else:
    with open(CODE_PATH+'federated_learning.json') as data_file:    
      experiments_raw = json.load(data_file)[args.schedule]

  hp_dicts = [hp for x in experiments_raw for hp in xpm.get_all_hp_combinations(x)][args.start:args.end]
  if args.reverse_order:
    hp_dicts = hp_dicts[::-1]
  experiments = [xpm.Experiment(hyperparameters=hp) for hp in hp_dicts]

  print("Running {} Experiments..\n".format(len(experiments)))
  for xp_count, experiment in enumerate(experiments):
    run_experiment(experiment, xp_count, len(experiments))


if __name__ == "__main__":
    run()
    