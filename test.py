import csv
import os
import numpy as np
import pandas as pd
import re
import torch
import yaml

from argparse import ArgumentParser
from pathlib import Path
from src.center import Center
from src.data import load_data, load_labels
from src.model import load_model
from src.trainer import Trainer

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='test_config.yaml', help='Path to config file')
    parser.add_argument('--batch_config', type=str, default=None, help='Path to batch config file')
    parser.add_argument('--epoch', type=int, default=-1, help='Epoch to evaluate')
    parser.add_argument('--results_record', type=str, default='results_record.csv', help='Path to csv keeping track of all experiments')
    return parser.parse_args()

def main(config, epoch):
       
    torch.manual_seed(config["seed"])

    device = f"cuda:{config['device']}" if torch.cuda.is_available() else "cpu"

    model_dir = Path(config["model_dir"])

    if epoch == -1:
        model_paths = [path for path in model_dir.glob('model-center*.pt') if not 'epoch' in path.stem]
        model_paths = sorted(model_paths, key=lambda f: int(re.search(r'center(\d+)', f.name).group(1)))
        if len(model_paths) == 0: # if no model then look for cluster model
            model_paths = [path for path in model_dir.glob('model-cluster*.pt') if not 'epoch' in path.stem]
            model_paths = sorted(model_paths, key=lambda f: int(re.search(r'cluster(\d+)', f.name).group(1))) # sort clusters
        
            with open(os.path.join(model_dir, "memberships.csv"), 'r') as f:
                csvreader = csv.reader(f)
                for row in csvreader:
                    cluster_ids = list(map(int, row[1:]))
            
            model_paths = [model_paths[cluster_id] for cluster_id in cluster_ids]

    else:
        model_paths = [path for path in model_dir.glob(f'model-center*-epoch{epoch}.pt')]
        model_paths = sorted(model_paths, key=lambda f: int(re.search(r'center(\d+)', f.name).group(1)))
        if len(model_paths) == 0: # if no model then look for cluster model
            model_paths = [path for path in model_dir.glob(f'model-cluster*-epoch{epoch}.pt')]
            model_paths = sorted(model_paths, key=lambda f: int(re.search(r'cluster(\d+)', f.name).group(1)))
        
            # for gt we only output memberships in the last epoch, so first check if this is gt
            with open(os.path.join(model_dir, "memberships.csv"), 'r') as f:
                csvreader = csv.reader(f)
                for row in csvreader:
                    is_gt = row[0] == "gt" or row[0] == "gtpp"
                    cluster_ids = list(map(int, row[1:]))

            if not is_gt:
                # for all other cases
                with open(os.path.join(model_dir, f"memberships_{epoch-1}.csv"), 'r') as f:
                    csvreader = csv.reader(f)
                    for row in csvreader:
                        cluster_ids = list(map(int, row[1:]))
            
            model_paths = [model_paths[cluster_id] for cluster_id in cluster_ids]


    assert len(model_paths) == config["num_centers"]

    memberships = []
    with open(os.path.join(model_dir, "memberships.csv"), 'r') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            memberships.append(row)
    print("Evaluating method with cluster memberships", memberships)

    losses = []
    accs = []
    for c_id, model_path in enumerate(model_paths):
        print("loading model from", Path(model_path).name)
        model = load_model(model_type=config['model_type'], label_len=len(load_labels(config["dataset"])), model_path=model_path)

        center = Center(id=c_id, model=model, optimizer=None, lr=0., device=device, cluster_id=None)

        trainer = Trainer(batch_size=config['batch_size'], 
                            num_workers=config['num_workers'])

        val = load_data(data=config["dataset"], split="test", center=c_id)

        loss, acc = trainer.test(center, val, verbose=True)
        losses.append(loss)
        accs.append(acc)

    return memberships, losses, accs

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    results_record = pd.read_csv(args.results_record, index_col=0)

    if args.batch_config is None:
        memberships, losses, accs = main(config, args.epoch)
        
        print("Average across all", np.mean(accs))
        acc_dict = {}
        for c_id, acc in enumerate(accs):
            acc_dict[f"c_{c_id}"] = round(acc, 6)
        acc_dict["avg"] = np.mean(accs)
        acc_dict["method"] = memberships[0][0]
        acc_dict["epoch"] = args.epoch

        new_row = {}
        new_row.update(config)
        new_row.update(acc_dict)

        results_record = pd.concat([results_record, pd.DataFrame([new_row])], ignore_index=True)
        results_record.to_csv(args.results_record, index=True)

    else:
        batch_config = pd.read_csv(args.batch_config)

        # check that batch config has all of the columns of config
        if set(config.keys()) != set(batch_config.columns.values):
            print("mismatch between config and batch config")
            exit()
        
        for c in batch_config.to_dict(orient='records'):
            memberships, losses, accs = main(c, c["epoch"])
            acc_dict = {}
            for c_id, acc in enumerate(accs):
                acc_dict[f"c_{c_id}"] = round(acc, 6)
            acc_dict["avg"] = np.mean(accs)
            acc_dict["method"] = memberships[0][0]
            acc_dict["epoch"] = c["epoch"]

            new_row = {}
            new_row.update(c)
            new_row.update(acc_dict)

            results_record = pd.concat([results_record, pd.DataFrame([new_row])], ignore_index=True)
            results_record.to_csv(args.results_record, index=True)

