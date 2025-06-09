import copy
import pandas as pd
import torch
import yaml

from argparse import ArgumentParser
from src.center import Center
from src.data import load_data, load_labels
from src.model import load_model
from src.trainer import Trainer
from src.utils import get_summarywriter

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--batch_config', type=str, default=None, help='Path to batch config file')
    parser.add_argument('--run_record', type=str, default='run_record.csv', help='Path to csv keeping track of all experiments')
    return parser.parse_args()

def main(config):

    torch.manual_seed(config["seed"])

    device = f"cuda:{config['device']}" if torch.cuda.is_available() else "cpu"
    model = load_model(model_type=config['model_type'], label_len=len(load_labels(config["dataset"])))
    writer = get_summarywriter(out_dir=config['out_dir'], num_centers=config["num_centers"], label_len=len(load_labels(config["dataset"])), method="cfl")

    centers = [Center(id, copy.deepcopy(model), optimizer=config['optimizer'], lr=config['lr'], device=device) for id in range(config["num_centers"]) ]
    server = Center(-1, copy.deepcopy(model), optimizer=config['optimizer'], lr=config['lr'], device=device)

    trainer = Trainer(batch_size=config['batch_size'], 
                        num_workers=config['num_workers'], 
                        epochs=config['epochs'], 
                        local_epochs=config['local_epochs'], 
                        save_every=config['save_every'],
                        writer=writer)
    
    center_trains = [load_data(config["dataset"], "train", c_id) for c_id in range(config["num_centers"])]
    center_vals =  [load_data(config["dataset"], "val", c_id) for c_id in range(config["num_centers"])]

    trainer.fit_cfl(server, centers, center_trains, center_vals)

    return writer.log_dir

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    

    if args.batch_config is None:
        print(config)
        log_dir = main(config)
        config['log_dir'] = log_dir
        run_record = pd.read_csv(args.run_record, index_col=0)
        run_record = pd.concat([run_record, pd.DataFrame([config])], ignore_index=True)
        run_record.to_csv(args.run_record)
    else:
        batch_config = pd.read_csv(args.batch_config)

        # check that batch config has all of the columns of config
        if set(config.keys()) != set(batch_config.columns.values):
            print("mismatch between config and batch config")
            exit(0)
        
        for i, c in enumerate(batch_config.to_dict(orient='records')):
            print("Experiment", 1+i, "of", len(batch_config))
            print(c)
            log_dir = main(c)
            c['log_dir'] = log_dir
            run_record = pd.read_csv(args.run_record, index_col=0)
            run_record = pd.concat([run_record, pd.DataFrame([c])], ignore_index=True)
            run_record.to_csv(args.run_record)
