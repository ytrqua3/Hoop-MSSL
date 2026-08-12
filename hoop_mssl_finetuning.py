from hoop_mssl_model import PlayerLevelFineTuningModel
from hoop_mssl_dataset import CustomDataset
import torch
import os
import argparse
from torch.utils.data import DataLoader
import json
from fine_tuning_engine import train

def main():
    print("🚀start fine tuning")

    #fetch arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--valid_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)

    args = parser.parse_args()

    models = [("height", torch.nn.MSELoss(), 1, 9), ("position", torch.nn.CrossEntropyLoss(), 6, 10), ("all star", torch.nn.BCEwithLogitsLoss(), 2, 11)]

    for name, loss_fn, out_dim, target_col in models:
        #load data
        train_dir = os.environ.get('SM_CHANNEL_TRAIN')
        test_dir = os.environ.get('SM_CHANNEL_VALIDATION')

        train_dataset = CustomDataset(train_dir, target_col)
        test_dataset = CustomDataset(test_dir, target_col)
        print("✅Successfully created a Dataset")

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True
        )

        model = PlayerLevelFineTuningModel(out_dim)
        state_dict = torch.load('hoop_mssl_model.pth')
        model.load_state_dict(state_dict, strict=False)

        results = train(model, train_dataloader, test_dataloader, loss_fn, torch.optim.Adam(model.parameters(), lr=0.0005), 5, torch.device("cuda"))

        print(f"✅Successfully finished {name} fine tuning training")

        model_dir = os.environ['SM_MODEL_DIR']
        metrics_dir = os.environ['SM_OUTPUT_DATA_DIR']

        print("saving model and metrics into s3")
        output_model_path = os.path.join(model_dir, f'{name}_model.pth')
        torch.save(model.state_dict(), output_model_path)

        output_metrics_path = os.path.join(metrics_dir, f'{name}_metrics.json')
        with open(output_metrics_path, 'w') as f:
            json.dump(results, f)

        print("✅sucessfully ended {name} fine tuning")