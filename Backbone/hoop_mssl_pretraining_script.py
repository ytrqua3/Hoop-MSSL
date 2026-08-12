from pathlib import Path
import torch
import numpy as np
import os
import torch.nn as nn
from pytorch_metric_learning import losses
import argparse
from torch.utils.data import DataLoader
import json
from hoop_mssl_model import HoopMssl_V1
from hoop_mssl_dataset import CustomDataset

def get_offense_team(possession):
  """
  return the team_id of the offensive team of the possession
  """
  team_counts = {}
  for moment in possession:
      ball_pos = moment[moment[:, 0] == -1.0][:, 2:4]
      diff = (moment[moment[:, 0] != -1][:, 2:4] - ball_pos)**2
      ball_handler = np.argmin(np.sqrt(diff[:, 0] + diff[:, 1]))
      ball_handling_team = moment[moment[:, 0] != -1][ball_handler, 0]
      team_counts[ball_handling_team] = team_counts.get(ball_handling_team, 0) + 1

  return int(max(team_counts, key=team_counts.get))

def get_targets(X, y, shuffled_idx1, shuffled_idx2, mask_mask1, mask_mask2):
    B = X.shape[0]
    T = X.shape[1]
    A = X.shape[2]

    # generate targets for the loss function
    shuffled_idx_4d_1 = shuffled_idx1.view(B, T, A, 1).expand(B, T, A, 3).to(dtype=torch.long)
    mr_target1 = torch.gather(X, dim=2, index=shuffled_idx_4d_1)[mask_mask1] # (mask_len, 3)
    shuffled_idx_4d_2 = shuffled_idx2.view(B, T, A, 1).expand(B, T, A, 3).to(dtype=torch.long)
    mr_target2 = torch.gather(X, dim=2, index=shuffled_idx_4d_2)[mask_mask2] #(mask_len, 3)
    mr_target = torch.cat([mr_target1, mr_target2], dim=0)
    
    pri_target = y.unsqueeze(1)

    cl_target = torch.cat([torch.arange(0, X.shape[0]), torch.arange(0, X.shape[0])]) # (64, ) [0, 1, 2, ..., 31, 0, 1, 2, ..., 31]

    return mr_target, pri_target, cl_target

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               current_epoch: int):
    model.train()
    train_loss = 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) #(32, 121, 11, 3), (32, 11)
        X = X.to(dtype=torch.float32)
        mr_loss_fn = nn.MSELoss()
        pri_loss_fn = nn.BCEWithLogitsLoss()
        cl_loss_fn = losses.NTXentLoss(temperature=0.07)
        lambda_pri = 100
        lambda_cl = 100

        # get prediction
        mr_pred, pri_pred, cl_pred, mask_mask1, mask_mask2, shuffled_idx1, shuffled_idx2 = model(X)

        mr_target, pri_target, cl_target = get_targets(shuffled_idx1, shuffled_idx2, mask_mask1, mask_mask2)

        # calculate loss
        loss_mr = mr_loss_fn(mr_pred, mr_target)
        loss_pri = pri_loss_fn(pri_pred, pri_target)
        loss_cl = cl_loss_fn(cl_pred, cl_target)
        loss = loss_mr + lambda_pri * loss_pri + lambda_cl * loss_cl
        train_loss += loss.item()

        # reset gradient
        optimizer.zero_grad()

        # calculate new gradient
        loss.backward()

        # update parameters
        optimizer.step()

    # get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    return train_loss, loss_mr.item(), loss_pri.item(), loss_cl.item()

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              device: torch.device):
    model.eval()
    test_loss, pri_acc = 0, 0

    mr_loss_fn = nn.MSELoss()
    pri_loss_fn = nn.BCEWithLogitsLoss()
    cl_loss_fn = losses.NTXentLoss(temperature=0.07)
    lambda_pri = 100
    lambda_cl = 100

    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            mr_pred, pri_pred, cl_pred, mask_mask1, mask_mask2, shuffled_idx1, shuffled_idx2 = model(X)

            mr_target, pri_target, cl_target = get_targets(shuffled_idx1, shuffled_idx2, mask_mask1, mask_mask2)

            # calculate loss
            loss_mr = mr_loss_fn(mr_pred, mr_target)
            loss_pri = pri_loss_fn(pri_pred, pri_target)
            loss_cl = cl_loss_fn(cl_pred, cl_target)
            loss = loss_mr + lambda_pri * loss_pri + lambda_cl * loss_cl
            test_loss += loss.item()

            # calculate accuracy
            pri_acc_pred = torch.argmax(pri_pred.view(X.shape[0]*10*2, 2), dim=1)
            pri_acc_target = torch.vstack([y, y]).view(X.shape[0]*2*10)
            pri_acc += (pri_acc_pred == pri_acc_target).to(dtype=torch.float32).mean()

    # get average loss per batch
    test_loss = test_loss / len(dataloader)
    pri_acc = pri_acc / len(dataloader)

    return test_loss, pri_acc, loss_mr.item(), loss_pri.item(), loss_cl.item()

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          d: int):
    results = {
       "epochs": [],
        "train_loss": [],
        "train_mr_loss": [],
        "train_pri_loss": [],
        "train_cl_loss": [],
        "test_loss": [],
        "test_mr_loss": [],
        "test_pri_loss": [],
        "test_cl_loss": [],
    }
    model.to(device=device)

    # Training Loop
    for epoch in range(epochs):
        train_loss, loss_mr, loss_pri, loss_cl = train_step(model=model,
                                dataloader=train_dataloader,
                                optimizer=optimizer,
                                device=device,
                                current_epoch=epoch)
        #report progress and test on every d epoch
        if epoch % d == 0:
          test_loss, pri_acc, test_mr_loss, test_pri_loss, test_cl_loss = test_step(model=model,
                                dataloader=test_dataloader,
                                device=device)

          print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train MR loss: {loss_mr:.4f} | "
            f"train PRI loss: {loss_pri:.4f} | "
            f"train CL loss: {loss_cl:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test pri_acc: {pri_acc:.4f} |"
            f"test MR loss: {test_mr_loss:.4f} | "
            f"test PRI loss: {test_pri_loss:.4f} | "
            f"test CL loss: {test_cl_loss:.4f} | "
          )

          # Update results
          results["epochs"].append(epoch)
          results["train_loss"].append(train_loss)
          results["test_loss"].append(test_loss)
          results["train_mr_loss"].append(loss_mr)
          results["train_pri_loss"].append(loss_pri)
          results["train_cl_loss"].append(loss_cl)
          results["test_mr_loss"].append(test_mr_loss)
          results["test_pri_loss"].append(test_pri_loss)
          results["test_cl_loss"].append(test_cl_loss)
    return results

def main():
    print("🚀start script.py")

    #fetch arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--valid_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    
    args = parser.parse_args()

    train_dir = os.environ.get('SM_CHANNEL_TRAIN') #copies the train s3 directory into a local directory
    test_dir   = os.environ.get('SM_CHANNEL_VALIDATION')
    print("train_dir: " + train_dir)
    print("test_dir: " + test_dir)

    print("🚀Creating Dataset ...")
    train_dataset = CustomDataset(dir=Path(train_dir), target_fn=get_offense_team)
    test_dataset = CustomDataset(dir=Path(test_dir), target_fn=get_offense_team)
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

    model1 = HoopMssl_V1().to(device='cuda')

    print("🚀Start training job ...")
    results = train(model1, train_dataloader, test_dataloader, optimizer = torch.optim.AdamW(model1.parameters(), lr=args.learning_rate), epochs=args.epochs, device=torch.device('cuda'), d=1)

    print("✅Successfully finished training")

    model_dir = os.environ['SM_MODEL_DIR']
    metrics_dir = os.environ['SM_OUTPUT_DATA_DIR']

    print("saving model and metrics into s3")
    output_model_path = os.path.join(model_dir, 'hoop_mssl_model.pth')
    torch.save(model1.state_dict(), output_model_path)

    output_metrics_path = os.path.join(metrics_dir, 'metrics.json')
    with open(output_metrics_path, 'w') as f:
       json.dump(results, f)

    print("✅sucessfully ended script.py")

if __name__ == '__main__':
    main()
