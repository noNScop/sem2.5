
import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import OneCycleLR

class LRFinder:
    """
    My implementation of learning rate finder for pytorch, based on https://docs.fast.ai/callback.schedule.html#lrfinder
    Methods:
        LRFinder.find_lr()
        LRFinder.plot()
    """
    def __init__(self, model, dataloader, loss_fn, optimizer):
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.stats = {'lr': [], 'loss': []}

    def find_lr(self, start_lr=1e-7, end_lr=10, num_iters=100):
        """
        Runs the learning rate test in a specified range.

        Args:
            start_lr: minimum learning rate to test
            end_lr: maximum learning rate to test
            num_iters: number of testing iterations
            
        I want my learning rate finder to handle only the case when all parameters have the same learning rate
        and I want my finder to be independent from learning rate set during the initialisation of optimizer, therefore
        I am setting 'lr' parameter of all parameter gropups to 1, as my scheduler is multiplying it later on
        """
        device = next(self.model.parameters()).device
        
        best_loss = float('inf')
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 1

        lr_lambda = lambda x: start_lr * (end_lr / start_lr) ** (x/num_iters)
        scheduler = LambdaLR(self.optimizer, lr_lambda)

        epochs = num_iters // len(self.dataloader)

        for epoch in range(epochs+1):
            # set the total number of iterations in epoch for tqdm progress bar
            if epochs == 0:
                total = num_iters
            elif epoch == epochs:
                total = num_iters % len(self.dataloader)
            else:
                total = len(self.dataloader)
                
            # Separating tqdm from dataloader to avoid conflicts between them
            dl_iterator = iter(self.dataloader)
            for _ in tqdm(range(len(self.dataloader)), total=total, desc="Iterations", leave=False):
                batch, targets = next(dl_iterator)
                batch, targets = batch.to(device), targets.to(device)
                
                logits = self.model(batch)
                loss = self.loss_fn(logits, targets)
                
                best_loss = min(best_loss, loss)
                if loss > 4 * best_loss:
                    print("Loss diverged, stopping early")
                    return
            
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                self.stats['lr'].append(scheduler.get_last_lr()[0])
                self.stats['loss'].append(loss.item())
                
                scheduler.step()
                if scheduler.get_last_lr()[0] > end_lr:
                    return

    def plot(self, skip_start=0, skip_end=5):
        """
        Plots the results of the learning rate test.

        Args:
            skip_start: skip n first data points
            skip_start: skip n last data points
        """
        stats = self.stats
        
        fig = plt.figure()
        ax = fig.add_subplot()
        
        ax.plot(stats['lr'][skip_start:-skip_end], stats['loss'][skip_start:-skip_end])
        
        ax.set_xscale("log")
        ax.set_xlabel("learning rate")
        ax.set_ylabel("loss")


# -------------------------------TRAINING FUNCTIONS


def train_step(model, dataloader, optimizer, scheduler, loss_fn, device, accuracy):
    avg_accuracy = 0
    avg_loss = 0
    model.train()

    # Necessary for dataloader to work with tqdm without errors, tqdm interferes with dataloader workers shutdown process,
    # therefore I separated them
    dl_iterator = iter(dataloader)
    for _ in tqdm(range(len(dataloader)), desc="Training", leave=False):
        batch, target = next(dl_iterator)
        batch, target = batch.to(device), target.to(device)
        
        logits = model(batch)
        loss = loss_fn(logits, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        avg_loss += loss.item()
        avg_accuracy += accuracy(logits, target).item()

    avg_loss /= len(dataloader)
    avg_accuracy /= len(dataloader)
    return avg_loss, avg_accuracy

def valid_step(model, dataloader, loss_fn, device, accuracy):
    avg_accuracy = 0
    avg_loss = 0
    model.eval()

    # Necessary for dataloader to work with tqdm without errors
    dl_iterator = iter(dataloader)
    with torch.inference_mode():
        for _ in tqdm(range(len(dataloader)), desc="Validation", leave=False):
            batch, target = next(dl_iterator)
            batch, target = batch.to(device), target.to(device)
            
            logits = model(batch)
            loss = loss_fn(logits, target)

            avg_loss += loss.item()
            avg_accuracy += accuracy(logits, target).item()

    avg_loss /= len(dataloader)
    avg_accuracy /= len(dataloader)
    return avg_loss, avg_accuracy

def train(model, train_dl, valid_dl, optimizer, loss_fn, epochs, writer=None):
    device = next(iter(model.parameters())).device
    acc_fn = Accuracy(task="multiclass", num_classes=len(train_dl.dataset.classes)).to(device)
    max_lr = optimizer.defaults['lr']
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_dl), epochs=epochs)

    for epoch in tqdm(range(epochs), desc="Epochs"):
        train_loss, train_acc = train_step(
            model,
            train_dl,
            optimizer,
            scheduler,
            loss_fn,
            device,
            acc_fn
        )

        valid_loss, valid_acc = valid_step(
            model,
            valid_dl,
            loss_fn,
            device,
            acc_fn
        )

        if writer is not None:
            writer.add_scalar("train_loss", train_loss, global_step=epoch)
            writer.add_scalar("validation_loss", valid_loss, global_step=epoch)
            
            writer.add_scalar("train_accuracy", train_acc, global_step=epoch)
            writer.add_scalar("validation_accuracy", valid_acc, global_step=epoch)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"valid_loss: {valid_loss:.4f} | "
            f"valid_acc: {valid_acc:.4f}"
        )