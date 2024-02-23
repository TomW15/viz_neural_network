from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import typing as t

try:
    from dataset import INPUT_SIZE, NUM_CLASSES
except ModuleNotFoundError:
    from .dataset import INPUT_SIZE, NUM_CLASSES

LATEST: str = "latest"
MODEL_PATH = Path(__file__).parent / "models"
MODEL_NAME = "mnist-model-{time}-state.dict"


class MnistModel(nn.Module):

    
    
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features=INPUT_SIZE, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self.fc_output = nn.Linear(in_features=32, out_features=NUM_CLASSES)
    
    def forward(self, x):
        
        # Reshape from image to vector
        x = x.reshape(-1, INPUT_SIZE)

        # Apply first layer
        x = self.fc1(x)
        x = F.relu(x)

        # Apply second layer
        x = self.fc2(x)
        x = F.relu(x)

        # Apply Output layer
        x = self.fc_output(x)
        output = F.softmax(x, dim=1)

        return output
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images) ## Generate predictions
        loss = F.cross_entropy(out, labels) ## Calculate the loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss, 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc' : epoch_acc.item()}
    
    def epoch_end(self, epoch,result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
        return


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    _, preds = torch.max(outputs, dim = 1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def evaluate(model: MnistModel, val_loader: DataLoader) -> t.Dict:
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs: int, lr: float, model: MnistModel, train_loader: DataLoader, val_loader: DataLoader, opt_func: torch.optim = torch.optim.SGD) -> t.List:
    
    history = []
    optimizer = opt_func(model.parameters(), lr)
    
    for epoch in range(epochs):
        
        ## Training Phas
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()


            optimizer.step()
            optimizer.zero_grad()
        
        ## Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    
    return history


def save(model: MnistModel) -> None:
    name = MODEL_PATH / MODEL_NAME.format(time=int(time.time()))
    torch.save(model.state_dict(), name)
    torch.save(model.state_dict(), MODEL_PATH / MODEL_NAME.format(time=LATEST))
    return


def load(name: str = None) -> MnistModel:

    if name is None:
        name = MODEL_NAME.format(time=LATEST)
    name = MODEL_PATH / name

    loaded_network = MnistModel()
    loaded_network.load_state_dict(torch.load(name))

    return loaded_network

