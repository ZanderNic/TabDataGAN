
from typing import Any, Callable, List, Optional, Tuple
import copy

# third party
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split

# Projects imports
from ..Data.dataset import CTGan_data_set
from ..Model.CTGan_utils import get_nolin_act, get_loss_function


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # If device is not set try to use cuda else cpu


class EarlyStopping:
    def __init__(self, patience=5, track_model=True, mode='min'):
        self.patience = patience if patience > 0 else np.inf
        self.counter = 0
        if mode == 'min':
            self.best_value = np.inf
            self.improvement = lambda new, best: new < best 
        elif mode == 'max':
            self.best_value = -np.inf
            self.improvement = lambda new, best: new > best  
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'min' for minimization or 'max' for maximization.")

        self.stop = False
        self.best_model = None
        self.track_model = track_model

    def __call__(self, value, model):
        if self.improvement(value, self.best_value):
            self.best_value = value
            self.counter = 0
            if self.track_model:
                self.best_model = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1

        self.stop = self.counter >= self.patience


class Classifier(nn.Module):
    def __init__(
        self, 
        classifier_n_units_in: int,
        classifier_n_units_out_per_category: list,  # This is used to determine if it is a categorical column or a normal colum
        classifier_n_layers_hidden: int = 3, 
        classifier_n_units_hidden: int = 4, 
        classifier_nonlin: str = "leaky_relu",
        classifier_batch_norm: bool = False, 
        classifier_dropout: float = 0.001,
        loss: str = "cross_entropy",
        device: Any = DEVICE, 
    ):
        super().__init__()

        self.device = device
        self.loss = get_loss_function(loss)  
        self.num_categories = len(classifier_n_units_out_per_category)
        self.classifier_n_units_out_per_category = classifier_n_units_out_per_category
        self.net = nn.Sequential()

        self.acc_output = np.cumsum([0] + classifier_n_units_out_per_category).tolist()

        classifier_nonlin = get_nolin_act(classifier_nonlin)

        self.net.append(nn.Linear(classifier_n_units_in, classifier_n_units_hidden))
        if classifier_batch_norm:
            self.net.append(nn.BatchNorm1d(classifier_n_units_hidden))
        self.net.append(classifier_nonlin)
        self.net.append(nn.Dropout(classifier_dropout))

        for _ in range(classifier_n_layers_hidden - 1):
            self.net.append(nn.Linear(classifier_n_units_hidden, classifier_n_units_hidden))
            if classifier_batch_norm:
                self.net.append(nn.BatchNorm1d(classifier_n_units_hidden))
            self.net.append(classifier_nonlin)
            self.net.append(nn.Dropout(classifier_dropout))

        self.output_layers = nn.ModuleList(
            [nn.Linear(classifier_n_units_hidden, out_dim) for out_dim in classifier_n_units_out_per_category]
        )

    def score(self, data_loader):
        self.net.eval()
        loss_func = self.loss

        data_loss = 0
        correct_predictions = [0] * self.num_categories
        total_samples = 0

        with torch.no_grad(): 
            for data, conditions in data_loader:
                data = data.to(self.device)
                conditions = conditions.to(self.device)

                outputs = self.forward(data) 
                loss_value = sum([loss_func(output, torch.argmax(conditions[:, self.acc_output[i]: self.acc_output[i+1]], dim=1)) for i, output in enumerate(outputs)])
        
                data_loss += loss_value.item()
                for i, output in enumerate(outputs):
                    _, predicted = torch.max(output, 1)
                    correct_predictions[i] += (predicted == torch.argmax(conditions[:, self.acc_output[i]: self.acc_output[i+1]], dim=1)).sum().item()

                total_samples += conditions.size(0)
        avg_data_loss = data_loss / len(data_loader)
        data_accuracy = [correct / total_samples * 100 for correct in correct_predictions]

        return avg_data_loss, data_accuracy

    def fit(
        self,
        train_loader,
        test_loader,
        lr,
        opt_betas,
        patience,
        epochs
    ):
        model = self.net.to(self.device)
        model.train()

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, betas=opt_betas)

        early_stopping = EarlyStopping(patience, track_model=True, mode="min")

        for epoch in range(1, epochs + 1):
            model.train() 
            for data, conditions in train_loader:
                data = data.to(self.device)
                conditions = conditions.to(self.device)
                  
                outputs = self.forward(data)

                loss_value = 0
                for i, output in enumerate(outputs):
                    if self.classifier_n_units_out_per_category[i] > 1:
                        loss_value += nn.CrossEntropyLoss()(output, torch.argmax(conditions[:, self.acc_output[i]:self.acc_output[i+1]], dim=1))
                    else:
                        loss_value += nn.MSELoss()(output, conditions[:, self.acc_output[i]:self.acc_output[i+1]])

                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

            loss, acc_test = self.score(test_loader)
            early_stopping(loss, model)  
            if early_stopping.stop:
                print('Classifier Terminated by early stopping')
                break

        if early_stopping.track_model:
            model.load_state_dict(early_stopping.best_model)

        loss_test, acc_test = self.score(test_loader)
        print(f"Training of the Classifier finished with accuracy: {acc_test} and loss: {loss_test}")

        return loss_test, acc_test


    def forward(self, x):
        net = self.net.to(self.device)
        self.output_layers.to(self.device)
        x = x.to(self.device)
        x = net(x)
        outputs = [layer(x) for layer in self.output_layers]  
        return outputs


    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            outputs = self.forward(x)
            predictions = []

            for i, output in enumerate(outputs):
                if self.classifier_n_units_out_per_category[i] > 1: 
                    softmax_output = torch.softmax(output, dim=1)
                    one_hot_pred = torch.zeros_like(softmax_output)
                    one_hot_pred.scatter_(1, torch.argmax(softmax_output, dim=1, keepdim=True), 1)
                    predictions.append(one_hot_pred)
                else:  
                    predictions.append(output)
                    
            return torch.cat(predictions, dim=1)

