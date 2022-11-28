from dotenv import load_dotenv

import os

from comet_ml import Experiment

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch.utils.data as data_utils
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from .hypernetwork import TrainingModes

torch.set_default_dtype(torch.float32)

from tqdm import trange


def get_dataset(size=60000, masked=False, mask_no=200, mask_size=700, shared_mask=False, batch_size=32, test_batch_size=32):
    mods = [transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,)),    #mean and std of MNIST
        transforms.Lambda(lambda x: torch.flatten(x))]
    mods = transforms.Compose(mods)
    
    trainset = datasets.MNIST(root='./data/train', train=True, download=True, transform=mods)
    testset = datasets.MNIST(root='./data/test', train=False, download=True, transform=mods)
    if masked:
        trainset = MaskedDataset(trainset, mask_no, mask_size)
        testset = MaskedDataset(testset, mask_no, mask_size)
        if shared_mask:
            testset.masks = trainset.masks
        
    indices = torch.arange(size)
    trainset = data_utils.Subset(trainset, indices)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                         shuffle=False, num_workers=1)
    return trainloader, testloader



class MaskedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, masks, mask_size):
        inputs = dataset[0][0].shape[0]
        self.mask_size = mask_size
        self.dataset = dataset
        self.template = np.zeros(inputs)
        self.masks = self._create_mask(masks)
        self.masks_indices = np.random.choice(np.arange(masks), len(dataset))
        
    def _create_mask(self, count):
        masks = np.array([np.random.choice((len(self.template)), self.mask_size, False) for _ in range(count)])
        tmp = np.array([self.template.copy() for _ in range(count)])
        for i, mask in enumerate(masks):
            tmp[i, mask] = 1
        mask = torch.from_numpy(tmp).to(torch.float32)
        return mask
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        mask = self.masks[self.masks_indices[idx]]
        return image, label, mask
    

def train_regular(network, optimizer, criterion, loaders, size, epochs, 
                  device='cuda:0', 
                  name="dense-net", 
                  test_every=10, experiment=None):
    
    if experiment is None:
        load_dotenv()
        experiment = Experiment(api_key=os.environ.get("COMET_KEY"), project_name="hypernetwork")
    experiment.add_tag(name)
    experiment.log_parameter("training_size", size)
    experiment.log_parameter("max_epochs", epochs)
    experiment.log_parameter("check_val_every_n_epoch", 5)
    
    trainloader, testloader = loaders
    train_loss = []
    test_loss = []
    test_accs = []
    with trange(epochs) as t:
        for epoch in t:
            total_loss = 0
            running_loss = 0.0
            correct = 0
            total = 0
            network = network.train()
            for i, data in enumerate(trainloader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = network(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                correct += (outputs.argmax(1)==labels).float().sum()
                total += outputs.shape[0]
                running_loss += loss.item()
                train_loss.append(loss.item())
                if i>0 and i % 100 == 0:
                    total_loss += running_loss/100
                    running_loss = 0.0
                    correct = 0
                    total=0

            total_loss = 0
            correct = 0
            denom = 0
            network = network.eval()
            if epoch%test_every==0:
                for i, data in enumerate(testloader):
                    images, labels = data
                    denom += len(labels)
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = network(images)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                    total_loss += criterion(outputs, labels).item()

                test_loss.append(total_loss/i)
                test_accs.append(correct/denom*100)

                t.set_postfix(test_acc=correct/denom*100, loss=total_loss/i)
                experiment.log_metric("test_accuracy", correct/denom*100, step=epoch)    
                experiment.log_metric("test_loss", test_loss[-1], step=epoch)
                
    experiment.end()
    return max(test_accs), test_loss[-1]

def train_slow_step(hypernet, optimizer, criterion, loaders, data_size, epochs, masks_no, 
                    experiment=None,
                    tag="slow-step-hypernet", 
                    device='cuda:0', 
                    project_name="hypernetwork",
                    test_every=5):
    """ Train hypernetwork using slow step method - use the same mask for a whole batch, change it once per iteration."""
    if experiment is None:
        experiment = Experiment(api_key=os.environ.get("COMET_KEY"), project_name=project_name, display_summary_level=0)
        experiment.add_tag(tag)
    experiment.log_parameter("test_nodes", hypernet.test_nodes)
    experiment.log_parameter("mask_size", hypernet.mask_size)
    experiment.log_parameter("node_hidden_size", hypernet.node_hidden_size)
    experiment.log_parameter("lr", optimizer.defaults['lr'])
    experiment.log_parameter("training_size", data_size)
    experiment.log_parameter("masks_no", masks_no)
    experiment.log_parameter("max_epochs", epochs)
    experiment.log_parameter("check_val_every_n_epoch", test_every)

    trainloader, testloader = loaders
    test_loss = []
    test_accs = []
    mask_idx = 0
    with trange(epochs) as t:
        for epoch in t:
            hypernet.train()
            for i, data in enumerate(trainloader):
                try:
                    inputs, labels, _ = data
                except ValueError:
                    inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                masks = hypernet.test_mask[mask_idx].view(1,-1)
                masks = torch.stack(masks).to(device)
                mask_idx = (mask_idx+1) % len(hypernet.test_mask)

                optimizer.zero_grad()
                outputs = hypernet(inputs, masks)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            total_loss = 0
            correct = 0
            denom = 0

            hypernet.eval()
            if epoch%test_every==0:
                for i, data in enumerate(testloader):
                    try:
                        images, labels, _ = data
                    except ValueError:
                        images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)

                    denom += len(labels)

                    outputs = hypernet(images)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                    total_loss += criterion(outputs, labels).item()

                test_loss.append(total_loss/denom)
                test_accs.append(correct/denom*100)

                t.set_postfix(test_acc=correct/denom*100, loss=total_loss/i)
                experiment.log_metric("test_accuracy", correct/len(testloader.dataset)*100, step=epoch)
                experiment.log_metric("test_loss", test_loss[-1], step=epoch)

    experiment.end()    
    return max(test_accs), test_loss[np.argmax(test_accs)]


def test_model(hypernet, testloader, device='cpu', verbose=False):
    criterion = torch.nn.CrossEntropyLoss()
    correct = 0
    loss = 0
    hypernet.eval()
    for i, data in enumerate(testloader):
        try:
            images, labels, _ = data
        except ValueError:
            images, labels = data
            
        images = images.to(device)
        labels = labels.to(device)
        outputs = hypernet(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        loss += criterion(outputs, labels).item()

    if verbose:
        print(f"Test acc: {correct/len(testloader.dataset)*100:.2f}")
        print(f"Test loss: {loss/(i+1):.2g}")
    return correct/len(testloader.dataset)*100


def batch_predict(network, loader, device='cpu'):
    res = []
    for i, data in enumerate(loader):
        try:
            images, labels, _ = data
        except ValueError:
            images, labels = data
            
        images = images.to(device)
        labels = labels.to(device)
        outputs = network(images)
        res.append(outputs.cpu().detach().numpy())
    return np.concatenate(res)


def train_model(hypernet, 
                optimizer, 
                criterion, 
                trainloader, 
                epochs,
                device='cpu',
                verbose=True):
    mask_idx = 0
    masks = None
    with trange(epochs, disable=not verbose) as t:
        for _ in t:
            total_loss = 0
            hypernet.train()
            for num, data in enumerate(trainloader):
                try:
                    inputs, labels, _ = data
                except ValueError:
                    inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                if hypernet.mode == TrainingModes.SLOW_STEP:
                    masks = hypernet.test_mask[mask_idx].repeat(len(inputs), 1)
                    mask_idx = (mask_idx+1) % len(hypernet.test_mask)

                    optimizer.zero_grad()

                    outputs = hypernet(inputs, masks)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    total_loss += loss.item()
                    optimizer.step()
                elif hypernet.mode == TrainingModes.CARTHESIAN:
                    mask_order = np.arange(len(hypernet.test_mask))
                    np.random.shuffle(mask_order)
                    for mask_idx in mask_order:
                        masks = hypernet.test_mask[mask_idx].repeat(len(inputs), 1)

                        optimizer.zero_grad()

                        outputs = hypernet(inputs, masks)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        total_loss += loss.item()
                        
                        optimizer.step()
                    total_loss /= len(mask_order)
            total_loss /= (num+1)
    return total_loss

def basic_train_loop(network, optimizer, criterion, trainloader, epochs, device="cpu"):
    for _ in range(epochs):
        network.train()
        total_loss = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss/(i+1)

class GenericDataset(torch.utils.data.Dataset):
    def __init__(self, data, samples_no: int=None):
        samples = samples_no or len(data[0])
        self.indices = np.arange(samples)
        self.index = 0
        self.max_samples = samples
        self.data_x = data[0].to(torch.float32)
        self.data_y = data[1]

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]
    
    def __len__(self):
        return self.max_samples
    
def get_dataloader(X, y, size=None, batch_size=32, shuffle=True):
    train_dataset = GenericDataset((X, y))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=shuffle)
    
    return trainloader

def train_carthesian(hypernet, optimizer, criterion, loaders, data_size, epochs, masks_no, tag="carthesian-hypernet", device='cuda:0', test_every=5):
    experiment = Experiment(api_key=os.environ.get("COMET_KEY"), project_name="hypernetwork", display_summary_level=0)
    experiment.add_tag(tag)
    experiment.log_parameter("test_nodes", hypernet.test_nodes)
    experiment.log_parameter("mask_size", hypernet.mask_size)
    experiment.log_parameter("training_size", data_size)
    experiment.log_parameter("masks_no", masks_no)
    experiment.log_parameter("max_epochs", epochs)
    experiment.log_parameter("check_val_every_n_epoch", test_every)

    trainloader, testloader = loaders
    
    test_accs = []
    test_loss = []
    with trange(epochs) as t:
        for epoch in t:
            hypernet.train()
            for i, data in enumerate(trainloader):
                try:
                    inputs, labels, _ = data
                except ValueError:
                    inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                mask_order = np.arange(len(hypernet.test_mask))
                np.random.shuffle(mask_order)
                for mask_idx in mask_order:
                    masks = hypernet.test_mask[mask_idx].view(1,-1)

                    optimizer.zero_grad()

                    outputs = hypernet(inputs, masks)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            denom = 0
            correct = 0
            total_loss = 0
            hypernet.eval()
            y_true = []
            y_pred = []
            if epoch%test_every==0 or epoch==epochs-1:
                for i, data in enumerate(testloader):
                    try:
                        images, labels, _ = data
                    except ValueError:
                        images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)

                    denom += len(labels)

                    outputs = hypernet(images)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
                    y_true.extend(labels.cpu().numpy().tolist())
                    y_pred.extend(predicted.cpu().detach().numpy().tolist())
                    total_loss += criterion(outputs, labels).item()

                test_loss.append(total_loss/i)
                test_accs.append(correct/denom*100)

                t.set_postfix(test_acc=correct/denom*100, loss=total_loss/i)
                experiment.log_metric("test_accuracy", correct/len(testloader.dataset)*100, step=epoch)
                experiment.log_metric("test_f1", f1_score(y_true, y_pred))
                experiment.log_metric("test_loss", test_loss[-1], step=epoch)
                    
    experiment.end()    
    return max(test_accs), test_loss[np.argmax(test_accs)]
