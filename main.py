import torch
import argparse
import os
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from model import CNNMNIST, FCNNMNIST, weight_init

def train_an_epoch(e, model, optimizer, loss_func, trainset_loader, device):
    model.train()
    loss_list, accu_list = [], []

    for i, (x, target) in enumerate(trainset_loader):
        x, target = x.to(device), target.to(device)
        x.requires_grad_(True)
        # target.requires_grad_(True)

        optimizer.zero_grad()
        out = model(x)
        loss = loss_func(out, target)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        accuracy = (torch.argmax(out, 1) == target).float().mean()
        accu_list.append(accuracy.item())

        print('%s - [Train Epoch %d|Batch %d] Loss = %.6f, Accuracy = %.4f' % (model.name, e, i, loss.item(), accuracy.item()))

    return loss_list, accu_list

def eval_an_epoch(e, model, loss_func, testset_loader, device):
    model.eval()
    
    with torch.no_grad():
        loss_list, accu_list = [], []

        for i, (x, target) in enumerate(testset_loader):
            x, target = x.to(device), target.to(device)

            out = model(x)
            loss = loss_func(out, target)

            loss_list.append(loss.item())
            accuracy = (torch.argmax(out, 1) == target).float().mean()
            accu_list.append(accuracy.item())

            print('%s - [Eval Epoch %d|Batch %d] Loss = %.6f, Accuracy = %.4f' % (model.name, e, i, loss.item(), accuracy.item()))

        return loss_list, accu_list

def train(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create the network
    full_model_path = config.model_path + config.model_name
    cnn, fcnn = CNNMNIST().to(device), FCNNMNIST().to(device)
    if os.path.exists(full_model_path):
        state_dict = torch.load(full_model_path)
        cnn.load_state_dict(state_dict['cnn'])
        fcnn.load_state_dict(state_dict['fcnn'])
        print('Load the pretrained model from %s successfully!' % full_model_path)
    else:
        if not os.path.exists(config.model_path):
            os.mkdir(config.model_path)
        weight_init(cnn)
        weight_init(fcnn)
        print('First time training!')

    # Optimizer
    cnn_optim = torch.optim.Adam(cnn.parameters(), lr = config.init_lr, betas = [0.5, 0.999])
    fcnn_optim = torch.optim.Adam(fcnn.parameters(), lr = config.init_lr, betas = [0.5, 0.999])

    # Loss function
    loss_func = torch.nn.CrossEntropyLoss()

    # Dataset
    tsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = datasets.MNIST(root = config.dataset_root_path, train = True, transform = tsfm, download = True)
    trainset_loader = torch.utils.data.DataLoader(trainset, batch_size = config.batch_size, shuffle = True)

    testset = datasets.MNIST(root = config.dataset_root_path, train = False, transform = tsfm)
    testset_loader = torch.utils.data.DataLoader(testset, batch_size = config.batch_size, shuffle = False)

    # Summary
    summary = SummaryWriter(config.summary_path)
    total_train_iters, total_eval_iters = 0, 0

    # Start training
    for e in range(1, config.epoch + 1):
        # Vanilla CNN
        cnn_train_losses, cnn_train_accus = train_an_epoch(e, cnn, cnn_optim, loss_func, trainset_loader, device)
        cnn_eval_losses, cnn_eval_accus = eval_an_epoch(e, cnn, loss_func, testset_loader, device)

        # Fourier CNN
        fcnn_train_losses, fcnn_train_accus = train_an_epoch(e, fcnn, fcnn_optim, loss_func, trainset_loader, device)
        fcnn_eval_losses, fcnn_eval_accus = eval_an_epoch(e, fcnn, loss_func, testset_loader, device)

        # Summary
        for idx in range(len(cnn_train_losses)):
            summary.add_scalar('CNN_Train/Loss', cnn_train_losses[idx], total_train_iters)
            summary.add_scalar('CNN_Train/Accuracy', cnn_train_accus[idx], total_train_iters)
            summary.add_scalar('FCNN_Train/Loss', fcnn_train_losses[idx], total_train_iters)
            summary.add_scalar('FCNN_Train/Accuracy', fcnn_train_accus[idx], total_train_iters)
            total_train_iters += 1

        for idx in range(len(cnn_eval_losses)):
            summary.add_scalar('CNN_Eval/Loss', cnn_eval_losses[idx], total_eval_iters)
            summary.add_scalar('CNN_Eval/Accuracy', cnn_eval_accus[idx], total_eval_iters)
            summary.add_scalar('FCNN_Eval/Loss', fcnn_eval_losses[idx], total_eval_iters)
            summary.add_scalar('FCNN_Eval/Accuracy', fcnn_eval_accus[idx], total_eval_iters)
            total_eval_iters += 1

        # Save the model
        state_dict = {
            'cnn' : cnn.state_dict(),
            'fcnn' : fcnn.state_dict()
        }
        torch.save(state_dict, full_model_path)

    summary.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_root_path', type = str, default = './MNIST/')
    parser.add_argument('--model_path', type = str, default = './model/')
    parser.add_argument('--model_name', type = str, default = 'model.pkl')
    parser.add_argument('--summary_path', type = str, default = './log/')
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--epoch', type = int, default = 10)
    parser.add_argument('--init_lr', type = float, default = 1e-3)

    config = parser.parse_args()
    train(config)