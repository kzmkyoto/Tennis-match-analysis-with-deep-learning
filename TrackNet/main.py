from model import TrackNet
import torch
from dataset import TrackNetDataset
import os
from train import train, validate
from tensorboardX import SummaryWriter
import argparse



# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--path_dataset', type=str, help='path of dataset')
#     parser.add_argument('--batch_size', type=int, default=2, help='batch size')
#     parser.add_argument('--exp_id', type=str, default='default', help='used to make path')
#     parser.add_argument('--num_epochs', type=int, default=500, help='number of training epochs')
#     parser.add_argument('--lr', type=float, default=1.0, help='learning rate')
#     parser.add_argument('--val_intervals', type=int, default=5, help='number of epochs to run validation')
#     parser.add_argument('--steps_per_epoch', type=int, default=200, help='number of steps per epoch')
#     args = parser.parse_args()

#     train_dataset = TrackNetDataset(model='train', path_dataset=args.path_dataset)
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=1,
#         pin_memory=True
#     )

#     val_dataset = TrackNetDataset(model='val', path_dataset=args.path_dataset)
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=1,
#         pin_memory=True
#     )

#     model = TrackNet()
#     device = 'cuda'
#     model = model.to(device)

#     exps_path = './exps/{}'.format(args.exp_id)
#     tb_path = os.path.join(exps_path, 'plots')
#     if not os.path.exists(tb_path):
#         os.makedirs(tb_path)
#     log_writer = SummaryWriter(tb_path)
#     model_last_path = os.path.join(exps_path, 'model_last.pt')
#     model_best_path = os.path.join(exps_path, 'model_best.pt')

#     optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
#     val_best_metric = 0

#     for epoch in range(args.num_epochs):
#         train_loss = train(model, train_loader, optimizer, device, epoch, args.steps_per_epoch)
#         print(f"train_loss = {train_loss}")
#         log_writer.add_scalar('Train/training_loss', train_loss, epoch)
#         log_writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)

#         if (epoch > 0) and (epoch & args.val_intervals == 0):
#             val_loss, precision, recall, f1 = validate(model, val_loader, device, epoch)
#             print(f"val_loss = {val_loss}")
#             log_writer.add_scalar('Val/loss', val_loss, epoch)
#             log_writer.add_scalar('Val/precision', precision, epoch)
#             log_writer.add_scalar('Val/recall', recall, epoch)
#             log_writer.add_scalar('Val/f1', f1, epoch)
#             if f1 > val_best_metric:
#                 val_best_metric = f1
#                 torch.save(model.state_dict(), model_best_path)
#             torch.save(model.state_dict(), model_last_path)

if __name__ == '__main__':

    path_dataset = '/content/drive/MyDrive/tennis deep learning/TrackNet/Dataset'
    batch_size = 2
    exp_id = '1'
    num_epochs = 500
    lr = 1.0
    val_intervals = 5
    steps_per_epoch = 200
    


    train_dataset = TrackNetDataset(mode='train', path_dataset=path_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    val_dataset = TrackNetDataset(mode='val', path_dataset=path_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    model = TrackNet()
    device = 'cuda'
    model = model.to(device)

    exps_path = './exps/{}'.format(exp_id)
    tb_path = os.path.join(exps_path, 'plots')
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    log_writer = SummaryWriter(tb_path)
    model_last_path = os.path.join(exps_path, 'model_last.pt')
    model_best_path = os.path.join(exps_path, 'model_best.pt')

    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    val_best_metric = 0

    for epoch in range(num_epochs):
        loss_fn = torch.nn.CrossEntropyLoss()
        train_loss = train(model, train_loader, loss_fn, optimizer, device, epoch, steps_per_epoch)
        print(f"train_loss = {train_loss}")
        log_writer.add_scalar('Train/training_loss', train_loss, epoch)
        log_writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)

        if (epoch > 0) and (epoch & val_intervals == 0):
            val_loss, precision, recall, f1 = validate(model, val_loader, device, epoch)
            print(f"val_loss = {val_loss}")
            log_writer.add_scalar('Val/loss', val_loss, epoch)
            log_writer.add_scalar('Val/precision', precision, epoch)
            log_writer.add_scalar('Val/recall', recall, epoch)
            log_writer.add_scalar('Val/f1', f1, epoch)
            if f1 > val_best_metric:
                val_best_metric = f1
                torch.save(model.state_dict(), model_best_path)
            torch.save(model.state_dict(), model_last_path)
