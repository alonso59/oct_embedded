import torch
from tqdm import tqdm
from .callbacks import TensorboardWriter
from .metrics import MIoU
import matplotlib.pyplot as plt
import numpy as np
from training.metric import SegmentationMetrics
import torch.nn.functional as F
from sklearn.metrics import classification_report, multilabel_confusion_matrix

def trainer(num_epochs, train_loader, val_loader, model, optimizer, loss_fn, device, checkpoint_path, scheduler,
            name_model, callback_stop_value, tb_dir, logger):
    writer = TensorboardWriter(name_dir=tb_dir + 'tb_' + name_model + '/')
    train_iter = 0.0
    val_iter = 0.0
    stop_early = 0

    train_best_loss = float("inf")
    train_best_dice = 0.0
    valid_best_loss = float("inf")
    valid_best_dice = 0.0

    train_metrics_list = []
    valid_metrics_list = []

    train_loss_history = []
    val_loss_history= []

    for epoch in range(num_epochs):
        lr_ = optimizer.param_groups[0]["lr"]
        str = f"Epoch: {epoch+1}/{num_epochs} --loss_fn:{loss_fn.__name__} --model:{name_model} --lr:{lr_:.4e}"
        logger.info(str)

        train_loss, train_metrics, train_iter = train(train_loader, model, optimizer, loss_fn, writer, train_iter, device)
        
        val_loss, valid_metrics, val_iter = validation(val_loader, model, loss_fn, writer, val_iter, device)

        train_metrics_list.append(train_metrics)
        valid_metrics_list.append(valid_metrics)
        # """ scheduler learning rate """
        scheduler.step()
        writer.learning_rate(optimizer.param_groups[0]["lr"], epoch)
        writer.loss_epoch(train_loss, val_loss, epoch)

        writer.metrics_epoch(
            train_metric=np.array(train_metrics_list)[-1, 1],
            val_metric=np.array(valid_metrics_list)[-1, 1],
            step=epoch, metric_name='Dice')

        writer.metrics_epoch(
            train_metric=np.array(train_metrics_list)[-1, 4],
            val_metric=np.array(valid_metrics_list)[-1, 4],
            step=epoch, metric_name='MIoU')

        # """ Saving the model """
        if np.array(valid_metrics_list)[-1, 1] > valid_best_dice:
            str_print = f"Valid dice Coef. improved from {valid_best_dice:2.5f} to {np.array(valid_metrics_list)[-1, 1]:2.5f}. Saving checkpoint: {checkpoint_path}"
            best_valid_loss = val_loss
            torch.save(model, checkpoint_path + f'/model.pth')
            torch.save(model.state_dict(), checkpoint_path + f'/weights.pth')
            stop_early = 0
            valid_best_dice = np.array(valid_metrics_list)[-1, 1]
        else:
            stop_early += 1
            str_print = f"Valid Dice not improved: {valid_best_dice:2.5f}, Val. Loss: {best_valid_loss:2.5f}, ESC: {stop_early}/{callback_stop_value} \nCheckpoint path: {checkpoint_path}"
        if stop_early == callback_stop_value:
            logger.info('+++++++++++++++++ Stop training early +++++++++++++')
            break
        if np.array(train_metrics_list)[-1, 1] > train_best_dice:
            # save last training
            torch.save(model, checkpoint_path + f'/model_last.pth')
            torch.save(model.state_dict(), checkpoint_path + f'/weights_last.pth')

        logger.info(f'----> Train Loss: {train_loss:.5f} \t Val. Loss: {val_loss:.5f}')
        metric_names = ['Pixel Acc', 'Dice Coef', 'Precision', 'Recall', 'mIoU']
        for idx, name in enumerate(metric_names):
            logger.info(f'----> Train {name}: {np.array(train_metrics_list)[-1, idx]:.5f} \t Val. {name}: {np.array(valid_metrics_list)[-1, idx]:0.5f}')
        logger.info(str_print)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

    plot_results(np.array(train_loss_history), np.array(val_loss_history), 'Loss', checkpoint_path)

    metric_names = ['Pixel Acc', 'Dice', 'Precision', 'Recall', 'mIoU']
    for idx, name in enumerate(metric_names):
        plot_results(np.array(train_metrics_list)[idx, :], np.array(train_metrics_list)[idx, : ], metric_names[idx], checkpoint_path)

def train(loader, model, optimizer, loss_fn, writer, iterations, device):
    loss_acum = 0.0
    pixel_acc = 0.0
    dice = 0.0
    precision = 0.0
    recall = 0.0
    f1_score = 0.0
    mean_iou = 0.0

    loop = tqdm(loader, ncols=150, ascii=False)
    
    iou_fn = MIoU(ignore_background=True, activation='softmax', device=device)

    metrics = SegmentationMetrics(ignore_background=True, activation='softmax', average=True)

    model.train()
    for batch_idx, (x, y) in enumerate(loop):
        x = x.type(torch.float).to(device)
        y = y.type(torch.long).to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        y_pred = model(x)
        # loss function
        loss = loss_fn(y_pred, y)
        loss_acum += loss.item()
        # backward
        loss.backward()
        optimizer.step()
        # metrics
        iou = iou_fn(y_pred, y)
        metrics_value = metrics(y, y_pred)

        pixel_acc += metrics_value[0].mean()
        dice += metrics_value[1].mean()
        precision += metrics_value[2].mean()
        recall += metrics_value[3].mean()
        f1_score += metrics_value[4].mean()
        mean_iou += iou.mean()
        # update tqdm loop
        loop.set_postfix(
            Pixel_acc=metrics_value[0].mean(),
            Dice=metrics_value[1].mean(),
            Precision=metrics_value[2].mean(),
            Recall=metrics_value[3].mean(),
            Loss=loss.item(),
            mIoU=iou.mean(),
            f1_score=metrics_value[4].mean()
        )

        # tensorboard callbacks
        # writer.loss_iter(loss.item(), iterations, stage='Train')

        # metric_names = ['Pixel Acc', 'Dice', 'Precision', 'Recall', 'mIoU']

        # for i, names in enumerate(metric_names):
        #     writer.metric_iter(metrics_values[i], iterations, stage='Val', metric_name=names)

        if iterations % (len(loader) * 5) == 0 and writer is not None:
            writer.save_images(x, y, y_pred, iterations, device, tag='train')

        iterations = iterations + 1
    return loss_acum / len(loader), [pixel_acc / len(loader),
                                    dice / len(loader),
                                    precision / len(loader),
                                    recall / len(loader),
                                    mean_iou / len(loader)], iterations

def validation(loader, model, loss_fn, writer, iterations, device):
    loss_acum = 0.0
    pixel_acc = 0.0
    dice = 0.0
    precision = 0.0
    recall = 0.0
    f1_score = 0.0
    mean_iou = 0.0
    loop = tqdm(loader, ncols=150, ascii=False)
    
    iou_fn = MIoU(ignore_background=True, activation='softmax', device=device)
    metrics = SegmentationMetrics(ignore_background=True, activation='softmax', average=True)
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loop):
            x = x.type(torch.float).to(device)
            y = y.type(torch.long).to(device)
            # forward
            y_pred = model(x)
            # loss function
            loss = loss_fn(y_pred, y)
            loss_acum += loss.item()
            # metrics
            iou = iou_fn(y_pred, y)
            metrics_value = metrics(y, y_pred)

            pixel_acc += metrics_value[0].mean()
            dice += metrics_value[1].mean()
            precision += metrics_value[2].mean()
            recall += metrics_value[3].mean()
            f1_score += metrics_value[4].mean()
            mean_iou += iou.mean()
            # update tqdm loop
            loop.set_postfix(
                Pixel_acc=metrics_value[0].mean(),
                Dice=metrics_value[1].mean(),
                Precision=metrics_value[2].mean(),
                Recall=metrics_value[3].mean(),
                Loss=loss.item(),
                mIoU=iou.mean(),
                f1_score=metrics_value[4].mean()
            )
                # tensorboard callbacks
            # writer.loss_iter(loss.item(), iterations, stage='Train')

            # metric_names = ['Pixel Acc', 'Dice', 'Precision', 'Recall', 'mIoU']

            # for i, names in enumerate(metric_names):
            #     writer.metric_iter(metrics_values[i], iterations, stage='Val', metric_name=names)

            if iterations % (len(loader) * 5) == 0 and writer is not None:
                writer.save_images(x, y, y_pred, iterations, device, tag='val')

            iterations = iterations + 1
        return loss_acum / len(loader), [pixel_acc / len(loader),
                                        dice / len(loader),
                                        precision / len(loader),
                                        recall / len(loader),
                                        mean_iou / len(loader)], iterations

def eval(model, loader, loss_fn, device):
    loss_acum = 0.0
    loop = tqdm(loader, ncols=150, ascii=False)
    iou_fn = MIoU(activation='softmax', ignore_background=False, device=device)
    metrics = SegmentationMetrics(ignore_background=False, activation='softmax', average=False)
    model.eval()
    pixel_acc_list = []
    dice_list = []
    precision_list = []
    recall_list = []
    iou_list = []
    y_preds = []
    y_trues = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loop):
            x = x.type(torch.float).to(device)
            y = y.type(torch.long).to(device)
            # forward
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            # Metrics
            iou = iou_fn(y_pred, y)
            metrics_value = metrics(y, y_pred)
            loss_acum += loss.item()
            iou_list.append(iou)
            pixel_acc_list.append(metrics_value[0])
            dice_list.append(metrics_value[1])
            precision_list.append(metrics_value[2])
            recall_list.append(metrics_value[3])
            y_pred1 = F.softmax(y_pred, dim=1)
            y_pred1 = torch.argmax(y_pred1, dim=1)
            y_pred1 = torch.flatten(y_pred1).detach().cpu().numpy()
            y1 = torch.flatten(y).detach().cpu().numpy()
            y_preds.append(list(y_pred1))
            y_trues.append(list(y1))

    y_preds = np.array(y_preds).reshape(-1)
    y_trues = np.array(y_trues).reshape(-1)
    target_names = np.array(['BG', 'EZ', 'OPL', 'ELM', 'BM'])
    print(classification_report(y_trues, y_preds, target_names=target_names))
    cf_mtx = multilabel_confusion_matrix(y_trues, y_preds)
    return loss_acum/len(loader), iou_list, pixel_acc_list, dice_list, precision_list, recall_list

def evaluation(model, loader, loss_fn, device):
    loss_acum = 0.0
    loop = tqdm(loader, ncols=150, ascii=True)
    iou_fn = MIoU(activation='softmax', ignore_background=False, device=device)
    metrics = SegmentationMetrics(ignore_background=False, activation='softmax', average=True)
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loop):
            x = x.type(torch.float).to(device)
            y = y.type(torch.long).to(device)
            # forward
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            # loss function
            loss_acum += loss.item()
            # metrics
            iou = iou_fn(y_pred, y)
            metrics_value = metrics(y, y_pred)
    return loss_acum / len(loader), iou.mean(), metrics_value

def plot_results(train, val, name, checkpoint_path):
    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(train, linestyle='-', linewidth=1)
    plt.plot(val, linestyle='--', linewidth=1)
    plt.title(name)
    plt.grid(color='lightgray', linestyle='-', linewidth=2)
    plt.ylabel(name)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.savefig(checkpoint_path + name + '.png')
