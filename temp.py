# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 21:59:30 2018

@author: Allen
"""


train_params = {
    'model_save_name': model_save_name,
    'save_model_every': save_model_every,
    'num_epochs': num_epochs,
    'print_every': print_every,
    'log': log,
    'print_every': print_every,
    'mask_cutoff': 0,
    'model_save_iou_threshold': 0.8
    }


def log_iter_stats(y_pred, y_batch, X_batch, y_batch, train_params,
                    other_data, epoch_losses, start=start):
    epoch_losses = [round(e.item(),4) for e in torch.stack(epoch_losses).mean(0)]
    iou_batch = calc_mean_iou(y_pred.ge(train_params['mask_cutoff']), y_batch)
    iou_acc = calc_clf_accuracy(y_pred.ge(train_params['mask_cutoff']), y_batch)

    log.info('Losses: {}, Batch IOU: {:.4f}, Batch Acc: {:.4f} at iter {}, epoch {}, Time: {}'.format(
            epoch_losses, iou_batch, iou_acc, iter_count, epoch, timeSince(start))
    )

    X_train = other_data['X_train']
    X_val = other_data['X_val']
    y_train = other_data['y_train']
    y_val = other_data['y_val']
    X_train_mean_img = other_data['X_train_mean_img']
    #print(all_losses)
    X_orig = X_train[X_id[0]].squeeze()/255
    X_tsfm = X_batch[0,0].squeeze().cpu().detach().numpy()
    X_tsfm = X_tsfm[13:114,13:114]
    y_orig = y_train[X_id[0]].squeeze()
    y_tsfm = (y_batch[0].squeeze().cpu().detach().numpy())
    y_tsfm_pred =  y_pred[0].squeeze().gt(mask_cutoff)
    plot_img_mask_pred([X_orig, X_tsfm, y_orig, y_tsfm, y_tsfm_pred],
                       ['X Original', 'X Transformed', 'y Original', 'y Transformed', 'y Predicted'])
    

def log_epoch_stats(pred_vs_true_epoch, mask_cutoff, phase, epoch, best_iou, best_model):    
    y_pred_epoch = torch.cat([e[0] for e in pred_vs_true_epoch])
    y_true_epoch = torch.cat([e[1] for e in pred_vs_true_epoch])

    mean_iou_epoch = calc_mean_iou(y_pred_epoch.ge(mask_cutoff), y_true_epoch.float())
    mean_acc_epoch = calc_clf_accuracy(y_pred_epoch.ge(mask_cutoff), y_true_epoch.float())
    log.info('{} Mean IOU: {:.4f}, Mean Acc: {:.4f}, Best Val IOU: {:.4f} at epoch {}'.format(phase, mean_iou_epoch, mean_acc_epoch, best_iou, epoch))

    if phase == 'val' and mean_iou_epoch > best_iou:
        best_iou = mean_iou_epoch
        stats = {'best_iou': best_iou,
                 'all_losses': all_losses,
                 'iter_count': iter_count}
        best_model = (epoch, copy.deepcopy(model.state_dict()),
                                            copy.deepcopy(optimizer.state_dict()),
                                            copy.deepcopy(scheduler.state_dict()), stats, model_save_name, '.')
        log.info(save_model_state_to_chunks(*best_model))
        
        log.info('Best Val Mean IOU so far: {}'.format(best_iou))
        # Visualize 1 val sample and predictions
        X_orig = X_val[X_id[0]].squeeze()/255
        y_orig = y_val[X_id[0]].squeeze()
        y_pred2 =  y_pred[0].squeeze().gt(mask_cutoff)
        plot_img_mask_pred([X_orig, y_orig, y_pred2],
                           ['Val X Original', 'Val y Original', 'Val y Predicted'])

    return best_iou, best_model

def train_model(model, dataloaders, loss_fns, loss_fn_weights, optimizer, scheduler, train_params, other_data):
    log.info('Start Training...')    
    num_epochs = train_params['num_epochs']
    start = time.time()
    if torch.cuda.is_available():
        model.cuda()
    best_model = None
    best_iou = prev_best_iou = 0.0    
    all_losses = []
    iter_count = 0

    for epoch in range(1, num_epochs+1):
        log.info('Epoch {}/{}'.format(epoch, num_epochs))
        log.info('-' * 20)
        if save_log_every is not None:
            if (epoch % save_log_every == 0):
                push_log_to_git()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step(loss.item())
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            epoch_losses = []
            pred_vs_true_epoch = []

            for X_batch, y_batch, d_batch, X_id in dataloaders[phase]:
                # zero the parameter gradients
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    y_pred = model(X_batch)
                    pred_vs_true_epoch.append([y_pred, y_batch])
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        losses = calc_loss(y_pred, y_batch.float(), loss_fns, loss_fn_weights)
                        epoch_losses.append(losses)
                        all_losses.append(losses)
                        loss = losses[-1]
                        loss.backward()
                        optimizer.step()
                        iter_count += 1
                if (phase == 'train') & (iter_count % print_every == 0):
                    log_iter_stats(y_pred, y_batch, X_batch, y_batch, 
                                   train_params, other_data, epoch_losses, start=start)
            best_iou, best_model = log_epoch_stats(pred_vs_true_epoch, train_params['mask_cutoff'], 
                                                   phase, epoch, best_iou, best_model)
            

        if (epoch % train_params['save_model_every'] == 0) | (epoch == num_epochs-1):
            if (best_model is not None) and (best_iou > train_params['model_save_iou_threshold']):
                log.info(save_model_state_to_chunks(*best_model))
                push_model_to_git(ckp_name=model_save_name)
                prev_best_iou = best_iou
            else:
                log.info("Skip pushing model to git as there's no improvement")

    # load best model weights
    model.load_state_dict(best_model[1])
    log.info('-' * 20)
    time_elapsed = time.time() - start
    log.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    log.info('Best val IOU: {:4f}'.format(best_iou))

    return model

print('test')

