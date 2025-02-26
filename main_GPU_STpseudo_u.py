from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.resnet_model import MetaResnet34
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset

parser = argparse.ArgumentParser(description='S+T+pseudo-U for Semi-supervised Domain Adaptation')
parser.add_argument('--steps', type=int, default=20000, metavar='N',
                    help='maximum number of iterations to train (default: 20000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./checkpoint',
                    help='directory to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='resnet34',
                    help='which network to use')
parser.add_argument('--source', type=str, default='real',
                    help='source domain')
parser.add_argument('--target', type=str, default='sketch',
                    help='target domain')
parser.add_argument('--st', type=int, default=0)
parser.add_argument('--dataset', type=str, default='multi')
parser.add_argument('--num', type=int, default=3,
                    help='3-shot/1-shot')
parser.add_argument('--th', type=float, default=0.5,
                    help='confidence threshold for pseudo-labeling')
parser.add_argument('--root', type=str, default='./')
parser.add_argument('--net_resume', type=str, default='')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--runs', type=int, default=999)
parser.add_argument('--eval', action='store_true', default=False)

args = parser.parse_args()

"""DomainNet-subset 7 adaptation scenarios"""
multi = [['real', 'clipart'], ['real', 'painting'],
         ['painting', 'clipart'], ['clipart', 'sketch'],
         ['sketch', 'painting'], ['real', 'sketch'],
         ['painting', 'real']]

if args.st != 0 and args.dataset == 'multi':
    args.source, args.target = multi[args.st-1]

print('Dataset %s Source %s Target %s Labeled num per class %s Network %s' %
      (args.dataset, args.source, args.target, args.num, args.net))

source_loader, target_loader, target_loader_unl, target_loader_val, \
    target_loader_test, class_list = return_dataset(args=args, return_idx=False)

# On utilise un seul modèle pour S+T+pseudo-U
use_gpu = torch.cuda.is_available()
if use_gpu:
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)

if args.net == 'resnet34':
    net = MetaResnet34(num_class=len(class_list))

# Récupération des paramètres du modèle
params = []
for value in net.G.params():
    if value.requires_grad:
        params += [{'params': [value], 'lr': args.multi, 'weight_decay': 0.0005}]
for value in net.F1.params():
    if value.requires_grad:
        params += [{'params': [value], 'lr': args.multi, 'weight_decay': 0.0005}]

""" record & resume path """
args.checkpath = os.path.join(args.checkpath, 'runs_{}'.format(args.runs))
if not os.path.exists(args.checkpath):
    os.makedirs(args.checkpath)

record_dir = './record/%s/self_training' % args.dataset
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir,
                           'exp_net_%s_%s_to_%s_num_%s_%d' %
                           (args.net, args.source, args.target, args.num, args.runs))

record_dir_confident_predictions = './record/%s/test_confident_predictions_STpseudo_u' % args.dataset
if not os.path.exists(record_dir_confident_predictions):
    os.makedirs(record_dir_confident_predictions)
record_dir_confident_predictions = os.path.join(record_dir_confident_predictions,
                              'exp_net_%s_%s_to_%s_num_%s_%d' %
                              (args.net, args.source, args.target, args.num, args.runs))

""" pre-train & resume """
pretrain_src_checkpoint = './pretrained_models/pretrained_src_{}_to_{}.pth.tar'.format(args.source, args.target)
if args.net_resume:
    p = os.path.join(args.checkpath, args.net_resume)
    net.load_state_dict(torch.load(p))
else:
    net.load_state_dict(torch.load(pretrain_src_checkpoint))

net.cuda()

lr = args.lr

# Allocation des tenseurs sur le GPU
im_data_s = torch.FloatTensor(1).cuda()
im_data_t = torch.FloatTensor(1).cuda()
im_data_tu = torch.FloatTensor(1).cuda()
gt_labels_s = torch.LongTensor(1).cuda()
gt_labels_t = torch.LongTensor(1).cuda()

im_data_s = Variable(im_data_s)
im_data_t = Variable(im_data_t)
im_data_tu = Variable(im_data_tu)
gt_labels_s = Variable(gt_labels_s)
gt_labels_t = Variable(gt_labels_t)

def train():
    net.train()

    optimizer = optim.SGD(params, momentum=0.9,
                          weight_decay=0.0005, nesterov=True)

    def zero_grad_all():
        optimizer.zero_grad()

    param_lr = [pg["lr"] for pg in optimizer.param_groups]

    criterion = nn.CrossEntropyLoss().cuda()
    criterion_no_reduce = nn.CrossEntropyLoss(reduction='none').cuda()

    all_step = args.steps
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_unl = len(target_loader_unl)
    best_acc = 0
    counter = 0

    for step in range(args.start, all_step):

        optimizer = inv_lr_scheduler(param_lr, optimizer, step, init_lr=args.lr)
        current_lr = optimizer.param_groups[0]['lr']

        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_target_unl == 0:
            data_iter_t_unl = iter(target_loader_unl)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)

        data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl)
        data_s = next(data_iter_s)

        with torch.no_grad():
            im_data_s.resize_(data_s[0].size()).copy_(data_s[0])
            gt_labels_s.resize_(data_s[1].size()).copy_(data_s[1])
            im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
            im_data_tu.resize_(data_t_unl[0].size()).copy_(data_t_unl[0])

        zero_grad_all()
        # Calcul de la perte sur la cible labellisée T
        out_t = net(im_data_t)
        loss_t = criterion_no_reduce(out_t, gt_labels_t).mean()
        zero_grad_all()

        # Calcul de la perte sur la source labellisée S
        out_s = net(im_data_s)
        loss_s = criterion_no_reduce(out_s, gt_labels_s).mean()
        zero_grad_all()

        # Génération des pseudo-labels sur U à partir du même modèle
        u_prob = torch.softmax(net(im_data_tu), dim=1)
        confidences, predictions = u_prob.max(1)
        mask = confidences >= args.th
        # Si aucune image ne dépasse le seuil, la perte sur pseudo-label sera nulle
        if mask.sum().item() > 0:
            im_u = im_data_tu[mask]
            psl_u = predictions[mask]
            out_u = net(im_u)
            loss_u = criterion(out_u, psl_u)
        else:
            loss_u = 0.0

        # Total loss : somme des pertes sur S, T et U pseudo-labellisées
        total_loss = loss_t + loss_s + loss_u

        zero_grad_all()
        total_loss.backward()
        optimizer.step()

        log_train = 'S {} T {} Train step: {} lr: {} Method: S+T+pseudo-U\n'.format(
            args.source, args.target, step, current_lr)
        if step % args.log_interval == 0:
            print(log_train + 'Loss: {:.4f} (T: {:.4f}, S: {:.4f}, pseudo-U: {:.4f})'.format(
                total_loss.item(), loss_t.item(), loss_s.item(), loss_u if isinstance(loss_u, float) else loss_u.item()))

        if step % args.save_interval == 0:
            acc_test, total_test, confident_predictions_test = test(net, target_loader_test)
            acc_val, total_val, confident_predictions_val = test(net, target_loader_val)
            net.train()
            if acc_val >= best_acc:
                best_acc = acc_val
                best_acc_test = acc_test
                counter = 0
            else:
                counter += 1

            print('Test acc: {} Best test acc: {} Best val acc: {}'.format(acc_test, best_acc_test, best_acc))
            with open(record_file, 'a') as f:
                f.write('step %d acc_test %f best_test %f best_val %f\n' %
                        (step, acc_test, best_acc_test, best_acc))
                
            print('record %s' % record_dir_confident_predictions)
            with open(record_dir_confident_predictions, 'a') as f:
                f.write('%d %d %d \n' %
                        (step, total_test, confident_predictions_test))
                
            if args.save_check:
                print('Saving model at step:', step)
                torch.save(net.state_dict(),
                           os.path.join(args.checkpath,
                                        "Model_iter_STpseudoU_{}_{}_to_{}_step_{}.pth.tar".format(
                                            args.method, args.source, args.target, step)))

def test(model, loader):
    model.eval()
    
    correct = 0
    total = 0
    
    confident_predictions = 0
    
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            im_data_t.resize_(data[0].size()).copy_(data[0])
            gt_labels_t.resize_(data[1].size()).copy_(data[1])
            
            outputs = model(im_data_t)
            # preds = outputs.max(1)[1]
            confidences, preds = outputs.max(1)
            confident_predictions += (confidences >= args.th).sum().item()
            
            total += gt_labels_t.size(0)
            correct += preds.eq(gt_labels_t).sum().item()
            
    acc = 100. * (float(correct) / total)
    
    return acc, total, confident_predictions

if __name__ == '__main__':
    if args.eval:
        print('Eval mode...')
        acc_test = test(net, target_loader_test)
        print('Model acc: {}'.format(acc_test))
    else:
        train()
