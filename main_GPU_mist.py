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

parser = argparse.ArgumentParser(description='MIST: MixUp Self-Training for Semi-supervised Domain Adaptation')
parser.add_argument('--steps', type=int, default=20000, metavar='N',
                    help='maximum number of iterations to train (default: 20000)')
parser.add_argument('--method', type=str, default='mist',
                    choices=['mico', 'mist'])
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./checkpoint',
                    help='dir to save checkpoint')
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
                    help='confidence threshold for pseudo-labels')
parser.add_argument('--root', type=str, default='./')
parser.add_argument('--net_resume', type=str, default='') 
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--runs', type=int, default=999)
parser.add_argument('--eval', action='store_true', default=False)

args = parser.parse_args()

"""DomainNet-subset 7 adaptation scenarios"""
multi = [['real', 'clipart'], ['real', 'painting'],
         ['painting', 'clipart'], ['clipart','sketch'],
         ['sketch', 'painting'], ['real', 'sketch'],
         ['painting', 'real']]

if args.st != 0 and args.dataset == 'multi':
    args.source, args.target = multi[args.st-1]

print('Dataset %s Source %s Target %s Labeled num perclass %s Network %s' %
      (args.dataset, args.source, args.target, args.num, args.net))

source_loader, target_loader, target_loader_unl, target_loader_val, \
    target_loader_test, class_list = return_dataset(args=args, return_idx=False)

# Définition du device : GPU si disponible, sinon CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)

if args.net == 'resnet34':
    model = MetaResnet34(num_class=len(class_list))

# Récupération des paramètres du modèle : G pour la convolution et F pour la tête de classification
params = []
for value in model.G.params():
    if value.requires_grad:
        params += [{'params': [value], 'lr': args.multi, 'weight_decay': 0.0005}]
for value in model.F1.params():
    if value.requires_grad:
        params += [{'params': [value], 'lr': args.multi, 'weight_decay': 0.0005}]

""" record & resume path """
args.checkpath = os.path.join(args.checkpath, 'runs_{}'.format(args.runs))
if not os.path.exists(args.checkpath):
    os.makedirs(args.checkpath)

record_dir = os.path.join('./record', args.dataset, 'mist')
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir,
                           'exp_net_%s_%s_to_%s_num_%s_%d' %
                           (args.net, args.source, args.target, args.num, args.runs))

# record_dir_3a = './record/%s/test_classifier_mist' % args.dataset
# if not os.path.exists(record_dir_3a):
#     os.makedirs(record_dir_3a)
# record_file_3a = os.path.join(record_dir_3a,
#                               'exp_net_%s_%s_to_%s_num_%s_%d' %
#                               (args.net, args.source, args.target, args.num, args.runs))

record_dir_confident_predictions = './record/%s/test_confident_predictions_mist' % args.dataset
if not os.path.exists(record_dir_confident_predictions):
    os.makedirs(record_dir_confident_predictions)
record_dir_confident_predictions = os.path.join(record_dir_confident_predictions,
                              'exp_net_%s_%s_to_%s_num_%s_%d' %
                              (args.net, args.source, args.target, args.num, args.runs))

""" pre-train & resume """
# On charge ici un modèle pré-entraîné sur la source (ou sur S+T si disponible)
pretrain_checkpoint = './pretrained_models/pretrained_src_{}_to_{}.pth.tar'.format(args.source, args.target)

if args.net_resume:
    p = os.path.join(args.checkpath, args.net_resume)
    model.load_state_dict(torch.load(p))
else:
    print(f"Loading checkpoint from: {pretrain_checkpoint}")
    model.load_state_dict(torch.load(pretrain_checkpoint, map_location=device))

model.to(device)

lr = args.lr

# Préparation des tenseurs pour les batches
im_data_s = torch.FloatTensor(1).to(device)
im_data_t = torch.FloatTensor(1).to(device)
im_data_tu = torch.FloatTensor(1).to(device)
gt_labels_s = torch.LongTensor(1).to(device)
gt_labels_t = torch.LongTensor(1).to(device)

im_data_s = Variable(im_data_s)
im_data_t = Variable(im_data_t)
im_data_tu = Variable(im_data_tu)
gt_labels_s = Variable(gt_labels_s)
gt_labels_t = Variable(gt_labels_t)

def train():
    model.train()

    optimizer = optim.SGD(params, momentum=0.9,
                          weight_decay=0.0005, nesterov=True)

    def zero_grad_all():
        optimizer.zero_grad()

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])

    criterion = nn.CrossEntropyLoss().to(device)  # Utilisation du device (GPU ou CPU)
    criterion_no_reduce = nn.CrossEntropyLoss(reduction='none').to(device)
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
        print(f'step : {step}')
        
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
        # Loss sur les données labellisées de la cible (T)
        out_t = model(im_data_t)
        loss_t = criterion_no_reduce(out_t, gt_labels_t).mean()

        zero_grad_all()
        # Loss sur les données labellisées de la source (S)
        out_s = model(im_data_s)
        loss_s = criterion_no_reduce(out_s, gt_labels_s).mean()

        # Pseudo-label sur les données non labellisées (U) via le même modèle
        u_prob = torch.softmax(model(im_data_tu), dim=1)
        confidences, predictions = u_prob.max(1)    # niveau de confiance et prédictions
        mask = confidences >= args.th               # seuil de confiance
        im_u = im_data_tu[mask]
        psl_u = predictions[mask]

        # MixUp : tirage de lambda depuis une loi Beta(1,1) (loi uniforme)
        alpha = 1
        lam = np.random.beta(alpha, alpha)

        # MixUp entre les données cibles labellisées et pseudo-labellisées
        loss_mix_t = 0
        if im_u.size(0) > 0:    # vérifie qu'il y a des images avec confiance suffisante
            size_t = im_u.size(0)
            t_idx = torch.randperm(im_data_t.size(0))[0:size_t]
            mixed_x_t = lam * im_data_t[t_idx] + (1-lam) * im_u 
            y_a_t, y_b_t = gt_labels_t[t_idx], psl_u
            out_mix_t = model(mixed_x_t)
            loss_mix_t = lam * criterion(out_mix_t, y_a_t) + (1-lam) * criterion(out_mix_t, y_b_t)

        # MixUp entre les données sources labellisées et pseudo-labellisées
        loss_mix_s = 0
        if im_u.size(0) > 0:
            size_s = im_u.size(0)
            s_idx = torch.randperm(im_data_s.size(0))[0:size_s]
            mixed_x_s = lam * im_data_s[s_idx] + (1-lam) * im_u 
            y_a_s, y_b_s = gt_labels_s[s_idx], psl_u
            out_mix_s = model(mixed_x_s)
            loss_mix_s = lam * criterion(out_mix_s, y_a_s) + (1-lam) * criterion(out_mix_s, y_b_s)

        total_loss = loss_t + loss_s + loss_mix_t + loss_mix_s

        zero_grad_all()
        total_loss.backward()
        optimizer.step()

        log_train = 'S {} T {} Train step: {} lr: {} Method: {}\n'.format(
            args.source, args.target, step, current_lr, args.method)
        if step % args.log_interval == 0:
            print(log_train)

        if step % args.save_interval == 0:
            acc_test, total_test, confident_predictions_test = test(model, target_loader_test)
            acc_val, total_val, confident_predictions_val = test(model, target_loader_val)
            model.train()
            if acc_val >= best_acc:
                best_acc = acc_val
                best_acc_test = acc_test
                counter = 0
            else:
                counter += 1

            print('Test acc: {} Best test acc: {} Best val acc: {}'.format(acc_test, best_acc_test, best_acc))
            
            print('Record file: {}'.format(record_file))
            with open(record_file, 'a') as f:
                f.write('step %d acc_test %f best_test %f best_val %f\n' %
                        (step, acc_test, best_acc_test, best_acc))
            
            # Hboth, Hone, Hnone = test_classifier_f3a(target_loader_test)
            
            # print('record %s' % record_dir_3a)
            # with open(record_file_3a, 'a') as f:
            #     f.write('%d  %d %d %d \n' % (step, Hboth, Hone, Hnone))
            
            print('record %s' % record_dir_confident_predictions)
            with open(record_dir_confident_predictions, 'a') as f:
                f.write('%d %d %d \n' %
                        (step, total_test, confident_predictions_test))
            
            model.train()
            if args.save_check:
                print(f'Saving model at step: {step}')
                torch.save(model.state_dict(),
                           os.path.join(args.checkpath,
                                        "Model_iter_mist_{}_to_{}_step_{}.pth.tar".format(args.source, args.target, step)))

def test(model, loader):
    
    model.eval()
    correct = 0
    total = 0
    
    confident_predictions = 0
    
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            im_data_t.resize_(data[0].size()).copy_(data[0])
            gt_labels_t.resize_(data[1].size()).copy_(data[1])
            
            output = model(im_data_t)
            # pred = output.max(1)[1]
            confidences, pred = output.max(1)
            confident_predictions += (confidences >= args.th).sum().item()
            
            total += gt_labels_t.size(0)
            correct += pred.eq(gt_labels_t).sum().item()
            confident_predictions += (confidences >= args.th).sum().item()
            
    acc = 100. * (float(correct)/total)
    return acc, total, confident_predictions

# def test_classifier_f3a(loader):
#     model.eval()
    
#     Hboth = 0
#     Hone = 0
#     Hnone = 0
    
#     with torch.no_grad():
#         for batch_idx, data_t in enumerate(loader):
#             im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
#             gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
#             output1 = net(im_data_t)
#             output2 = twin(im_data_t)
            
#             u_1_prob = torch.softmax(net(im_data_t), dim=1)
#             u_1_pred = u_1_prob.max(1)
#             u_2_prob = torch.softmax(twin(im_data_t), dim=1)
#             u_2_pred = u_2_prob.max(1)
            
#             u_1_mask = u_1_pred[0] >= args.th
#             u_2_mask = u_2_pred[0] >= args.th
        
#             if len(u_1_mask) != len(u_2_mask):
#                 raise ValueError("Les deux listes doivent avoir la même taille")
    
#             for v1, v2 in zip(u_1_mask, u_2_mask):
#                 if v1 == v2 == True:
#                     Hboth += 1
#                 elif v1 == v2 == False:
#                     Hnone += 1
#                 elif v1 != v2:
#                     Hone += 1
            
#             print(Hboth, Hone, Hnone)

#     return Hboth, Hone, Hnone

if __name__ == '__main__':
    if args.eval:
        print('Eval mode...')
        acc_test = test(model, target_loader_test)
        print('Model acc: {}'.format(acc_test))
    else:
        train()
