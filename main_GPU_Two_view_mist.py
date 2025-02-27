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

parser = argparse.ArgumentParser(description='Two-view MIST for Semi-supervised Domain Adaptation')
parser.add_argument('--steps', type=int, default=20000, metavar='N',
                    help='maximum number of iterations to train (default: 20000)')
parser.add_argument('--method', type=str, default='TwoViewMist',
                    choices=['TwoViewMist'])
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
parser.add_argument('--th', type=float, default=0.5, help='confidence threshold')
parser.add_argument('--root', type=str, default='./')
parser.add_argument('--net_resume', type=str, default='')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--runs', type=int, default=999)
parser.add_argument('--eval', action='store_true', default=False)
parser.add_argument('--test_classifier_3a', action='store_true', default=False)

args = parser.parse_args()

""" DomainNet-subset : 7 scenarios d'adaptation """
multi = [['real', 'clipart'], ['real', 'painting'],
         ['painting', 'clipart'], ['clipart', 'sketch'],
         ['sketch', 'painting'], ['real', 'sketch'],
         ['painting', 'real']]
if args.st != 0 and args.dataset == 'multi':
    args.source, args.target = multi[args.st-1]

print('Dataset %s | Source %s | Target %s | Labeled num per class %s | Network %s' %
      (args.dataset, args.source, args.target, args.num, args.net))

source_loader, target_loader, target_loader_unl, target_loader_val, \
    target_loader_test, class_list = return_dataset(args=args, return_idx=False)

# Définition du device : GPU si disponible, sinon CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)
    
# Instanciation de deux réseaux pour les deux vues
if args.net == 'resnet34':
    net = MetaResnet34(num_class=len(class_list))
    twin = MetaResnet34(num_class=len(class_list))

# Constitution des groupes de paramètres pour chaque réseau
# Pour net
params = []
for value in net.G.params():
    if value.requires_grad:
        params.append({'params': [value], 'lr': args.multi, 'weight_decay': 0.0005})
params_F1 = []
for value in net.F1.params():
    if value.requires_grad:
        params_F1.append({'params': [value], 'lr': args.multi, 'weight_decay': 0.0005})
# Pour twin
params_2 = []
for value in twin.G.params():
    if value.requires_grad:
        params_2.append({'params': [value], 'lr': args.multi, 'weight_decay': 0.0005})
params_F2 = []
for value in twin.F1.params():
    if value.requires_grad:
        params_F2.append({'params': [value], 'lr': args.multi, 'weight_decay': 0.0005})

""" record & resume path """
args.checkpath = os.path.join(args.checkpath, 'runs_{}'.format(args.runs))
if not os.path.exists(args.checkpath):
    os.makedirs(args.checkpath)

record_dir = os.path.join('./record', args.dataset, 'two_view_mist')
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir,
                           'exp_net_%s_%s_to_%s_num_%s_%d' %
                           (args.net, args.source, args.target, args.num, args.runs))

record_dir_confident_predictions = './record/%s/test_confident_predictions_TwoViewMiST' % args.dataset
if not os.path.exists(record_dir_confident_predictions):
    os.makedirs(record_dir_confident_predictions)
record_dir_confident_predictions = os.path.join(record_dir_confident_predictions,
                              'exp_net_%s_%s_to_%s_num_%s_%d.txt' %
                              (args.net, args.source, args.target, args.num, args.runs))

# Chargement des checkpoints pré-entraînés
pretrain_src_checkpoint = './pretrained_models/pretrained_src_{}_to_{}.pth.tar'.format(args.source, args.target)
pretrain_t_checkpoint = './pretrained_models/pretrained_tgt_{}_to_{}.pth.tar'.format(args.source, args.target)

if args.net_resume:
    p1 = os.path.join(args.checkpath, args.net_resume)
    p2 = os.path.join(args.checkpath, 'Twin' + args.net_resume[3:])
    net.load_state_dict(torch.load(p1))
    twin.load_state_dict(torch.load(p2))
else:
    # Pour two-view MIST, on charge chacun sur sa cible (self‑training)
    print("Loading checkpoint for net from:", pretrain_t_checkpoint)
    net.load_state_dict(torch.load(pretrain_t_checkpoint, map_location=torch.device('cpu')))
    print("Loading checkpoint for twin from:", pretrain_t_checkpoint)
    twin.load_state_dict(torch.load(pretrain_t_checkpoint, map_location=torch.device('cpu')))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)
twin.to(device)

# Préparation des tenseurs d'entrée partagés
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
    net.train()
    twin.train()

    # Optimiseurs distincts pour chaque réseau
    optimizer_g = optim.SGD(params, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(params_F1, lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_g_2 = optim.SGD(params_2, momentum=0.9,
                              weight_decay=0.0005, nesterov=True)
    optimizer_f_2 = optim.SGD(params_F2, lr=1.0, momentum=0.9,
                              weight_decay=0.0005, nesterov=True)

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        optimizer_g_2.zero_grad()
        optimizer_f_2.zero_grad()

    # Sauvegarde des taux d'apprentissage initiaux
    param_lr = [group["lr"] for group in optimizer_g.param_groups]
    param_lr_f = [group["lr"] for group in optimizer_f.param_groups]
    param_lr_2 = [group["lr"] for group in optimizer_g_2.param_groups]
    param_lr_f2 = [group["lr"] for group in optimizer_f_2.param_groups]

    all_step = args.steps
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_unl = len(target_loader_unl)
    best_acc = 0
    counter = 0
    base_lr = args.lr

    criterion = nn.CrossEntropyLoss().to(device)
    criterion_no_reduce = nn.CrossEntropyLoss(reduction='none').to(device)
    alpha = 1  # coefficient pour MixUp

    for step in range(args.start, all_step):
        print(f'step : {step}')

        optimizer_g = inv_lr_scheduler(param_lr, optimizer_g, step, init_lr=base_lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step, init_lr=base_lr)
        optimizer_g_2 = inv_lr_scheduler(param_lr_2, optimizer_g_2, step, init_lr=base_lr)
        optimizer_f_2 = inv_lr_scheduler(param_lr_f2, optimizer_f_2, step, init_lr=base_lr)

        # Réinitialisation des itérateurs si besoin
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
            # Données source
            im_data_s.resize_(data_s[0].size()).copy_(data_s[0].to(device))
            gt_labels_s.resize_(data_s[1].size()).copy_(data_s[1].to(device))
            # Données cible étiquetées
            im_data_t.resize_(data_t[0].size()).copy_(data_t[0].to(device))
            gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1].to(device))
            # Données cible non étiquetées
            im_data_tu.resize_(data_t_unl[0].size()).copy_(data_t_unl[0].to(device))

        ##############################################
        # Mise à jour du réseau "net" (vue 1) avec self‑training
        ##############################################
        zero_grad_all()
        # Calcul de la perte sur les données étiquetées pour net
        out_t_net = net(im_data_t)
        loss_target_net = criterion_no_reduce(out_t_net, gt_labels_t).mean()
        out_s_net = net(im_data_s)
        loss_source_net = criterion_no_reduce(out_s_net, gt_labels_s).mean()

        # Pseudo‑étiquetage (self‑prediction) par net
        u_net_prob = torch.softmax(net(im_data_tu), dim=1)
        u_net_pred = u_net_prob.max(1)  # (valeurs max, indices)
        mask_net = u_net_pred[0] >= args.th
        im_u_net = im_data_tu[mask_net]
        psl_net = u_net_pred[1][mask_net]

        lam = np.random.beta(alpha, alpha)
        loss_mix_net = 0
        loss_mix_net2 = 0
        if im_u_net.size(0) > 0:
            size_net = im_u_net.size(0)
            # MixUp avec les données cible pour net
            t_idx = torch.randperm(im_data_t.size(0))[:size_net]
            mixed_x_net = lam * im_data_t[t_idx] + (1-lam) * im_u_net
            y_a_net = gt_labels_t[t_idx]
            y_b_net = psl_net
            out_mix_net = net(mixed_x_net)
            loss_mix_net = lam * criterion(out_mix_net, y_a_net) + (1-lam) * criterion(out_mix_net, y_b_net)
            # MixUp avec les données source pour net
            s_idx = torch.randperm(im_data_s.size(0))[:size_net]
            mixed_x_net2 = (1-lam) * im_data_s[s_idx] + lam * im_u_net
            y_a_net2 = gt_labels_s[s_idx]
            y_b_net2 = psl_net
            out_mix_net2 = net(mixed_x_net2)
            loss_mix_net2 = (1-lam) * criterion(out_mix_net2, y_a_net2) + lam * criterion(out_mix_net2, y_b_net2)

        total_loss_net = loss_source_net + loss_target_net + loss_mix_net + loss_mix_net2
        total_loss_net.backward()
        optimizer_f.step()
        optimizer_g.step()

        ##############################################
        # Mise à jour du réseau "twin" (vue 2) avec self‑training
        ##############################################
        zero_grad_all()
        out_t_twin = twin(im_data_t)
        loss_target_twin = criterion_no_reduce(out_t_twin, gt_labels_t).mean()
        out_s_twin = twin(im_data_s)
        loss_source_twin = criterion_no_reduce(out_s_twin, gt_labels_s).mean()

        # Pseudo‑étiquetage par twin
        u_twin_prob = torch.softmax(twin(im_data_tu), dim=1)
        u_twin_pred = u_twin_prob.max(1)
        mask_twin = u_twin_pred[0] >= args.th
        im_u_twin = im_data_tu[mask_twin]
        psl_twin = u_twin_pred[1][mask_twin]

        lam = np.random.beta(alpha, alpha)
        loss_mix_twin = 0
        loss_mix_twin2 = 0
        if im_u_twin.size(0) > 0:
            size_twin = im_u_twin.size(0)
            # MixUp avec les données cible pour twin
            t_idx = torch.randperm(im_data_t.size(0))[:size_twin]
            mixed_x_twin = lam * im_data_t[t_idx] + (1-lam) * im_u_twin
            y_a_twin = gt_labels_t[t_idx]
            y_b_twin = psl_twin
            out_mix_twin = twin(mixed_x_twin)
            loss_mix_twin = lam * criterion(out_mix_twin, y_a_twin) + (1-lam) * criterion(out_mix_twin, y_b_twin)
            # MixUp avec les données source pour twin
            s_idx = torch.randperm(im_data_s.size(0))[:size_twin]
            mixed_x_twin2 = (1-lam) * im_data_s[s_idx] + lam * im_u_twin
            y_a_twin2 = gt_labels_s[s_idx]
            y_b_twin2 = psl_twin
            out_mix_twin2 = twin(mixed_x_twin2)
            loss_mix_twin2 = (1-lam) * criterion(out_mix_twin2, y_a_twin2) + lam * criterion(out_mix_twin2, y_b_twin2)

        total_loss_twin = loss_source_twin + loss_target_twin + loss_mix_twin + loss_mix_twin2
        total_loss_twin.backward()
        optimizer_f_2.step()
        optimizer_g_2.step()

        zero_grad_all()

        log_train = 'S {} T {} Train step: {} lr: {} \t Method: {} \n'.format(
            args.source, args.target, step, optimizer_f.param_groups[0]['lr'], args.method)
        
        
        if step % args.log_interval == 0:
            print(log_train)

        if step % args.save_interval == 0:
            acc_test_net, acc_test_twin, acc_test, total_test, confident_predictions_test = test_ensemble(target_loader_test)
            acc_val_net, acc_val_twin, acc_val, total_val, confident_predictions_val = test_ensemble(target_loader_val)
            net.train()
            twin.train()
            if acc_val >= best_acc:
                best_acc = acc_val
                best_acc_test = acc_test
                counter = 0
            else:
                counter += 1

            print('Test acc ensemble: {:.2f} | Best test: {:.2f} | Best val: {:.2f}'.format(acc_test, best_acc_test, best_acc))
            print('record %s' % record_file)
            with open(record_file, 'a') as f:
                f.write('step %d | net acc: %.2f | twin acc: %.2f | ensemble acc: %.2f | loss_net: %.4f | loss_twin: %.4f | best ensemble: %.2f | best val: %.2f\n' %
                        (step, acc_test_net, acc_test_twin, acc_test, total_loss_net.item(), total_loss_twin.item(), best_acc_test, best_acc))
                
            print('record %s' % record_dir_confident_predictions)
            with open(record_dir_confident_predictions, 'a') as f:
                f.write('%d %d %d \n' %
                        (step, total_test, confident_predictions_test))
                
            if args.save_check:
                print(f'Saving models at step: {step}')
                torch.save(net.state_dict(),
                           os.path.join(args.checkpath,
                                        "Net_two_view_mist_{}_{}_to_{}_step_{}.pth.tar".format(args.method, args.source, args.target, step)))
                torch.save(twin.state_dict(),
                           os.path.join(args.checkpath,
                                        "Twin_two_view_mist_{}_{}_to_{}_step_{}.pth.tar".format(args.method, args.source, args.target, step)))

def test_ensemble(loader):
    net.eval()
    twin.eval()
    
    correct = 0
    correct_net = 0
    correct_twin = 0
    total = 0
    
    confident_predictions = 0
    
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            im_data_t.resize_(data[0].size()).copy_(data[0].to(device))
            gt_labels_t.resize_(data[1].size()).copy_(data[1].to(device))
            
            out_net = net(im_data_t)
            out_twin = twin(im_data_t)
            
            pred_net = out_net.max(1)[1]
            pred_twin = out_twin.max(1)[1]
            
            correct_net += pred_net.eq(gt_labels_t).sum().item()
            correct_twin += pred_twin.eq(gt_labels_t).sum().item()
            
            # Ensemble par somme des probabilités
            out_ensemble = torch.softmax(out_net, dim=1) + torch.softmax(out_twin, dim=1)
            # pred_ensemble = out_ensemble.max(1)[1]
            confidences, pred_ensemble = out_ensemble.max(1)
            confident_predictions += (confidences >= args.th).sum().item()
            
            correct += pred_ensemble.eq(gt_labels_t).sum().item()
            total += gt_labels_t.size(0)
            
    acc_net = 100. * correct_net / total
    acc_twin = 100. * correct_twin / total
    acc_ensemble = 100. * correct / total
    
    return acc_net, acc_twin, acc_ensemble, total, confident_predictions

if __name__ == '__main__':
    if args.eval:
        print('Evaluation mode...')
        acc_net, acc_twin, acc_ensemble = test_ensemble(target_loader_test)
        print('net acc: {:.2f} | twin acc: {:.2f} | ensemble acc: {:.2f}'.format(acc_net, acc_twin, acc_ensemble))
    elif args.test_classifier_3a:
        print("Test classifier 3a n'est pas implémenté pour two-view MIST.")
    else:
        train()
