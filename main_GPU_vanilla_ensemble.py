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

parser = argparse.ArgumentParser(description='Vanilla Ensemble: deux modèles MIST entraînés indépendamment et combinés en fin de parcours')
parser.add_argument('--steps', type=int, default=20000, metavar='N',
                    help='Nombre maximum d’itérations (default: 20000)')
parser.add_argument('--method', type=str, default='mist',
                    choices=['mico', 'mist'])
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='Learning rate (default: 0.01)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='Multiplicateur de lr')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='Sauvegarder le checkpoint ou non')
parser.add_argument('--checkpath', type=str, default='./checkpoint',
                    help='Répertoire de sauvegarde des checkpoints')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='Seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', 
                    help='Nombre de batches entre chaque log')
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='Nombre de batches entre chaque sauvegarde')
parser.add_argument('--net', type=str, default='resnet34',
                    help='Réseau à utiliser')
parser.add_argument('--source', type=str, default='real',
                    help='Domaine source')
parser.add_argument('--target', type=str, default='sketch',
                    help='Domaine cible')
parser.add_argument('--st', type=int, default=0)
parser.add_argument('--dataset', type=str, default='multi')
parser.add_argument('--num', type=int, default=3,
                    help='Nombre d’images étiquetées par classe (3-shot/1-shot)')
parser.add_argument('--th', type=float, default=0.5, help='Seuil de confiance')
parser.add_argument('--root', type=str, default='./')
parser.add_argument('--net_resume', type=str, default='')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--runs', type=int, default=999)
parser.add_argument('--eval', action='store_true', default=False)
parser.add_argument('--test_classifier_3a', action='store_true', default=False)

args = parser.parse_args()

# Scénarios d'adaptation pour DomainNet-subset
multi = [['real', 'clipart'], ['real', 'painting'],
         ['painting', 'clipart'], ['clipart', 'sketch'],
         ['sketch', 'painting'], ['real', 'sketch'],
         ['painting', 'real']]
if args.st != 0 and args.dataset == 'multi':
    args.source, args.target = multi[args.st-1]

print('Dataset {} | Source {} | Target {} | Num par classe {} | Réseau {}'.format(
    args.dataset, args.source, args.target, args.num, args.net))

# Chargement des datasets
source_loader, target_loader, target_loader_unl, target_loader_val, \
    target_loader_test, class_list = return_dataset(args=args, return_idx=False)

# Pour Vanilla Ensemble, nous entraînons deux modèles MIST indépendants
if args.net == 'resnet34':
    net = MetaResnet34(num_class=len(class_list))
    twin = MetaResnet34(num_class=len(class_list))

# Constitution des groupes de paramètres pour chaque modèle
# Pour net
params = []
for value in net.G.parameters():
    if value.requires_grad:
        params.append({'params': [value], 'lr': args.multi, 'weight_decay': 0.0005})
params_F1 = []
for value in net.F1.parameters():
    if value.requires_grad:
        params_F1.append({'params': [value], 'lr': args.multi, 'weight_decay': 0.0005})
# Pour twin
params_2 = []
for value in twin.G.parameters():
    if value.requires_grad:
        params_2.append({'params': [value], 'lr': args.multi, 'weight_decay': 0.0005})
params_F2 = []
for value in twin.F1.parameters():
    if value.requires_grad:
        params_F2.append({'params': [value], 'lr': args.multi, 'weight_decay': 0.0005})

# Répertoire pour sauvegarder les checkpoints et logs
args.checkpath = os.path.join(args.checkpath, 'runs_{}'.format(args.runs))
if not os.path.exists(args.checkpath):
    os.makedirs(args.checkpath)
record_dir = os.path.join('./record', args.dataset, 'vanilla_ensemble')
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir,
                           'exp_net_{}_{}_to_{}_num_{}_{}.txt'.format(
                               args.net, args.source, args.target, args.num, args.runs))

record_dir_confident_predictions = './record/%s/test_confident_predictions_Vanilla_Ensemble' % args.dataset
if not os.path.exists(record_dir_confident_predictions):
    os.makedirs(record_dir_confident_predictions)
record_dir_confident_predictions = os.path.join(record_dir_confident_predictions,
                              'exp_net_%s_%s_to_%s_num_%s_%d.txt' %
                              (args.net, args.source, args.target, args.num, args.runs))

# Détection du GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Utilisation de l'appareil :", device)

# Chargement des checkpoints pré-entraînés (pour initialisation)
pretrain_t_checkpoint = './pretrained_models/pretrained_tgt_{}_to_{}.pth.tar'.format(args.source, args.target)

if args.net_resume:
    net.load_state_dict(torch.load(os.path.join(args.checkpath, args.net_resume), map_location=device))
    twin.load_state_dict(torch.load(os.path.join(args.checkpath, 'Twin' + args.net_resume[3:]), map_location=device))
else:
    print("Chargement du checkpoint pour net depuis :", pretrain_t_checkpoint)
    net.load_state_dict(torch.load(pretrain_t_checkpoint, map_location=device))
    print("Chargement du checkpoint pour twin depuis :", pretrain_t_checkpoint)
    twin.load_state_dict(torch.load(pretrain_t_checkpoint, map_location=device))

# Transfert des modèles sur le GPU
net.to(device)
twin.to(device)

# Préparation des tenseurs d'entrée (initialisés avec une taille fictive)
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

    # Chaque modèle a son propre optimiseur
    optimizer_net = optim.SGD(params + params_F1, momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer_twin = optim.SGD(params_2 + params_F2, momentum=0.9, weight_decay=0.0005, nesterov=True)

    # Sauvegarde des lr initiales
    lr_net = [group["lr"] for group in optimizer_net.param_groups]
    lr_twin = [group["lr"] for group in optimizer_twin.param_groups]

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
    alpha = 1  # Paramètre MixUp

    for step in range(args.start, all_step):
        print('Step : {}'.format(step))

        optimizer_net = inv_lr_scheduler(lr_net, optimizer_net, step, init_lr=base_lr)
        optimizer_twin = inv_lr_scheduler(lr_twin, optimizer_twin, step, init_lr=base_lr)

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

        ##############################
        # Mise à jour du modèle "net" (Vue 1) avec MIST (self‑training)
        ##############################
        optimizer_net.zero_grad()
        out_t_net = net(im_data_t)
        loss_target_net = criterion_no_reduce(out_t_net, gt_labels_t).mean()
        out_s_net = net(im_data_s)
        loss_source_net = criterion_no_reduce(out_s_net, gt_labels_s).mean()

        # Génération de pseudo‑étiquettes par net (self‑prediction)
        u_net_prob = torch.softmax(net(im_data_tu), dim=1)
        u_net_pred = u_net_prob.max(1)  # (valeur max, indice)
        mask_net = u_net_pred[0] >= args.th
        im_u_net = im_data_tu[mask_net]
        psl_net = u_net_pred[1][mask_net]

        lam = np.random.beta(alpha, alpha)
        loss_mix_net = 0
        loss_mix_net2 = 0
        if im_u_net.size(0) > 0:
            size_net = im_u_net.size(0)
            t_idx = torch.randperm(im_data_t.size(0))[:size_net]
            mixed_x_net = lam * im_data_t[t_idx] + (1-lam) * im_u_net
            y_a_net = gt_labels_t[t_idx]
            y_b_net = psl_net
            out_mix_net = net(mixed_x_net)
            loss_mix_net = lam * criterion(out_mix_net, y_a_net) + (1-lam) * criterion(out_mix_net, y_b_net)
            s_idx = torch.randperm(im_data_s.size(0))[:size_net]
            mixed_x_net2 = (1-lam) * im_data_s[s_idx] + lam * im_u_net
            y_a_net2 = gt_labels_s[s_idx]
            y_b_net2 = psl_net
            out_mix_net2 = net(mixed_x_net2)
            loss_mix_net2 = (1-lam) * criterion(out_mix_net2, y_a_net2) + lam * criterion(out_mix_net2, y_b_net2)

        total_loss_net = loss_source_net + loss_target_net + loss_mix_net + loss_mix_net2
        total_loss_net.backward()
        optimizer_net.step()

        ##############################
        # Mise à jour du modèle "twin" (Vue 2) avec MIST (self‑training)
        ##############################
        optimizer_twin.zero_grad()
        out_t_twin = twin(im_data_t)
        loss_target_twin = criterion_no_reduce(out_t_twin, gt_labels_t).mean()
        out_s_twin = twin(im_data_s)
        loss_source_twin = criterion_no_reduce(out_s_twin, gt_labels_s).mean()

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
            t_idx = torch.randperm(im_data_t.size(0))[:size_twin]
            mixed_x_twin = lam * im_data_t[t_idx] + (1-lam) * im_u_twin
            y_a_twin = gt_labels_t[t_idx]
            y_b_twin = psl_twin
            out_mix_twin = twin(mixed_x_twin)
            loss_mix_twin = lam * criterion(out_mix_twin, y_a_twin) + (1-lam) * criterion(out_mix_twin, y_b_twin)
            s_idx = torch.randperm(im_data_s.size(0))[:size_twin]
            mixed_x_twin2 = (1-lam) * im_data_s[s_idx] + lam * im_u_twin
            y_a_twin2 = gt_labels_s[s_idx]
            y_b_twin2 = psl_twin
            out_mix_twin2 = twin(mixed_x_twin2)
            loss_mix_twin2 = (1-lam) * criterion(out_mix_twin2, y_a_twin2) + lam * criterion(out_mix_twin2, y_b_twin2)

        total_loss_twin = loss_source_twin + loss_target_twin + loss_mix_twin + loss_mix_twin2
        total_loss_twin.backward()
        optimizer_twin.step()

        log_train = 'Source: {} | Target: {} | Step: {} | lr: {} | Method: Vanilla Ensemble\n'.format(
            args.source, args.target, step, optimizer_net.param_groups[0]['lr'])
        if step % args.log_interval == 0:
            print(log_train)

        if step % args.save_interval == 0:
            acc_net, acc_twin, acc_ensemble = test_ensemble(target_loader_test)
            acc_val_net, acc_val_twin, acc_val = test_ensemble(target_loader_val)
            net.train()
            twin.train()
            if acc_val >= best_acc:
                best_acc = acc_val
                best_acc_test = acc_ensemble
                counter = 0
            else:
                counter += 1

            print('Ensemble acc: {:.2f} | Best ensemble: {:.2f} | Best val: {:.2f}'.format(acc_ensemble, best_acc_test, best_acc))
            print('Enregistrement dans {}'.format(record_file))
            with open(record_file, 'a') as f:
                f.write('Step {} | net acc: {:.2f} | twin acc: {:.2f} | ensemble acc: {:.2f} | loss_net: {:.4f} | loss_twin: {:.4f} | best ensemble: {:.2f} | best val: {:.2f}\n'
                        .format(step, acc_net, acc_twin, acc_ensemble, total_loss_net.item(), total_loss_twin.item(), best_acc_test, best_acc))
            
            print('record %s' % record_dir_confident_predictions)
            with open(record_dir_confident_predictions, 'a') as f:
                f.write('%d %d %d \n' %
                        (step, total_test, confident_predictions_test))
            
            if args.save_check:
                print('Sauvegarde des modèles à l’étape {}'.format(step))
                torch.save(net.state_dict(),
                           os.path.join(args.checkpath,
                                        "Net_vanilla_ensemble_{}_{}_to_{}_step_{}.pth.tar".format(args.method, args.source, args.target, step)))
                torch.save(twin.state_dict(),
                           os.path.join(args.checkpath,
                                        "Twin_vanilla_ensemble_{}_{}_to_{}_step_{}.pth.tar".format(args.method, args.source, args.target, step)))

def test_ensemble(loader):
    net.eval()
    twin.eval()
    correct_net = 0
    correct_twin = 0
    correct_ensemble = 0
    total = 0
    
    confident_predictions = 0
    
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            im_data_t.resize_(data[0].size()).copy_(data[0].to(device))
            gt_labels_t.resize_(data[1].size()).copy_(data[1].to(device))
            
            # Prédictions individuelles
            out_net = net(im_data_t)
            out_twin = twin(im_data_t)
            pred_net = out_net.max(1)[1]
            pred_twin = out_twin.max(1)[1]
            
            # Ensemble : somme des probabilités softmax
            ensemble_out = torch.softmax(out_net, dim=1) + torch.softmax(out_twin, dim=1)
            # pred_ensemble = ensemble_out.max(1)[1]
            confidences, pred_ensemble = ensemble_out.max(1)
            confident_predictions += (confidences >= args.th).sum().item()
            
            total += gt_labels_t.size(0)
            correct_net += pred_net.eq(gt_labels_t).sum().item()
            correct_twin += pred_twin.eq(gt_labels_t).sum().item()
            correct_ensemble += pred_ensemble.eq(gt_labels_t).sum().item()
            
    acc_net = 100.0 * correct_net / total
    acc_twin = 100.0 * correct_twin / total
    acc_ensemble = 100.0 * correct_ensemble / total
    
    return acc_net, acc_twin, total, confident_predictions

if __name__ == '__main__':
    if args.eval:
        print('Mode évaluation...')
        acc_net, acc_twin, acc_ensemble = test_ensemble(target_loader_test)
        print('Précision Net: {:.2f}% | Twin: {:.2f}% | Ensemble: {:.2f}%'.format(acc_net, acc_twin, acc_ensemble))
    else:
        train()
