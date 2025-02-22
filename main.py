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

parser = argparse.ArgumentParser(description='DeCoTa, previously MiCo, for Semi-supervised Domain Adaptation')
parser.add_argument('--steps', type=int, default=20000, metavar='N',
                    help='maximum number of iterations '
                        'to train (default: 20000)')
parser.add_argument('--method', type=str, default='mico',
                    choices=['mico', 'mist'])
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./checkpoint',
                    help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', 
                    help='how many batches to wait before logging '
                        'training status')
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before saving a model')  # cf. Fig. 3.a)
parser.add_argument('--net', type=str, default='resnet34',
                    help='which network to use')
parser.add_argument('--source', type=str, default='real',
                    help='source domain')
parser.add_argument('--target', type=str, default='sketch',
                    help='target domain')
parser.add_argument('--st',type=int, default=0)
parser.add_argument('--dataset', type=str, default='multi')
parser.add_argument('--num', type=int, default=3,
                    help='3-shot/1-shot')
parser.add_argument('--th', type=float, default=0.5)
parser.add_argument('--root', type=str, default='./')
parser.add_argument('--net_resume', type=str, default='') # que fait ça ?
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


""" net: w_f; twin: w_g """             # net : w_f : UDA/SSL ?
use_gpu = torch.cuda.is_available()
if use_gpu:
    torch.cuda.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)

if args.net == 'resnet34':
    net = MetaResnet34(num_class=len(class_list))
    twin = MetaResnet34(num_class=len(class_list))

params = []
for value in net.G.params():    # Paramètres G : paramètre de la convolution de net (wf)
    if value.requires_grad:
        params += [{'params': [value], 'lr': args.multi,
                    'weight_decay': 0.0005}] # terme de régularisation L2 : évite l'overfitting
params_F1 = []                  
for value in net.F1.params():   # paramètres F1 : paramètre du RCL de net (wf)
    if value.requires_grad:
        params_F1 += [{'params': [value], 'lr': args.multi,
                    'weight_decay': 0.0005}]

params_2 = []
for value in twin.G.params():   # paramètre G : paramètre de la convolution de twin (wg)
    if value.requires_grad:
        params_2 += [{'params': [value], 'lr': args.multi,
                    'weight_decay': 0.0005}]

params_F2 = []
for value in twin.F1.params():  # paramètre F2 : paramètre du FCL de twin (wg)
    if value.requires_grad: 
        params_F2 += [{'params': [value], 'lr': args.multi,
                    'weight_decay': 0.0005}]

""" record & resume path """
args.checkpath = os.path.join(args.checkpath, 'runs_{}'.format(args.runs))
if not os.path.exists(args.checkpath):
    os.makedirs(args.checkpath)

record_dir = './record/%s/mico' % args.dataset
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir,
                        'exp_net_%s_%s_to_%s_num_%s_%d' %
                        (args.net, args.source,
                            args.target, args.num, args.runs))


""" pre-train & resume """
pretrain_src_checkpoint = './pretrained_models/pretrained_src_{}_to_{}.pth.tar'.format(args.source, args.target) # modèle fine tunné sur la source : twin = wg (UDA) (à vérifier, premier check : parait ok)
pretrain_t_checkpoint = './pretrained_models/pretrained_tgt_{}_to_{}.pth.tar'.format(args.source, args.target)   # modèle fine tunné sur la source puis sur la target : Net = wf (SSL) (à vérifier, premier check : parait ok)

if args.net_resume:

    p1 = os.path.join(args.checkpath, args.net_resume)
    p2 = os.path.join(args.checkpath, 'Twin' + args.net_resume[3:]) # wg (UDA) (à vérifier)

    net.load_state_dict(
        torch.load(p1)
    )
    twin.load_state_dict(
        torch.load(p2)
    )
else:
    print(f"Loading checkpoint from: {pretrain_src_checkpoint}")
    net.load_state_dict(            # net : wf : SSL
        torch.load(pretrain_src_checkpoint, map_location=torch.device('cpu'))
    )
    twin.load_state_dict(           # twin : wg : UDA
        torch.load(pretrain_t_checkpoint, map_location=torch.device('cpu'))
    )


lr = args.lr
net.to(torch.device('cpu'))
twin.to(torch.device('cpu'))

im_data_s = torch.FloatTensor(1)
im_data_t = torch.FloatTensor(1)
im_data_tu = torch.FloatTensor(1)
im_data_tu_2 = torch.FloatTensor(1)
gt_labels_s = torch.LongTensor(1)
gt_labels_t = torch.LongTensor(1)

im_data_s = im_data_s.to(torch.device('cpu'))
im_data_t = im_data_t.to(torch.device('cpu'))
im_data_tu = im_data_tu.to(torch.device('cpu'))
im_data_tu_2 = im_data_tu_2.to(torch.device('cpu'))
gt_labels_s = gt_labels_s.to(torch.device('cpu'))
gt_labels_t = gt_labels_t.to(torch.device('cpu'))

im_data_s = Variable(im_data_s)
im_data_t = Variable(im_data_t)
im_data_tu = Variable(im_data_tu)
im_data_tu_2 = Variable(im_data_tu_2)
sgt_labels_s = Variable(gt_labels_s)
gt_labels_t = Variable(gt_labels_t)


def train():
    net.train()
    twin.train()

    optimizer_g = optim.SGD(params, momentum=0.9,
                            weight_decay=0.0005, nesterov=True) # moment de nesterov : permet d'ajuster la descente de gradient plus finement : compliqué à clarifier
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

    param_lr_g = []
    param_lr_g_2 = []
    param_lr_f = []
    param_lr_f_2 = []

    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    for param_group in optimizer_g_2.param_groups:
        param_lr_g_2.append(param_group["lr"])
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])
    for param_group in optimizer_f_2.param_groups:
        param_lr_f_2.append(param_group["lr"])

    criterion = nn.CrossEntropyLoss().to(torch.device('cpu')) # perte sur l'ensemble du batch moyenné
    criterion_no_reduce = nn.CrossEntropyLoss(reduction='none').to(torch.device('cpu')) # les pertes pour chaque échantillon sont conservées et on calcule la perte pour chaque échantillon
    all_step = args.steps
    data_iter_s = iter(source_loader)           # crée un objet pour itérer sur le loader
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)
    len_train_source = len(source_loader)       # donne le nombre de batchs dans le loader
    len_train_target = len(target_loader)       
    len_train_target_semi = len(target_loader_unl)
    best_acc = 0
    counter = 0

    for step in range(args.start, all_step): # itération pas sur les epochs mais sur les batchs

        print(f'step : {step}')
        
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step,
                                    init_lr=args.lr)
        optimizer_g_2 = inv_lr_scheduler(param_lr_g_2, optimizer_g_2, step,
                                    init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step,
                                    init_lr=args.lr)
        optimizer_f_2 = inv_lr_scheduler(param_lr_f_2, optimizer_f_2, step,
                                    init_lr=args.lr)
        lr = optimizer_f.param_groups[0]['lr']

        if step % len_train_target == 0:        # on s'assure que il y a encore des batchs non utilisés dans le loader
            data_iter_t = iter(target_loader)
        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_loader_unl)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        data_t = next(data_iter_t)              # on prends le prochain batch
        data_t_unl = next(data_iter_t_unl)
        data_s = next(data_iter_s)

        with torch.no_grad():

            im_data_s.resize_(data_s[0].size()).copy_(data_s[0])          # images dans la source
            gt_labels_s.resize_(data_s[1].size()).copy_(data_s[1])        # étiquettes des images dans source
            im_data_t.resize_(data_t[0].size()).copy_(data_t[0])          # images dans target qui ont une étiquettes
            gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])        # étiquettes des images dans target
            im_data_tu.resize_(data_t_unl[0].size()).copy_(data_t_unl[0]) # images non étiquetées de target

        """stream 1"""
        zero_grad_all()
        data = im_data_t        # images contenus dans target
        target = gt_labels_t    # étiquettes des images contenues dans target
        out_1 = net(data)       # sortie de net sur les images du domaine cible
        loss_1 = criterion_no_reduce(out_1, target).mean() # perte moyenne sur les prédictions par net (UDA ?) sur les images contenues dans target
        zero_grad_all()

        """stream 2"""
        data = im_data_s      # images contenues dans la source
        target = gt_labels_s  # labels des images contenues dans la source
        out_2 = twin(data)    # prédictions des étiquettes des images contenues dans la source par le réseau twin
        loss_2 = criterion_no_reduce(out_2, target).mean() # perte moyenne sur la prédiction
        zero_grad_all()

        """pseudo-label"""
        u_1_prob = torch.softmax(net(im_data_tu), dim=1) # prédictions des étiquettes par le réseau Net sur les données non étiquetées du domaine cible
        u_1_pred = u_1_prob.max(1)                       # on prend les prédictions les plus élévées pour chaque image du batch
        u_2_prob = torch.softmax(twin(im_data_tu), dim=1)# prédictions des étiquettes par le réseau Twin sur les données non étiquetées du domaine cible
        u_2_pred = u_2_prob.max(1)                       # on prend les prédictions les plus élévées pour chaque image du batch

        u_1_mask = u_1_pred[0] >= args.th       # on prend juste les indices des images qui ont eu une prédiction par net > seuil
        u_2_mask = u_2_pred[0] >= args.th       # on prend juste les indices des images qui ont eu une prédiction par twin > seuil

        im_u_1 = im_data_tu[u_2_mask]   # récupération des images non étiquetées dont les labels ont été prédis par !!!TWIN (u2)!!!  avec une probalité supérieure au seuil (th, par défaut : 0.5)
        psl_u_1 = u_2_pred[1][u_2_mask] # récupération de l'étiquette des images non étiquetées dont les labels ont été prédis par !!!TWIN (u2)!!! avec confiance > seuil
        im_u_2 = im_data_tu[u_1_mask]   # idem pour le réseau !!! NET (u1) !!!
        psl_u_2 = u_1_pred[1][u_1_mask]

        """mix_up"""
        alpha = 1
        lam = np.random.beta(alpha, alpha)  # utilisation de la loi beta(1,1) (loi uniforme continue sur [0,1]) pour déterminer la loi MixUp
        # stream 1               # UDA/SSL ?
        if im_u_1.size(0) > 0:   # si il y au moins une image qui a eu une prédiction confiante sur twin

            size_1 = im_u_1.size(0) # nombre d'images qui ont une prédiction confiante par twin (wg)
            # print('stream 1: {}'.format(size_1))

            t_idx = torch.randperm(im_data_t.size(0))[0:size_1] # génère une liste d'indice aléatoire de la taille du nombre d'images dans les targets. Puis en séléctionne le nombre d'images qui ont eu une confiance importante par twin
            mixed_x = lam * im_data_t[t_idx] + (1-lam) * im_u_1 # MixUp entre t_idx (c'est un chiffre ou un nombre) images provenant de target et t_idx image provenant des images non étiquetées auxquelles nous venons d'ajouter une étiquette
            y_a, y_b = gt_labels_t[t_idx], psl_u_1  # y_a => vrais labels d'images étiquetées dans target, y_b => pseudos labels

            out_mix = net(mixed_x)  # prédictions des étiquettes avec comme jeu de données les images ayant servies au MixUp par !!NET(wf)!!!
            loss_mix_1 = lam * criterion(out_mix, y_a) + (1-lam) * criterion(out_mix, y_b) # pertes associées

            loss_1 += loss_mix_1 # ajout de la perte sur MixUp à la perte globale sur target par net : voir aide
            zero_grad_all()
            loss_1.backward(retain_graph=True)
            optimizer_f.step()
            optimizer_g.step()
        else:
            zero_grad_all()
            loss_1.backward(retain_graph=True)
            optimizer_f.step() # optimisation des classifieurs net et twin
            optimizer_g.step()
        # aide MixUp : pourquoi MixUp ? Les modèles wf et wg génèrent des pseudos labels qui peuvent être incorrects
        # si le modèles apprends sur ces données, cela peut causer dégrader les performances. Plutot que d'entrainer sur ces échantillons, MixUp crée des échantillons entre les vrais échantillons et et les pseudos échantillons : on 
        # évite de surajuster sur les faux labels et on "adoucit" le co-entrainemnt.

        zero_grad_all()
        # stream 2              # Idem que vu avec le stream 1 mais via net : wf
        if im_u_2.size(0) > 0:
            
            size_2 = im_u_2.size(0)
            # print('stream 2: {}'.format(size_2))
            
            s_idx = torch.randperm(im_data_s.size(0))[0:size_2]
            mixed_x = (1-lam) * im_data_s[s_idx] + lam * im_u_2
            y_a, y_b = gt_labels_s[s_idx], psl_u_2
            
            out_mix = twin(mixed_x)
            loss_mix_2 = (1-lam) * criterion(out_mix, y_a) + lam * criterion(out_mix, y_b)
            
            loss_2 += loss_mix_2
            zero_grad_all()
            loss_2.backward()
            optimizer_f_2.step()
            optimizer_g_2.step()
        else:
            zero_grad_all()
            loss_2.backward()
            optimizer_f_2.step()
            optimizer_g_2.step()
        zero_grad_all()

        log_train = 'S {} T {} Train Ep: {} lr{} \t ' \
                    'Method {}\n'.\
            format(args.source, args.target,
                step, lr,
                args.method)
        net.zero_grad()
        twin.zero_grad()
        zero_grad_all()
        
        if step % args.log_interval == 0:           # par défaut : 100
            print(log_train)

        if step % args.save_interval == 0:          # par défaut : 500

            acc_test_net, acc_test_twin, acc_test = test_ensemble(target_loader_test)
            acc_val_net, acc_val_twin, acc_val = test_ensemble(target_loader_val)

            net.train()
            twin.train()
            if acc_val >= best_acc: # best_acc = 0 à l'initialisation
                best_acc = acc_val
                best_acc_test = acc_test
                counter = 0
            else:
                counter += 1

            print('test acc %f best acc test %f best acc val %f' % (acc_test,
                                                                    best_acc_test,
                                                                    best_acc))
            print('record %s' % record_file)
            with open(record_file, 'a') as f:
                f.write('step %d wf %f wg %f mico %f best mico %f best val %f \n' % (step,
                                                                                    acc_test_net,
                                                                                    acc_test_twin,
                                                                                    acc_test,
                                                                                    best_acc_test,
                                                                                    best_acc))
            net.train()
            twin.train()
            if args.save_check:
                # print('saving model')
                print(f'saving model step : {step}')
                torch.save(net.state_dict(),
                        os.path.join(args.checkpath,
                                        "Net_iter_model_{}_{}_"
                                        "to_{}_step_{}.pth.tar".
                                        format(args.method, args.source,
                                            args.target, step)))
                torch.save(twin.state_dict(),
                        os.path.join(args.checkpath,
                                        "Twin_iter_model_{}_{}_"
                                        "to_{}_step_{}.pth.tar".
                                        format(args.method, args.source,
                                            args.target, step)))


def test_ensemble(loader): # test les modèles wf et wg sur l'ensemble de test
    net.eval()
    twin.eval()
    correct = 0
    correct_test_1 = 0
    correct_test_2 = 0
    total = 0

    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
            output1 = net(im_data_t) # prédictions des labels par net des images dans target 
            output2 = twin(im_data_t) # prédictions des labels par twin des images dans target

            """test 1 and 2"""
            pred_test_1 = output1.max(1)[1] # récupération des prédictions les plus confiantes par net
            pred_test_2 = output2.max(1)[1] # idem pour twin

            correct_test_1 += pred_test_1.eq(gt_labels_t).sum().item() # nombre de bonnes prédictions par net
            correct_test_2 += pred_test_2.eq(gt_labels_t).sum().item() # nombre de bonnes prédictions par twin

            """ensemble results"""
            output = torch.softmax(output1, dim=1) + torch.softmax(output2, dim=1) # sommes des distributions de proba (pas supérieur à 1 ???)
            pred = output.max(1)[1] # récupération du label le plus probable d'après la prédictions

            total += gt_labels_t.size(0) # ajut à total des labels utilisés pour les prédictions
            correct += pred.eq(gt_labels_t).sum().item() # ajout des labels bien prédits parla méthode ensembliste

    acc_test_1 = 100. * (float(correct_test_1)/total)  # accuracy du test sur net (UDA)
    acc_test_2 = 100. * (float(correct_test_2)/total)  # idem pour twin (SSL)
    acc = 100. * (float(correct)/total)                # accuracy des deux réseaux assemblées

    return acc_test_1, acc_test_2, acc

if __name__ == '__main__':
    if args.eval: # evaluation du modèle par le vanilla-ensemble
        print('eval mode...')
        acc_test_net, acc_test_twin, acc_test = test_ensemble(target_loader_test)
        print('net acc: {}, twin acc: {}, mico acc: {}'.format(acc_test_net, acc_test_twin, acc_test))
    else: # sinon entraine le modèle
        train() 
