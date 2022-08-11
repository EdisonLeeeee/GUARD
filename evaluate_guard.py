import os
import torch
from tqdm import tqdm
from graphattack.data import GraphAttackDataset
from graphattack.training import Trainer, RobustGCNTrainer
from graphattack.training.callbacks import ModelCheckpoint
from graphattack.models import GCN, SGC
from graphattack.defense.model_level import MedianGCN, RobustGCN, ReliableGNN, ElasticGNN
from graphattack import set_seed
from graphattack.utils import split_nodes, flip_graph, setup_logger
from graphattack.functional import feat_normalize
from ogb.nodeproppred import DglNodePropPredDataset

from guard import GUARD, RandomGUARD, DegreeGUARD

log = setup_logger(name='GUARD EXP', output='GUARD.txt')


def evaluate_guard(dataset='cora', defense="GUARD", attack='SGAttack', model="SGC", seed=2022, k=500, alpha=2, clean=False):
    log.info(f"{'='*10}dataset={dataset}, attack={attack}, defense={(defense, k)}, model={model} clean={clean}, alpha={alpha}{'='*10}")
    # ============ Load datasets ================================
    data = GraphAttackDataset(dataset, verbose=True, standardize=True if dataset != 'reddit' else False)
    g = data[0]
    y = g.ndata['label']
    splits = split_nodes(y, random_state=15)

    if dataset == 'reddit':
        g.ndata['feat'] = feat_normalize(g.ndata['feat'], norm='standardize', dim=0)

    num_feats = g.ndata['feat'].size(1)
    num_classes = data.num_classes
    y_train = y[splits.train_nodes]
    y_val = y[splits.val_nodes]
    y_test = y[splits.test_nodes]

    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    g = g.to(device)

    if defense == "GUARD":
        surrogate = GCN(num_feats, num_classes, bias=False, acts=None)
        surrogate_trainer = Trainer(surrogate, device=device)
        cb = ModelCheckpoint('surrogate_guard.pth', monitor='val_accuracy')
        surrogate_trainer.fit(g, y_train, splits.train_nodes, val_y=y_val, val_index=splits.val_nodes, callbacks=[cb], verbose=0)
        guard = GUARD(g.ndata['feat'], g.in_degrees(), alpha=alpha, device=device)
        guard.setup_surrogate(surrogate, y_train)
    elif defense == "RandomGUARD":
        guard = RandomGUARD(g.num_nodes(), device=device)
    elif defense == "DegreeGUARD":
        guard = DegreeGUARD(g.in_degrees(), device=device)
    else:
        raise ValueError(f"Unknown defense {defense}")

    set_seed(seed)
    # ============ Setup victim model ==========================
    if model == "SGC":
        victim = SGC(num_feats, num_classes)
        trainer = Trainer(victim, device=device, lr=0.1, weight_decay=1e-5)
    elif model == "GCN":
        victim = GCN(num_feats, num_classes)
        trainer = Trainer(victim, device=device)
    elif model == 'Median':
        victim = MedianGCN(num_feats, num_classes)
        trainer = Trainer(victim, device=device)
    elif model == 'RGCN':
        victim = RobustGCN(num_feats, num_classes)
        trainer = RobustGCNTrainer(victim, device=device)
    elif model == 'Elastic':
        victim = ElasticGNN(num_feats, num_classes)
        trainer = Trainer(victim, device=device)
    else:
        raise ValueError(f"Unknown model {model}")

    cb = ModelCheckpoint('victim_model.pth', monitor='val_accuracy')
    trainer.fit(g, y_train, splits.train_nodes, val_y=y_val, val_index=splits.val_nodes, callbacks=[cb], verbose=0)
    acc = trainer.evaluate(g, y_test, splits.test_nodes, verbose=0).accuracy
    log.info(f"Accuracy on test set: {acc:.2%}")

    # ============ Load attackers ====================================
    dir_name = f"perturbations/{dataset}"
    file_name = f"{dir_name}/{attack}.pkl"
    perturbation = torch.load(file_name)

    # ============ load target nodes to attack =====================
    targets = torch.as_tensor(list(perturbation.keys()))
    
    acc = trainer.evaluate(g, y[targets], targets, verbose=0).accuracy
    log.info(f"Accuracy on target nodes: {acc:.2%}")
    
    pbar = tqdm(targets)
    counts1 = counts2 = total = 0

    # ============ test performance under attack =======
    for target in pbar:
        target_label = y[target]

        if clean:
            attack_g = g
        else:
            edges = perturbation[target.item()].to(g.device)  # [2, M]
            attack_g = flip_graph(g, edges)

        # Important for SGC
        trainer.cache_clear()
        predict = trainer.predict(attack_g, target)

        if predict.argmax(-1) == target_label:
            counts1 += 1

        defense_g = guard(attack_g, target, k=k)
        # Important for SGC
        trainer.cache_clear()
        predict = trainer.predict(defense_g, target)

        if predict.argmax(-1) == target_label:
            counts2 += 1

        total += 1
        pbar.set_description(f"Before defense: {counts1/len(pbar):.2%}, "
                             f"after defense: {counts2/total:.2%}")
    log.info(f"Before defense: {counts1/len(pbar):.2%}, "
             f"after defense: {counts2/total:.2%}")
    log.info('=' * 100)
    log.info('')
    return counts1 / len(pbar), counts2 / total


def evaluate_guard_ogb(dataset='ogbn-arxiv', defense="GUARD", attack='SGAttack', model="SGC", seed=2022, k=100, alpha=2, clean=False):
    log.info(f"{'='*10}dataset={dataset}, attack={attack}, defense={(defense, k)}, model={model} clean={clean}, alpha={alpha}{'='*10}")
    # ============ Load datasets ================================
    data = DglNodePropPredDataset(name=dataset)
    splits = data.get_idx_split()
    g, y = data[0]
    y = y.flatten()

    srcs, dsts = g.edges()
    g.add_edges(dsts, srcs)
    g = g.remove_self_loop()

    num_feats = g.ndata["feat"].size(1)
    num_classes = (y.max() + 1).item()
    y_train = y[splits['train']]
    y_val = y[splits['valid']]
    y_test = y[splits['test']]

    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    g = g.to(device)

    if defense == "GUARD":
        surrogate = SGC(num_feats, num_classes, bias=False)
        surrogate_trainer = Trainer(surrogate, device=device, lr=0.1, weight_decay=1e-5)
        cb = ModelCheckpoint('surrogate_guard.pth', monitor='val_accuracy')
        surrogate_trainer.fit(g, y_train, splits['train'], val_y=y_val, val_index=splits['valid'], callbacks=[cb], verbose=0, epochs=200)
        guard = GUARD(g.ndata['feat'], g.in_degrees(), device=device)
        guard.setup_surrogate(surrogate, y_train)
    elif defense == "RandomGUARD":
        guard = RandomGUARD(g.num_nodes(), device=device)
    elif defense == "DegreeGUARD":
        guard = DegreeGUARD(g.in_degrees(), device=device)
    else:
        raise ValueError(f"Unknown defense {defense}")

    set_seed(seed)
    # ============ Setup victim model ==========================
    if model == "SGC":
        victim = SGC(num_feats, num_classes, bias=False)
        trainer = Trainer(victim, device=device, lr=0.1, weight_decay=1e-5)
    elif model == "GCN":
        victim = GCN(num_feats, num_classes, hids=[256, 256], bn=True)
        trainer = Trainer(victim, device=device)
    elif model == 'Median':
        victim = ReliableGNN(num_feats, num_classes, hids=[256, 256], bn=False)
        trainer = Trainer(victim, device=device)
    elif model == 'RGCN':
        victim = RobustGCN(num_feats, num_classes, hids=[256, 256])
        trainer = RobustGCNTrainer(victim, device=device)
    else:
        raise ValueError(f"Unknown model {model}")

    cb = ModelCheckpoint('victim_model.pth', monitor='val_accuracy')
    trainer.fit(g, y_train, splits['train'], val_y=y_val, val_index=splits['valid'], callbacks=[cb], verbose=1, epochs=200)
    acc = trainer.evaluate(g, y_test, splits['test'], verbose=0).accuracy
    log.info(f"Accuracy on test set: {acc:.2%}")

    # ============ Load attackers ====================================
    dir_name = f"perturbations/{dataset}"
    file_name = f"{dir_name}/{attack}.pkl"
    perturbation = torch.load(file_name)

    # ============ load target nodes to attack =====================
    targets = torch.as_tensor(list(perturbation.keys()))
    
    acc = trainer.evaluate(g, y[targets], targets, verbose=0).accuracy
    log.info(f"Accuracy on target nodes: {acc:.2%}")
    
    pbar = tqdm(targets)
    counts1 = counts2 = total = 0

    # ============ test performance under attack =======
    for target in pbar:
        target_label = y[target]

        if clean:
            attack_g = g
        else:
            edges = perturbation[target.item()].to(g.device)
            attack_g = flip_graph(g, edges)

        # Important for SGC
        trainer.cache_clear()
        predict = trainer.predict(attack_g, target)

        if predict.argmax(-1) == target_label:
            counts1 += 1

        defense_g = guard(attack_g, target, k=k)
        # Important for SGC
        trainer.cache_clear()
        predict = trainer.predict(defense_g, target)

        if predict.argmax(-1) == target_label:
            counts2 += 1

        total += 1
        pbar.set_description(f"Before defense: {counts1/len(pbar):.2%}, "
                             f"after defense: {counts2/total:.2%}")

    log.info(f"Before defense: {counts1/len(pbar):.2%}, "
             f"after defense: {counts2/total:.2%}")
    log.info('=' * 100)
    log.info('')
    return counts1 / len(pbar), counts2 / total


# # ================================================================== #
# #                      Run on small datasets                         #
# # ================================================================== #
alpha = 2.0
for clean in [False, True]:
    for model in ["GCN", "SGC", "Median", "RGCN", "Elastic"]:
        for dataset in ['cora', 'pubmed']:
            if dataset == 'cora':
                k = 200
            else:
                k = 500
            for attack in ['SGAttack', 'FGAttack', 'IGAttack']:
                for defense in ['GUARD', 'DegreeGUARD', 'RandomGUARD']:
                    acc_before, acc_after = evaluate_guard(dataset, attack=attack, defense=defense,
                                                           model=model, clean=clean, k=k, alpha=alpha)

# # ================================================================== #
# #                      Run on large datasets                         #
# #       FGAttack and IGAttack are not available for large datasets   #
# # ================================================================== #
k = 10000
alpha = 2.0
for clean in [False]:
    for model in ["GCN", "SGC"]:
        for defense in ['GUARD', 'DegreeGUARD', 'RandomGUARD']:
            evaluate_guard_ogb('ogbn-arxiv', attack='SGAttack', defense=defense, model=model, k=k, clean=clean, alpha=alpha)

            
k = 20000
alpha = 2.0
for clean in [False]:
    for model in ["SGC", "GCN"]:
        for defense in ['GUARD', 'DegreeGUARD', 'RandomGUARD']:
            evaluate_guard('reddit', attack='SGAttack', defense=defense, model=model, k=k, clean=clean, alpha=alpha)