import sys
import argparse
import torch
import torch.optim as optim

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders
from h02_learn.model import BiaffineParser, MSTParser
from h02_learn.model import NeuralTransitionParser, ChartParser
from h02_learn.model import MH4, Hybrid, ArcEager, ArcStandard
from h02_learn.train_info import TrainInfo
from h02_learn.algorithm.mst import get_mst_batch
from utils import constants
from utils import utils
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import wandb


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--language', type=str, required=True)
    parser.add_argument('--data-path', type=str, default='data/')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--batch-size-eval', type=int, default=128)
    parser.add_argument('--key', type=str)
    # Model

    parser.add_argument('--dropout', type=float, default=.33)
    parser.add_argument('--weight-decay', type=float, default=0.01)

    parser.add_argument('--easy-first', choices=["False", "True"], default="True")
    # Optimization
    parser.add_argument('--optim', choices=['adam', 'adamw', 'sgd'], default='adamw')
    parser.add_argument('--eval-batches', type=int, default=20)
    parser.add_argument('--wait-epochs', type=int, default=10)
    parser.add_argument('--lr-decay', type=float, default=.5)
    parser.add_argument('--lr', type=float, default=1e-5)
    # Save
    parser.add_argument('--name', type=str, default='generic-experiment')
    parser.add_argument('--checkpoints-path', type=str, default='checkpoints/')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--save-periodically', action='store_true')

    args = parser.parse_args()
    args.wait_iterations = 3  # args.wait_epochs * args.eval_batches
    s = None

    if args.easy_first == "True":
        s = "EasyFirst"
    else:
        s = "ShiftReduce"
    args.save_path = '%s/%s/%s/%s/' % (args.checkpoints_path, args.language, s, args.name)
    utils.config(args.seed)
    return args


def get_optimizer(paramters, lr, lr_decay, weight_decay):

    optimizer = optim.AdamW(paramters, betas=(.9, .9), weight_decay=weight_decay, lr=lr)

    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max")
    return optimizer, lr_scheduler


def get_model(vocabs, args):
    return ChartParser(language=args.language, vocabs=vocabs,
                       batch_size=args.batch_size,
                       hypergraph=MH4, dropout=args.dropout).to(
        device=constants.device)


def calculate_attachment_score(heads_tgt, heads, predicted_rels, rels):
    predicted_rels = predicted_rels.permute(1, 0)
    acc_h = (heads_tgt == heads)[heads != -1]

    rels = rels.permute(1, 0)
    acc_l = (predicted_rels == rels)[rels != 0]

    uas = acc_h.float().mean().item()
    las = (acc_h & acc_l).float().mean().item()
    return las, uas


def simple_attachment_scores(predicted_heads, heads, lengths):
    correct = torch.eq(predicted_heads[:, lengths], heads[:, lengths]).sum().item()
    total = torch.sum(lengths).item()

    return correct / total


def _evaluate(evalloader, model):
    # pylint: disable=too-many-locals
    dev_loss, dev_las, dev_uas, n_instances = 0, 0, 0, 0
    steps = 0
    for (text, maps), (heads, rels), (transitions, relations_in_order) in evalloader:
        steps += 1
        maps = maps.to(device=constants.device)
        text = text.to(device=constants.device)
        heads, rels = heads.to(device=constants.device), rels.to(device=constants.device)
        transitions = transitions.to(device=constants.device)
        relations_in_order = relations_in_order.to(device=constants.device)
        loss, predicted_heads, predicted_rels = model(text, transitions, relations_in_order, maps, heads=heads,
                                                      rels=rels)

        las, uas = calculate_attachment_score(predicted_heads, heads, predicted_rels, rels)
        batch_size = text.shape[0]
        dev_loss += (loss * batch_size)
        dev_las += (las * batch_size)
        dev_uas += (uas * batch_size)
        n_instances += batch_size

    return dev_loss / n_instances, dev_las / n_instances, dev_uas / n_instances


def evaluate(evalloader, model):
    model.eval()
    #model.bert.eval()
    with torch.no_grad():
        result = _evaluate(evalloader, model)
    model.train()
    #model.bert.train()
    return result


def train_batch(text, heads, rels, transitions, relations_in_order, maps, model, optimizer):
    optimizer.zero_grad()
    maps = maps.to(device=constants.device)
    text = text.to(device=constants.device)
    heads, rels = heads.to(device=constants.device), rels.to(device=constants.device)

    transitions = transitions.to(device=constants.device)
    relations_in_order = relations_in_order.to(device=constants.device)

    loss, pred_h, pred_rel = model(text, transitions, relations_in_order, maps, heads=heads, rels=rels)

    # las, uas = calculate_attachment_score(pred_h, heads, pred_rel, rels)
    loss.backward()
    optimizer.step()

    return loss.item()


def train(trainloader, devloader, model, eval_batches, lr, wait_iterations, weight_decay,
          save_path, save_batch=False, file=None):
    # pylint: disable=too-many-locals,too-many-arguments
    # torch.autograd.set_detect_anomaly(True)

    # optimizer, lr_scheduler = get_optimizer(model.parameters(), optim_alg, lr_decay, weight_decay)
    optimizer, lr_scheduler = get_optimizer(model.parameters(),lr,lr_decay=.5,weight_decay=weight_decay)
    train_info = TrainInfo(wait_iterations, eval_batches)
    while not train_info.finish:
        steps = 0
        for (text, maps), (heads, rels), (transitions, relations_in_order) in trainloader:
            steps += 1
            # maps are used to average the split embeddings from BERT
            loss = train_batch(text, heads, rels, transitions, relations_in_order, maps, model, optimizer)
            #lr_scheduler.step()
            train_info.new_batch(loss)
            if train_info.eval:
                dev_results = evaluate(devloader, model)
                if train_info.is_best(dev_results):
                    model.set_best()
                    # if save_batch:
                    model.save(save_path)
                elif train_info.reduce_lr:
                    lr_scheduler.step()
                    optimizer.state.clear()
                    model.recover_best()
                    print('\tReduced lr')
                elif train_info.finish:
                    train_info.print_progress(dev_results, file)
                    break
                train_info.print_progress(dev_results, file)

    model.recover_best()


def main():
    # sys.stdout = open("test.txt", "w")
    # pylint: disable=too-many-locals

    args = get_args()
    wandb.login(key=args.key)

    ef = args.easy_first == "True"
    s = None
    if ef:
        s = "EasyFirst"
    else:
        s = "ShiftReduce"
    trainloader, devloader, testloader, vocabs, max_sent_len = \
        get_data_loaders(args.data_path, args.language, args.batch_size, args.batch_size_eval,
                         is_easy_first=ef)
    print('Train size: %d Dev size: %d Test size: %d' %
          (len(trainloader.dataset), len(devloader.dataset), len(testloader.dataset)))
    save_name = "final_output_%s_%s.txt".format(s,args.language)
    file1 = open(save_name, "w")
    WANDB_PROJECT = f"{args.language}_{s}"

    model = get_model(vocabs, args)
    run = wandb.init(project=WANDB_PROJECT, config={'wandb_nb': 'wandb_three_in_one_hm'},
                     settings=wandb.Settings(start_method="fork"))

    # Start tracking your model's gradients
    wandb.watch(model)
    # if args.model != 'agenda-std':
    num_epochs = 10
    train(trainloader, devloader, model, args.eval_batches, args.wait_iterations, args.lr, args.weight_decay, args.save_path,
          args.save_periodically, file=file1)
    model.save(args.save_path)
    train_loss, train_las, train_uas = evaluate(trainloader, model)
    dev_loss, dev_las, dev_uas = evaluate(devloader, model)
    test_loss, test_las, test_uas = evaluate(testloader, model)

    file1.write('Final Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
                (train_loss, dev_loss, test_loss))
    file1.write('Final Training las: %.4f Dev las: %.4f Test las: %.4f' %
                (train_las, dev_las, test_las))
    file1.write('Final Training uas: %.4f Dev uas: %.4f Test uas: %.4f' %
                (train_uas, dev_uas, test_uas))

    log_dict_loss = {'Training loss': train_loss, 'Dev Loss': dev_loss, 'Test Loss': test_loss}
    wandb.log(log_dict_loss)
    log_dict_las = {"Training LAS": train_las, "Dev LAS": dev_las, "Test LAS": test_las}
    wandb.log(log_dict_las)
    log_dict_uas = {"Training UAS": train_uas, "Dev UAS": dev_uas, "Test UAS": test_uas}
    wandb.log(log_dict_uas)
    file1.close()
    wandb.finish()
    print('Final Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
          (train_loss, dev_loss, test_loss))
    print('Final Training las: %.4f Dev las: %.4f Test las: %.4f' %
          (train_las, dev_las, test_las))
    print('Final Training uas: %.4f Dev uas: %.4f Test uas: %.4f' %
          (train_uas, dev_uas, test_uas))

    # sys.stdout.close()


if __name__ == '__main__':
    main()
