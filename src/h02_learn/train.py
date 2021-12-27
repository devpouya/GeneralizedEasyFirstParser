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
import numpy as np
from transformers import get_linear_schedule_with_warmup
import wandb


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--language', type=str, required=True)
    parser.add_argument('--data-path', type=str, default='data_nonproj/')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--batch-size-eval', type=int, default=128)
    parser.add_argument('--key', type=str)
    # Model

    parser.add_argument('--embedding-size', type=int, default=768)
    parser.add_argument('--hidden-size', type=int, default=100)
    parser.add_argument('--rel-embedding-size', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=.33)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--model', choices=['easy-first', 'easy-first-hybrid', 'biaffine', 'mst', 'arc-standard',
                                            'arc-eager', 'hybrid', 'mh4', 'easy-first-mh4', 'chart', 'agenda-std',
                                            'agenda-hybrid', 'agenda-eager', 'agenda-mh4'],
                        default='agenda-mh4')
    parser.add_argument('--mode', choices=['shift-reduce', 'easy-first'], default='easy-first')
    parser.add_argument('--bert-model', type=str, default='bert-base-cased')
    # Optimization
    parser.add_argument('--optim', choices=['adam', 'adamw', 'sgd'], default='adam')
    parser.add_argument('--eval-batches', type=int, default=20)
    parser.add_argument('--wait-epochs', type=int, default=10)
    parser.add_argument('--lr-decay', type=float, default=.5)
    # Save
    parser.add_argument('--name', type=str, default='generic-experiment')
    parser.add_argument('--checkpoints-path', type=str, default='checkpoints/')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--save-periodically', action='store_true')

    args = parser.parse_args()
    args.wait_iterations = 3  # args.wait_epochs * args.eval_batches
    s = args.mode
    args.save_path = '%s/%s/%s/%s/' % (args.checkpoints_path, s, args.model, args.name)
    utils.config(args.seed)
    print("RUNNING {}".format(args.name))
    return args


def get_optimizer(paramters, optim_alg, lr_decay, weight_decay):
    if optim_alg == "adamw":
        optimizer = optim.AdamW(paramters, betas=(.9, .9), weight_decay=weight_decay,lr=1e-5)
    elif optim_alg == "adam":
        optimizer = optim.Adam(paramters, betas=(0.9,0.999),eps=1e-08,lr=2e-5)
    else:
        optimizer = optim.SGD(paramters, lr=0.01)

    #lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max")
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=9)
    return optimizer, lr_scheduler


def get_model(lang, num_rels, args, ef):
    return ChartParser(lang=lang,num_rels=num_rels,
                       batch_size=args.batch_size,
                       hypergraph=MH4, dropout=args.dropout, is_easy_first=ef).to(
        device=constants.device)


def calculate_attachment_score(heads_tgt, heads, predicted_rels, rels):
    predicted_rels = predicted_rels.permute(1, 0)
    acc_h = (heads_tgt == heads)[heads != -1]
    # predicted_rels = predicted_rels[predicted_rels != -1]
    # rels = rels[rels != -1]
    # print(predicted_rels.shape)
    # print(rels.shape)
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
    for (text, pos), (heads, rels), (transitions, relations_in_order), maps in evalloader:
        steps += 1
        maps = maps.to(device=constants.device)
        text, pos = text.to(device=constants.device), pos.to(device=constants.device)
        heads, rels = heads.to(device=constants.device), rels.to(device=constants.device)
        transitions = transitions.to(device=constants.device)
        relations_in_order = relations_in_order.to(device=constants.device)
        loss, predicted_heads, predicted_rels = model((text, pos), transitions, relations_in_order, maps, heads=heads,
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
    with torch.no_grad():
        #result = _evaluate(evalloader, model)
        results_per_language = evaluate_per_language(evalloader,model)

    model.train()
    return results_per_language

def evaluate_per_language(evalloader, model):
    results_per_language = {}
    for language in evalloader.keys():
        evalloader_language = evalloader[language]
        result_language = _evaluate(evalloader_language, model)
        results_per_language[language] = result_language
    return results_per_language



def train_batch(text, pos, heads, rels, transitions, relations_in_order, maps, model, optimizer, lr_scheduler):
    #optimizer.zero_grad()
    maps = maps.to(device=constants.device)
    text, pos = text.to(device=constants.device), pos.to(device=constants.device)
    heads, rels = heads.to(device=constants.device), rels.to(device=constants.device)

    transitions = transitions.to(device=constants.device)
    relations_in_order = relations_in_order.to(device=constants.device)

    loss, pred_h, pred_rel = model((text, pos), transitions, relations_in_order, maps, heads=heads, rels=rels)

    # las, uas = calculate_attachment_score(pred_h, heads, pred_rel, rels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    optimizer.step()
    lr_scheduler.step()

    # shit = 0
    # total = 0
    ##for item in model.parameters():
    ##    total += 1
    ##    if item.grad is None:
    #        shit += 1
    # print("SHIT {} OF {}".format(shit, total))
    return loss.item()


def train(trainloader, devloader, model, eval_batches, wait_iterations, optim_alg, lr_decay, weight_decay,
          save_path, save_batch=False, file=None):
    # pylint: disable=too-many-locals,too-many-arguments
    torch.autograd.set_detect_anomaly(True)
    #optimizer, lr_scheduler = get_optimizer(model.parameters(), optim_alg, lr_decay, weight_decay)
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-08, lr=2e-5)
    num_iter = trainloader.dataset.n_instances*5
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_iter)
    train_info = TrainInfo(wait_iterations, eval_batches)
    for epoch in range(5):
        steps = 0
        for (text, pos), (heads, rels), (transitions, relations_in_order), maps in trainloader:
            steps += 1
            # maps are used to average the split embeddings from BERT
            model.zero_grad()
            loss = train_batch(text, pos, heads, rels, transitions, relations_in_order, maps, model, optimizer, lr_scheduler)
            train_info.new_batch(loss)
            if train_info.eval:
                dev_results = evaluate(devloader, model)
                if train_info.is_best(dev_results):
                    model.set_best()
                    # if save_batch:
                    model.save(save_path)
                #elif train_info.reduce_lr:
                #    lr_scheduler.step()
                #    optimizer.state.clear()
                #    model.recover_best()
                #    print('\tReduced lr')
                #elif train_info.finish:
                #    train_info.print_progress(dev_results, file)
                #    break
                train_info.print_progress(dev_results, file)

    model.recover_best()


def main():
    # sys.stdout = open("test.txt", "w")
    # pylint: disable=too-many-locals

    args = get_args()
    wandb.login(key=args.key)

    if args.model == "arc-standard":  # or args.model == "easy-first":
        transition_system = constants.arc_standard
    elif args.model == "easy-first":
        transition_system = constants.easy_first
    elif args.model == "arc-eager":
        transition_system = constants.arc_eager
    elif args.model == "hybrid" or args.model == "easy-first-hybrid":
        transition_system = constants.hybrid
    elif args.model == "mh4" or args.model == 'easy-first-mh4':
        transition_system = constants.mh4
    else:
        transition_system = constants.agenda

    if args.model == 'chart':
        fname = "arc-standard"
    else:
        fname = args.model

    #all_languages = ["af", "da", "eu", "ga", "hu", "ko", "la", "lt", "nl", "qhe", "sl", "ur"]
    if args.language == "multilingual":
        all_languages = ["af", "da", "eu", "ga", "hu", "ko", "la", "nl", "ur"]
    else:
        all_languages = [args.language]
    sizes = []
    trainloader_dict = {}
    testloader_dict = {}
    devloader_dict = {}
    max_num_rels = 0
    if args.mode == "easy-first":
        ef = True
    else:
        ef = False
    #for ind, lang in enumerate(all_languages):
    trainloader, devloader, testloader, rels_size = \
        get_data_loaders(args.data_path, all_languages, args.batch_size, args.batch_size_eval, fname,
                         transition_system=transition_system, bert_model=args.bert_model, is_easy_first=ef)


    save_name = "final_output_%s.txt".format(args.model)
    file1 = open(save_name, "w")
    s = args.mode
    WANDB_PROJECT = f"{s}_{args.model}"
    # WANDB_PROJECT = "%s_%s".format(args.language,args.model)
    model = get_model(args.language, rels_size, args, ef)
    run = wandb.init(project=WANDB_PROJECT, config={'wandb_nb': 'wandb_three_in_one_hm'},
                     settings=wandb.Settings(start_method="fork"))

    # Start tracking your model's gradients
    wandb.watch(model)
    # if args.model != 'agenda-std':
    train(trainloader, devloader, model, args.eval_batches, args.wait_iterations,
          args.optim, args.lr_decay, args.weight_decay, args.save_path, args.save_periodically, file=file1)
    model.save(args.save_path)

    #train_results = evaluate(trainloader, model)
    #train_loss, train_las, train_uas = evaluate(trainloader, model)
    dev_results = evaluate(devloader, model)
    #dev_loss, dev_las, dev_uas = evaluate(devloader, model)
    test_results = evaluate(testloader, model)
    #test_loss, test_las, test_uas = evaluate(testloader, model)
    for language in dev_results.keys():
        dev_loss, dev_las, dev_uas = dev_results[language]
        test_loss, test_las, test_uas = test_results[language]
        #file1.write('Final Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
        #            (train_loss, dev_loss, test_loss))
        #file1.write('Final Training las: %.4f Dev las: %.4f Test las: %.4f' %
        #            (train_las, dev_las, test_las))
        #file1.write('Final Training uas: %.4f Dev uas: %.4f Test uas: %.4f' %
        #            (train_uas, dev_uas, test_uas))
        devlosslang_str = "Dev Loss {}".format(language)
        devLASlang_str = "Dev LAS {}".format(language)
        devUASlang_str = "Dev UAS {}".format(language)
        testlosslang_str = "Test Loss {}".format(language)
        testLASlang_str = "Test LAS {}".format(language)
        testUASlang_str = "Test UAS {}".format(language)
        log_dict_loss = {devlosslang_str: dev_loss, testlosslang_str: test_loss}
        wandb.log(log_dict_loss)
        log_dict_las = {devLASlang_str: dev_las, testLASlang_str: test_las}
        wandb.log(log_dict_las)
        log_dict_uas = {devUASlang_str: dev_uas, testUASlang_str: test_uas}
        wandb.log(log_dict_uas)
        file1.close()
        wandb.finish()
        #print('Final Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
        #      (train_loss, dev_loss, test_loss))
        #print('Final Training las: %.4f Dev las: %.4f Test las: %.4f' %
        #      (train_las, dev_las, test_las))
        #print('Final Training uas: %.4f Dev uas: %.4f Test uas: %.4f' %
        #      (train_uas, dev_uas, test_uas))

    # sys.stdout.close()


if __name__ == '__main__':
    main()
