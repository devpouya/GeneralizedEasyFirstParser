import sys
import argparse
import torch
import torch.optim as optim

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders
from h02_learn.model import BiaffineParser, MSTParser
from h02_learn.model import NeuralTransitionParser, EasyFirstParser, ChartParser
from h02_learn.model import LazyMH4,LazyHybrid,LazyArcEager,LazyArcStandard
from h02_learn.train_info import TrainInfo
from h02_learn.algorithm.mst import get_mst_batch
from utils import constants
from utils import utils


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--language', type=str, required=True)
    parser.add_argument('--data-path', type=str, default='data/')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--batch-size-eval', type=int, default=128)
    # Model
    parser.add_argument('--embedding-size', type=int, default=768)
    parser.add_argument('--rel-embedding-size', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=.33)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--model', choices=['easy-first','easy-first-hybrid','biaffine', 'mst', 'arc-standard',
                                            'arc-eager', 'hybrid', 'mh4','easy-first-mh4','chart','agenda-std'],
                        default='agenda-std')
    parser.add_argument('--bert-model',type=str,default='bert-base-cased')
    # Optimization
    parser.add_argument('--optim', choices=['adam', 'adamw', 'sgd'], default='adamw')
    parser.add_argument('--eval-batches', type=int, default=20)
    parser.add_argument('--wait-epochs', type=int, default=10)
    parser.add_argument('--lr-decay', type=float, default=.5)
    # Save
    parser.add_argument('--name', type=str, default='generic-experiment')
    parser.add_argument('--checkpoints-path', type=str, default='checkpoints/')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--save-periodically', action='store_true')

    args = parser.parse_args()
    args.wait_iterations = args.wait_epochs * args.eval_batches
    args.save_path = '%s/%s/%s/%s/' % (args.checkpoints_path, args.language, args.model, args.name)
    utils.config(args.seed)
    print("RUNNING {}".format(args.name))
    return args


def get_optimizer(paramters, optim_alg, lr_decay, weight_decay):
    if optim_alg == "adamw":
        optimizer = optim.AdamW(paramters, betas=(.9, .9), weight_decay=weight_decay)
    elif optim_alg == "adam":
        optimizer = optim.Adam(paramters, betas=(.9, .9), weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(paramters, lr=0.01)

    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)
    #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max")
    return optimizer, lr_scheduler


def get_model(vocabs,embeddings,args,max_sent_len):

    if args.model == 'arc-standard': # or args.model=='easy-first':
        return NeuralTransitionParser(
            vocabs=vocabs, embedding_size=args.embedding_size,rel_embedding_size=args.rel_embedding_size, batch_size=args.batch_size,
            dropout=args.dropout,
            transition_system=constants.arc_standard) \
            .to(device=constants.device)
    elif args.model == 'chart':
        return ChartParser(vocabs=vocabs, embedding_size=args.embedding_size, rel_embedding_size=args.rel_embedding_size, batch_size=args.batch_size,
                           hypergraph=LazyArcStandard,dropout=0.33, beam_size=10,max_sent_len=max_sent_len, easy_first=False).to(device=constants.device)
    elif args.model == 'easy-first':
        return EasyFirstParser(vocabs=vocabs, embedding_size=args.embedding_size,rel_embedding_size=args.rel_embedding_size, batch_size=args.batch_size,
            dropout=args.dropout,
            transition_system=constants.easy_first) \
            .to(device=constants.device)
    elif args.model == 'arc-eager':
        return NeuralTransitionParser(
            vocabs=vocabs, embedding_size=args.embedding_size,rel_embedding_size=args.rel_embedding_size, batch_size=args.batch_size,
            dropout=args.dropout,
            transition_system=constants.arc_eager) \
            .to(device=constants.device)
    elif args.model == 'hybrid' or args.model == 'easy-first-hybrid':
        return NeuralTransitionParser(
            vocabs=vocabs, embedding_size=args.embedding_size,rel_embedding_size=args.rel_embedding_size, batch_size=args.batch_size,
            dropout=args.dropout,
            transition_system=constants.hybrid) \
            .to(device=constants.device)
    elif args.model == 'mh4' or args.model == 'easy-first-mh4':
        return NeuralTransitionParser(
            vocabs=vocabs, embedding_size=args.embedding_size,rel_embedding_size=args.rel_embedding_size, batch_size=args.batch_size,
            dropout=args.dropout,
            transition_system=constants.mh4) \
            .to(device=constants.device)
    elif args.model == 'agenda-std':
        return ChartParser(vocabs=vocabs, embedding_size=args.embedding_size, rel_embedding_size=args.rel_embedding_size, batch_size=args.batch_size,
                           hypergraph=LazyArcStandard,dropout=0.33, beam_size=10,max_sent_len=max_sent_len, easy_first=False).to(device=constants.device)
    else:
        return BiaffineParser(
            vocabs, args.embedding_size, args.hidden_size, args.arc_size, args.label_size,
            nlayers=args.nlayers, dropout=args.dropout, pretrained_embeddings=embeddings) \
            .to(device=constants.device)

def calculate_attachment_score(heads_tgt, heads, predicted_rels, rels):
    predicted_rels = predicted_rels.permute(1,0)
    acc_h = (heads_tgt == heads)[heads != -1]
    #predicted_rels = predicted_rels[predicted_rels != -1]
    #rels = rels[rels != -1]
    #print(predicted_rels.shape)
    #print(rels.shape)
    rels = rels.permute(1,0)
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
    for (text, pos), (heads, rels), (transitions, relations_in_order),maps in evalloader:
        steps += 1
        maps = maps.to(device=constants.device)
        text, pos = text.to(device=constants.device), pos.to(device=constants.device)
        heads, rels = heads.to(device=constants.device), rels.to(device=constants.device)
        transitions = transitions.to(device=constants.device)
        relations_in_order = relations_in_order.to(device=constants.device)
        loss, predicted_heads, predicted_rels = model((text, pos), transitions, relations_in_order,maps,heads=heads,rels=rels)
        #print("EEEEEEVAAAAAAALLLLLLLLLLEVAAAAALLLLLLLLLEVALLLL")
        #print("predicted heads {}".format(predicted_heads))
        #print("real heads {}".format(heads))
        ##print(torch.all(torch.eq(heads, predicted_heads)))
        #print("--------------------------------")
        #print("predicted rels {}".format(predicted_rels))
        #print("real rels {}".format(rels))
        ##print(torch.all(torch.eq(predicted_rels, rels)))
        #print("EEEEEEVAAAAAAALLLLLLLLLLEVAAAAALLLLLLLLLEVALLLL")

        #jhjh
        # loss = model.loss(h_logits, l_logits, heads, rels)
        lengths = (text != 0).sum(-1)
        # heads_tgt = get_mst_batch(h_logits, lengths)
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
        result = _evaluate(evalloader, model)
    model.train()
    return result


def train_batch(text, pos, heads, rels, transitions, relations_in_order, maps,model, optimizer):
    optimizer.zero_grad()
    maps = maps.to(device=constants.device)
    text, pos = text.to(device=constants.device), pos.to(device=constants.device)
    heads, rels = heads.to(device=constants.device), rels.to(device=constants.device)

    transitions = transitions.to(device=constants.device)
    relations_in_order = relations_in_order.to(device=constants.device)

    loss, pred_h, pred_rel = model((text, pos), transitions, relations_in_order,maps,heads=heads,rels=rels)
    #print("çççççççççççççççççççççççççççççççç")
    #print(pred_h)
    #print(heads)
    ##(torch.all(torch.eq(heads,pred_h)))
    #print("--------------------------------")
    #print(pred_rel)
    #print(rels)
    ##(torch.all(torch.eq(pred_rel,rels)))
    #print("çççççççççççççççççççççççççççççççç")

    loss.backward()
    optimizer.step()

    return loss.item()


def train(trainloader, devloader, model, eval_batches, wait_iterations, optim_alg, lr_decay, weight_decay,
          save_path, save_batch=False):
    # pylint: disable=too-many-locals,too-many-arguments
    torch.autograd.set_detect_anomaly(True)

    optimizer, lr_scheduler = get_optimizer(model.parameters(), optim_alg, lr_decay, weight_decay)
    train_info = TrainInfo(wait_iterations, eval_batches)
    while not train_info.finish:
        steps = 0
        for (text, pos), (heads, rels), (transitions, relations_in_order),maps in trainloader:
            steps += 1
            # maps are used to average the split embeddings from BERT
            loss = train_batch(text, pos, heads, rels, transitions, relations_in_order, maps,model, optimizer)
            train_info.new_batch(loss)
            if train_info.eval:
                dev_results = evaluate(devloader, model)
                if train_info.is_best(dev_results):
                    model.set_best()
                    if save_batch:
                        model.save(save_path)
                elif train_info.reduce_lr:
                    lr_scheduler.step()
                    optimizer.state.clear()
                    model.recover_best()
                    print('\tReduced lr')
                elif train_info.finish:
                    train_info.print_progress(dev_results)
                    break
                train_info.print_progress(dev_results)

    model.recover_best()


def train_batch_chart(text,pos, heads, rels, maps, model, optimizer):
    optimizer.zero_grad()
    maps = maps.to(device=constants.device)
    text, pos = text.to(device=constants.device), pos.to(device=constants.device)
    heads, rels = heads.to(device=constants.device), rels.to(device=constants.device)

    loss, pred_h, pred_rel = model((text, pos),maps,heads,rels)
    #print("çççççççççççççççççççççççççççççççç")
    #print(pred_h)
    #print(heads)
    #print(torch.all(torch.eq(heads, pred_h[0])))
    #print("--------------------------------")
    ## print(pred_rel)
    ## print(rels)
    #print(rels.shape)
    #print(pred_rel.shape)
    ##print(torch.all(torch.eq(pred_rel, rels)))
    #print("çççççççççççççççççççççççççççççççç")

    loss.backward()
    optimizer.step()

    return loss.item()
def train_chart(trainloader, devloader, model, eval_batches, wait_iterations, optim_alg, lr_decay, weight_decay,
                save_path, save_batch=False):
    torch.autograd.set_detect_anomaly(True)
    optimizer, lr_scheduler = get_optimizer(model.parameters(), optim_alg, lr_decay, weight_decay)
    train_info = TrainInfo(wait_iterations,eval_batches)
    while not train_info.finish:
        step = 0
        for (text, pos), (heads, rels),(hypergraph,relation_in_order), maps in trainloader:
            loss = train_batch_chart(text,pos, heads, rels, maps, model, optimizer)
            train_info.new_batch(loss)
            if train_info.eval:
                dev_results = evaluate_chart(devloader, model)
                if train_info.is_best(dev_results):
                    model.set_best()
                    if save_batch:
                        model.save(save_path)
                elif train_info.reduce_lr:
                    lr_scheduler.step()
                    optimizer.state.clear()
                    model.recover_best()
                    print('\tReduced lr')
                elif train_info.finish:
                    train_info.print_progress(dev_results)
                    break
                train_info.print_progress(dev_results)

        model.recover_best()


def _evaluate_chart(evalloader, model):
    # pylint: disable=too-many-locals
    dev_loss, dev_las, dev_uas, n_instances = 0, 0, 0, 0
    steps = 0
    for (text, pos), (heads, rels),_, maps in evalloader:
        steps += 1
        maps = maps.to(device=constants.device)
        text, pos = text.to(device=constants.device), pos.to(device=constants.device)
        heads, rels = heads.to(device=constants.device), rels.to(device=constants.device)
        loss, predicted_heads, predicted_rels = model((text, pos), maps, heads, rels) # model((text, pos), maps,rels)

        #print("EEEEEEVAAAAAAALLLLLLLLLLEVAAAAALLLLLLLLLEVALLLL")
        #print("predicted heads {}".format(predicted_heads))
        #print("real heads {}".format(heads))
        #print(torch.all(torch.eq(heads, predicted_heads)))
        #print("--------------------------------")
        #print("predicted rels {}".format(predicted_rels.shape))
        #print("real rels {}".format(rels.shape))
        #print(torch.all(torch.eq(predicted_rels, rels)))
        #print("EEEEEEVAAAAAAALLLLLLLLLLEVAAAAALLLLLLLLLEVALLLL")
        # loss = model.loss(h_logits, l_logits, heads, rels)
        lengths = (text != 0).sum(-1)
        # heads_tgt = get_mst_batch(h_logits, lengths)
        las, uas = calculate_attachment_score(predicted_heads, heads, predicted_rels, rels)
        batch_size = text.shape[0]
        dev_loss += (loss * batch_size)
        dev_las += (las * batch_size)
        dev_uas += (uas * batch_size)
        n_instances += batch_size

    return dev_loss / n_instances, dev_las / n_instances, dev_uas / n_instances

def evaluate_chart(evalloader, model):
    model.eval()
    with torch.no_grad():
        result = _evaluate_chart(evalloader, model)
    model.train()
    return result


def main():
    # pylint: disable=too-many-locals
    args = get_args()
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
    trainloader, devloader, testloader, vocabs, embeddings,max_sent_len = \
        get_data_loaders(args.data_path, args.language, args.batch_size, args.batch_size_eval, fname,
                         transition_system=transition_system, bert_model=args.bert_model)
    print('Train size: %d Dev size: %d Test size: %d' %
          (len(trainloader.dataset), len(devloader.dataset), len(testloader.dataset)))

    model = get_model(vocabs, embeddings, args,max_sent_len)
    #if args.model != 'agenda-std':
    train(trainloader, devloader, model, args.eval_batches, args.wait_iterations,
          args.optim, args.lr_decay, args.weight_decay, args.save_path, args.save_periodically)
    model.save(args.save_path)
    train_loss, train_las, train_uas = evaluate(trainloader, model)
    dev_loss, dev_las, dev_uas = evaluate(devloader, model)
    test_loss, test_las, test_uas = evaluate(testloader, model)
    #else:
    #    model = get_model(vocabs, embeddings, args,max_sent_len)
    #    train_chart(trainloader, devloader, model, args.eval_batches, args.wait_iterations,
    #          args.optim, args.lr_decay, args.weight_decay, args.save_path, args.save_periodically)
    #    model.save(args.save_path)
    #    train_loss, train_las, train_uas = evaluate_chart(trainloader, model)
    #    dev_loss, dev_las, dev_uas = evaluate_chart(devloader, model)
    #    test_loss, test_las, test_uas = evaluate_chart(testloader, model)

    print('Final Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
          (train_loss, dev_loss, test_loss))
    print('Final Training las: %.4f Dev las: %.4f Test las: %.4f' %
          (train_las, dev_las, test_las))
    print('Final Training uas: %.4f Dev uas: %.4f Test uas: %.4f' %
          (train_uas, dev_uas, test_uas))


if __name__ == '__main__':
    main()
