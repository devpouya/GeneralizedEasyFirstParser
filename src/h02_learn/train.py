import sys
import argparse
import torch
import torch.optim as optim

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders
from h02_learn.model import BiaffineParser, MSTParser, ArcStandardStackLSTM, \
    ArcEagerStackLSTM, HybridStackLSTM, NonProjectiveStackLSTM
from h02_learn.model import NeuralTransitionParser
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
    parser.add_argument('--nlayers', type=int, default=3)
    parser.add_argument('--embedding-size', type=int, default=100)
    parser.add_argument('--hidden-size', type=int, default=400)
    parser.add_argument('--arc-size', type=int, default=500)
    parser.add_argument('--label-size', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=.33)
    parser.add_argument('--model', choices=['biaffine', 'mst', 'arc-standard',
                                            'arc-eager', 'hybrid', 'non-projective'],
                        default='arc-standard')
    # Optimization
    parser.add_argument('--optim', choices=['adam', 'adamw', 'sgd'], default='adam')
    parser.add_argument('--eval-batches', type=int, default=20)
    parser.add_argument('--wait-epochs', type=int, default=10)
    parser.add_argument('--lr-decay', type=float, default=.5)
    # Save
    parser.add_argument('--checkpoints-path', type=str, default='checkpoints/')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--save-periodically', action='store_true')

    args = parser.parse_args()
    args.wait_iterations = args.wait_epochs * args.eval_batches
    args.save_path = '%s/%s/%s/%s/' % (args.checkpoints_path, args.language, args.model, args.batch_size)
    utils.config(args.seed)
    return args


def get_optimizer(paramters, optim_alg, lr_decay):
    if optim_alg == "adamw":
        optimizer = optim.AdamW(paramters, betas=(.9, .9))
    elif optim_alg == "adam":
        optimizer = optim.Adam(paramters, betas=(.9, .9))
    else:
        optimizer = optim.SGD(paramters, lr=0.01)

    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)
    return optimizer, lr_scheduler


def get_model(vocabs, embeddings, args):
    if args.model == 'mst':
        return MSTParser(
            vocabs, args.embedding_size, args.hidden_size, args.arc_size, args.label_size,
            nlayers=args.nlayers, dropout=args.dropout, pretrained_embeddings=embeddings) \
            .to(device=constants.device)
    if args.model == 'arc-standard':
        return NeuralTransitionParser(
            vocabs, args.embedding_size, args.hidden_size, args.arc_size, args.label_size, args.batch_size,
            nlayers=args.nlayers, dropout=args.dropout, pretrained_embeddings=embeddings,
            transition_system=constants.arc_standard) \
            .to(device=constants.device)
    elif args.model == 'arc-eager':
        return NeuralTransitionParser(
            vocabs, args.embedding_size, args.hidden_size, args.arc_size, args.label_size, args.batch_size,
            nlayers=args.nlayers, dropout=args.dropout, pretrained_embeddings=embeddings,
            transition_system=constants.arc_eager) \
            .to(device=constants.device)
    elif args.model == 'hybrid':
        return HybridStackLSTM(
            vocabs, args.embedding_size, args.hidden_size, args.arc_size, args.label_size,
            nlayers=args.nlayers, dropout=args.dropout, pretrained_embeddings=embeddings) \
            .to(device=constants.device)
    elif args.model == 'non-projective':
        return NonProjectiveStackLSTM(
            vocabs, args.embedding_size, args.hidden_size, args.arc_size, args.label_size,
            nlayers=args.nlayers, dropout=args.dropout, pretrained_embeddings=embeddings) \
            .to(device=constants.device)
    else:
        return BiaffineParser(
            vocabs, args.embedding_size, args.hidden_size, args.arc_size, args.label_size,
            nlayers=args.nlayers, dropout=args.dropout, pretrained_embeddings=embeddings) \
            .to(device=constants.device)


"""
def calculate_attachment_score(heads_tgt, l_logits, heads, rels):
    acc_h = (heads_tgt == heads)[heads != -1]
    acc_l = (l_logits.argmax(-1) == rels)[heads != -1]

    uas = acc_h.float().mean().item()
    las = (acc_h & acc_l).float().mean().item()

    return las, uas
"""


def calculate_attachment_score(heads_tgt, heads,predicted_rels,rels):
    acc_h = (heads_tgt == heads)[heads != -1]
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
    for (text, pos), (heads, rels), (transitions, relations_in_order) in evalloader:
        steps += 1
        # print(steps)
        # h_logits, l_logits = model((text, pos))
        # h_logits = model((text, pos),heads)
        #print("********************{}***************".format("EVAAL"))

        loss, predicted_heads, predicted_rels = model((text, pos), transitions, relations_in_order, mode='eval')
        # loss = model.loss(h_logits, l_logits, heads, rels)
        lengths = (text != 0).sum(-1)

        # heads_tgt = get_mst_batch(h_logits, lengths)

        #print("predicted heads {}".format(predicted_heads))
        #print("predicted rels {}".format(predicted_rels))
        #print("real heads {}".format(heads))
        #print((torch.eq(predicted_heads, heads)==True).sum(0))
        #print((torch.eq(predicted_rels, rels)==True).sum(0))

        #print("real rels {}".format(rels))
        #print("********************{}***************".format("EVAAL"))

        las, uas = calculate_attachment_score(predicted_heads, heads,predicted_rels,rels)
        # uas = simple_attachment_scores(predicted_heads,heads,lengths)
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


def train_batch(text, pos, heads, rels, transitions, relations_in_order, model, optimizer):
    optimizer.zero_grad()

    text, pos = text.to(device=constants.device), pos.to(device=constants.device)
    #heads, rels = heads.to(device=constants.device), rels.to(device=constants.device)
    transitions = transitions.to(device=constants.device)
    relations_in_order = relations_in_order.to(device=constants.device)
    #print("********************{}***************".format("TRAAAAIN"))

    loss, pred_h ,pred_rel= model((text, pos), transitions, relations_in_order,mode='train')
    #for param in model.parameters():
    #    print("*********{}*******************".format(loss))
    #    print(param.data)
    #    print("*********{}*******************".format(loss))

    #print("predicted heads {}".format(pred_h))
    #print("predicted rels {}".format(pred_rel))
    #print(torch.all(torch.eq(pred_h,heads)).item())
    #print(torch.all(torch.eq(pred_rel,rels)).item())

    #print("real heads {}".format(heads))
    #print("real rels {}".format(rels))
    #print("********************{}***************".format("TRAAAAIN"))

    loss.backward(retain_graph=True)
    optimizer.step()

    return loss.item()


def train(trainloader, devloader, model, eval_batches, wait_iterations, optim_alg, lr_decay,
          save_path, save_batch=False):
    # pylint: disable=too-many-locals,too-many-arguments
    optimizer, lr_scheduler = get_optimizer(model.parameters(), optim_alg, lr_decay)
    train_info = TrainInfo(wait_iterations, eval_batches)
    while not train_info.finish:
        steps = 0

        for (text, pos), (heads, rels), (transitions, relations_in_order) in trainloader:

            steps += 1
            loss = train_batch(text, pos, heads, rels, transitions, relations_in_order, model, optimizer)
            # print("train loss in step {} is {}".format(steps,loss))
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


def main():
    # pylint: disable=too-many-locals
    args = get_args()
    if args.model == "arc-standard":
        transition_system = constants.arc_standard
    elif args.model == "arc-eager":
        transition_system = constants.arc_eager
    elif args.model == "hybrid":
        transition_system = constants.hybrid

    trainloader, devloader, testloader, vocabs, embeddings = \
        get_data_loaders(args.data_path, args.language, args.batch_size, args.batch_size_eval, args.model,
                         transition_system)
    print('Train size: %d Dev size: %d Test size: %d' %
          (len(trainloader.dataset), len(devloader.dataset), len(testloader.dataset)))

    model = get_model(vocabs, embeddings, args)
    train(trainloader, devloader, model, args.eval_batches, args.wait_iterations,
          args.optim, args.lr_decay, args.save_path, args.save_periodically)

    model.save(args.save_path)

    train_loss, train_las, train_uas = evaluate(trainloader, model)
    dev_loss, dev_las, dev_uas = evaluate(devloader, model)
    test_loss, test_las, test_uas = evaluate(testloader, model)

    print('Final Training loss: %.4f Dev loss: %.4f Test loss: %.4f' %
          (train_loss, dev_loss, test_loss))
    print('Final Training las: %.4f Dev las: %.4f Test las: %.4f' %
          (train_las, dev_las, test_las))
    print('Final Training uas: %.4f Dev uas: %.4f Test uas: %.4f' %
          (train_uas, dev_uas, test_uas))


if __name__ == '__main__':
    main()
