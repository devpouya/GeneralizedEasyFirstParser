import sys
import argparse

sys.path.append('./src/')
from h02_learn.dataset import get_data_loaders
from h02_learn.model import BiaffineParser
from h02_learn.train import evaluate
from utils import constants


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--language', type=str, required=True)
    parser.add_argument('--data-path', type=str, default='data/')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--name', type=str)
    parser.add_argument('--model', choices=['biaffine', 'mst', 'arc-standard',
                                            'arc-eager', 'hybrid', 'non-projective'],
                        default='arc-standard')
    # Model
    parser.add_argument('--checkpoints-path', type=str, default='checkpoints/')

    return parser.parse_args()


def load_model(checkpoints_path, language,model,name):
    load_path = '%s/%s/%s/%s/' % (checkpoints_path, language,model,name)
    return BiaffineParser.load(load_path).to(device=constants.device)


def main():
    # pylint: disable=too-many-locals
    args = get_args()
    if args.model == "arc-standard":
        transition_system = constants.arc_standard
    elif args.model == "arc-eager":
        transition_system = constants.arc_eager
    elif args.model == "hybrid":
        transition_system = constants.hybrid

    trainloader, devloader, testloader, _, _ = \
        get_data_loaders(args.data_path, args.language, args.batch_size,args.batch_size, args.model,
                         transition_system=transition_system, bert_model='bert-base-cased')
    print('Train size: %d Dev size: %d Test size: %d' %
          (len(trainloader.dataset), len(devloader.dataset), len(testloader.dataset)))

    model = load_model(args.checkpoints_path, args.language,args.model,args.name)

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
