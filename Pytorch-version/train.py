import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from src.model import SyntaxTextCNN
from src.dataset import *

seed = 1
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
else:
    torch.manual_seed(seed)
torch.random.manual_seed(1234)

parser = argparse.ArgumentParser(
    "Implementation of the model: Syntax Encoding with Application in Authorship Attribution")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=4 * 1e-4)
parser.add_argument("--text_out_dim", type=int, default="500")
parser.add_argument("--syntax_out_dim", type=int, default="50")
parser.add_argument("--text_embed_size", type=int, default="300")
parser.add_argument("--syntax_embed_size", type=int, default="60")
parser.add_argument("--hidden", type=int, default=256, help="hidden size of transformer model")
parser.add_argument("--layers", type=int, default=8, help="number of layers")
parser.add_argument("--attn_heads", type=int, default=8, help="number of attention heads")
parser.add_argument("--dataset", type=str, default="ag_news", choices=datasets)
parser.add_argument("--author_num", type=int, default=4)
parser.add_argument("--pre_trained", type=str, default=None)
args = parser.parse_args()


def train():
    test_dataset = SyntaxDataset(args.dataset, args.author_num)
    test_params = {"batch_size": args.batch_size, "shuffle": True, "drop_last": True}
    test_generator = DataLoader(test_dataset, **test_params)
    # test_dataset.train_flag = False

    if args.pre_trained is None:
        # bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)
        model = SyntaxTextCNN(args.text_out_dim,
                              args.syntax_out_dim,
                              args.text_embed_size,
                              args.syntax_embed_size,
                              test_dataset.char_len,
                              test_dataset.syntax_len,
                              150, [3, 4, 5],
                              args.author_num)
        if torch.cuda.is_available():
            model.cuda()
    else:
        model = torch.load(saved_path + args.pre_trained)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    max_accuracy = 0
    num_iter_per_epoch = len(test_generator)
    model.train()
    for i in range(args.epochs):
        for iter, (text_test, syntax_test, label_test) in enumerate(test_generator):
            if torch.cuda.is_available():
                text_test = text_test.cuda()
                syntax_test = syntax_test.cuda()
                label_test = label_test.cuda()

            optimizer.zero_grad()
            output = model(text_test, syntax_test)
            loss = criterion(output, label_test)
            loss.backward()
            optimizer.step()
            test_metrics = get_evaluation(label_test.cpu().numpy(), output.cpu().detach().numpy(), list_metrics=["accuracy"])
            print("Epoch: {}/{}, Iteration: {}/{}, Loss: {}, Accuracy: {}".format(
                i + 1, args.epochs, iter + 1, num_iter_per_epoch, loss, test_metrics["accuracy"]))

        model.eval()
        test_dataset.train_flag = False
        loss_ls, te_label_ls, te_pred_ls = [], [], []
        with torch.no_grad():
            for text_test, syntax_test, label_test in tqdm(test_generator):
                if torch.cuda.is_available():
                    text_test = text_test.cuda()
                    syntax_test = syntax_test.cuda()
                    label_test = label_test.cuda()

                output = model(text_test, syntax_test)
                te_label_ls.extend(label_test.clone().cpu())
                te_pred_ls.append(output.clone().cpu())
            te_pred = torch.cat(te_pred_ls, 0)
            te_label = np.array(te_label_ls)
            test_metrics = get_evaluation(te_label, te_pred.detach().numpy(), list_metrics=["accuracy"])
            print("Epoch: {}/{}, Accuracy: {}".format(i + 1, args.epochs, test_metrics["accuracy"]))

        if test_metrics["accuracy"] > max_accuracy:
            max_accuracy = test_metrics["accuracy"]
            torch.save(model, saved_path + 'model_' + str(i) + '.pkt')

        model.train()
        test_dataset.train_flag = True


if __name__ == '__main__':
    train()
