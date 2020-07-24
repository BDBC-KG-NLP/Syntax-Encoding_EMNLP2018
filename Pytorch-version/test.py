import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from src.dataset import *

parser = argparse.ArgumentParser(
        "Implementation of the model: Syntax Encoding with Application in Authorship Attribution")
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--dataset", type=str, default="CCAT", choices=datasets)
parser.add_argument("--author_num", type=int, default="10")
args = parser.parse_args()


def test():
    test_dataset = SyntaxDataset(args.dataset, args.author_num)
    test_params = {"batch_size": args.batch_size, "shuffle": False, "drop_last": True}
    test_generator = DataLoader(test_dataset, **test_params)
    test_dataset.train_flag = False

    model = torch.load(saved_path + 'model.pkt')
    te_label_ls, te_pred_ls = [], []

    model.eval()
    with torch.no_grad():
        for text_test, syntax_test, label_test in tqdm(test_generator):
            output = model(text_test, syntax_test)
            te_label_ls.extend(label_test.clone().cpu())
            te_pred_ls.append(output.clone().cpu())
        te_pred = torch.cat(te_pred_ls, 0)
        te_label = np.array(te_label_ls)
        test_metrics = get_evaluation(te_label, te_pred.detach().numpy(), list_metrics=["accuracy"])
        print("Accuracy: {}".format(test_metrics["accuracy"]))


if __name__ == '__main__':
    test()
