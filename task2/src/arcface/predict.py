import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

from .lightning import LightningModel
from .data import TestDataset


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--encoder", type=str, default="/notebook/temp/encoder.pkl"
    )
    parser.add_argument(
        "--test", type=str, default="/notebook/temp/Test/UNKNOWN"
    )
    parser.add_argument("--pred", type=str, default="preds.txt")
    return parser.parse_args()


def main(args: Namespace):
    test_dataset = TestDataset(args.test, args.encoder)
    test_loader = DataLoader(
        test_dataset, batch_size=2048, shuffle=False, num_workers=10
    )
    model = LightningModel.load_from_checkpoint(args.ckpt, map_location="cpu")
    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model.eval()
    model.to(device)

    preds = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            x = batch["image"]
            logits = model.infer(x.to(device))
            classes = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(classes)

    with open(args.pred, "w") as f_pred:
        for idx, pred in tqdm(
            enumerate(preds), total=len(preds), desc="File dump"
        ):
            name = test_dataset.get_name(idx)
            cls = test_dataset.get_class_by_idx(pred)
            print(name, cls, file=f_pred)


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
