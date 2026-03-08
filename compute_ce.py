import pandas as pd
from pprint import pprint
from pycocoevalcap.bleu.bleu import Bleu
from modules.metrics import compute_mlc


def eval_bleu(gts_csv, res_csv):

    gts = pd.read_csv(gts_csv, header=None, dtype=str)[0].fillna("").tolist()
    res = pd.read_csv(res_csv, header=None, dtype=str)[0].fillna("").tolist()
    assert len(res) == len(gts)

    refs = {i: [g.strip()] for i, g in enumerate(gts)}
    hyps = {i: [r.strip()] for i, r in enumerate(res)}

    bleu = Bleu(4)
    overall, _ = bleu.compute_score(refs, hyps, verbose=0)
    #  ↑ overall 是 [BLEU-1, BLEU-2, BLEU-3, BLEU-4]

    return dict(zip(["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"], overall))

def main():
    res_path = "results/mimic_cxr/res_labeled.csv"
    gts_path = "results/mimic_cxr/gts_labeled.csv"
    res_data, gts_data = pd.read_csv(res_path), pd.read_csv(gts_path)
    res_data, gts_data = res_data.fillna(0), gts_data.fillna(0)

    label_set = res_data.columns[1:].tolist()
    res_data, gts_data = res_data.iloc[:, 1:].to_numpy(), gts_data.iloc[:, 1:].to_numpy()
    res_data[res_data == -1] = 0
    gts_data[gts_data == -1] = 0

    metrics = compute_mlc(gts_data, res_data, label_set)
    pprint(metrics)


if __name__ == '__main__':
    main()


