import argparse
import os
import warnings
import numpy as np
import torch
from transformers import AutoModel
from models.models import  MoE_model
from modules.Comparative_learning import  ImpressionContrastiveLoss
from modules.dataloaders import R2DataLoader
from modules.loss import  compute_loss_nlg
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler, count_parameters_by_module
from modules.tokenizers import Tokenizer
from modules.trainer import Trainer
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--ann_path', type=str)
    parser.add_argument('--bert_path',type=str,default='D:\Worktwo\MOE-RRG-main\BiomedVLP-CXR-BERT')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr', choices=['iu_xray', 'mimic_cxr'])

    parser.add_argument('--max_seq_length', type=int, default=100, help='报告的最大生成长度.')

    parser.add_argument('--threshold', type=int, default=0, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=4, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', choices=['resnet101', 'rad-dino'])
    parser.add_argument('--rad_dino_path', type=str, default='rad-dino')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int)
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.15, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # for Cross-modal Memory
    parser.add_argument('--topk', type=int, default=32, help='the number of k.')
    parser.add_argument('--cmm_size', type=int, default=2048, help='the numebr of cmm size.')
    parser.add_argument('--cmm_dim', type=int, default=512, help='the dimension of cmm dimension.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search',help='生成报告时的采样方法。可选值如 greedy（贪心）、beam_search（束搜索）等。')
    parser.add_argument('--beam_size', type=int, default=3,help='beam search时的束宽。决定每一步保留的候选序列数量，值越大越精确，但计算成本越高。')
    parser.add_argument('--temperature', type=float, default=1.0,help='采样温度系数。用于控制输出概率分布的平滑程度，值越小越确定性，值越大越随机。')
    parser.add_argument('--sample_n', type=int, default=1,help='每张图像采样的生成数量。如果设置为大于1，将生成多个不同版本的报告。')
    parser.add_argument('--group_size', type=int, default=1,help='用于多样性束搜索（diverse beam search）的组数。通常在需要提高生成多样性时使用。')
    parser.add_argument('--output_logsoftmax', type=int, default=1,help='是否输出 log softmax 概率。1 表示输出 log(prob)，0 表示输出未归一化的 logits。')
    parser.add_argument('--decoding_constraint', type=int, default=0,help='是否启用解码约束，防止生成重复词或非法模式。1 表示启用。')
    parser.add_argument('--block_trigrams', type=int, default=1,help='是否屏蔽重复的三元组（trigram）组合，以避免重复生成。1 表示启用。')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments.')
    parser.add_argument('--log_period', type=int, default=5, help='the logging interval (in batches).')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period (in epochs).')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='可以选Adam，或者AdamW.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=7e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.98), help='the weight decay.')
    parser.add_argument('--adam_eps', type=float, default=1e-9, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')
    parser.add_argument('--noamopt_warmup', type=int, default=5000, help='.')
    parser.add_argument('--noamopt_factor', type=int, default=1, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    args = parser.parse_args()


    if args.visual_extractor == "resnet101":
        args.d_vf = 2048
    elif  args.visual_extractor == "rad-dino":
        args.d_vf = 768
    if args.dataset_name == "mimic_cxr":
        args.image_dir = 'D:/Dataset/mimic_cxr/images/'
        args.ann_path = 'data/patched_annotations.json'
        # args.ann_path = 'data/mimic_test.json'
        args.save_dir = 'results/mimic_cxr'
        args.log_period = 5
    elif args.dataset_name == "iu_xray":
        args.image_dir = 'D:/Dataset/iu_xray/images/'
        #args.ann_path = 'data/iu_xray_annotation_disease.json'
        args.ann_path = 'D:/Dataset/iu_xray/annotation.json'
        args.save_dir = 'results/iu_xray'
    else:
        raise ValueError(f"不支持的数据集: {args.dataset_name}")
    return args


def main():
    # parse arguments
    args = parse_agrs()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_path, trust_remote_code=True, local_files_only=True)
    bert = AutoModel.from_pretrained(args.bert_path, trust_remote_code=True, local_files_only=True)
    bert.eval()
    for param in bert.parameters():
        param.requires_grad = False

    special_tokens = ["[INDICATION]", "[IMPRESSION]", "[PRIOR]", "[COMPARISON]", "[NHPR]"]
    bert_tokenizer.add_tokens(special_tokens, special_tokens=True)
    bert.resize_token_embeddings(len(bert_tokenizer))  # 扩展 embedding 层


    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, bert_tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, bert_tokenizer,split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, bert_tokenizer,split='test', shuffle=False)

    # build model architecture
    model = MoE_model(args, tokenizer, bert)

    count_parameters_by_module(model)
    # get function handles of loss and metrics
    c_loss_nlg = compute_loss_nlg
    c_loss_imp = ImpressionContrastiveLoss(bert_tokenizer=bert_tokenizer,bert=bert,tokenizer=tokenizer,tau=0.07)
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)
    print(optimizer)

    # build trainer and start to train
    trainer = Trainer(model, c_loss_nlg,c_loss_imp, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()
