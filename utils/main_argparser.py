import argparse

def main_parse_args(args):
    parser = argparse.ArgumentParser()

    # Training 
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", default=16, type=int, help="batch size per round")
    parser.add_argument("--total_batch_size", default=32, type=int, help="batch size per step")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_training_steps", type=int, default=10000)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--mixed_precision", default=None, type=str, choices=['bf16', 'fp16'])

    # subscaf
    parser.add_argument("--comp_dim", default=64, type=int, help="the compression dimension")
    parser.add_argument("--tau", type=int, help="inner loop steps")
    parser.add_argument("--gene_method", default='cd', type=str, 
                        help="set the method to generate compression matrix")
    parser.add_argument("--jump_certain_modules", action="store_true")
    parser.add_argument("--update_cp_freq", type=int, default=1, help="inner loop steps")
    parser.add_argument("--adaptive_cp_rate", type=float, default=0.0)

    # model
    parser.add_argument("--model_config", type=str, default="configs/llama_60m.json")

    # optimizer
    parser.add_argument("--optimizer", choices=['subscafsgd', 'subscafadam', 'scaf', 'sgd', 'adam'], default='subscafsgd',
                        type=str, help="assign the optimization algorithm")
    parser.add_argument("--momentum", default=0, type=float)
    parser.add_argument("--dampening", default=0, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--per_layer_weight_update", action="store_true")

    # wandb
    parser.add_argument("--wandb_run_name", default='subscaf_sgd')

    # memory monitor
    parser.add_argument("--mem_monitor", action="store_true")

    args, unknown_args = parser.parse_known_args()
    #args = parser.parse_args(args)

    return args, unknown_args