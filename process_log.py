import re
import os
import pandas as pd

def extract_info_from_log(log_line, pattern):
    match = re.search(pattern, log_line)
    return match

def process_log(file_path, pattern):
    loss_data = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            info = extract_info_from_log(line, pattern)
            if info is not None:
                loss_data.append(float(info.group(1)))
            else:
                print(f"line {line_num} get no {pattern}")
                print(f"The line is {line}")
    
    return loss_data

def save_to_csv(save_path, *args):
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame()
    # align length
    for columns_name, data in args:
        if len(df) < len(data):
            num_features = len(df.columns)
            none_df = pd.DataFrame([[None] * num_features for _ in range(len(data) - len(df))]) 
            df = pd.concat([df, none_df])
        elif len(df) > len(data):
            data.extend([None] * (len(df) - len(data)))
        df[columns_name] = data
    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    opt_list = ["subscafsgd5", "subscafsgd_full", "fedavgsgd"]
    # process llm pretraining
    pretrain_pattern_dict = {"Loss": r"Loss: (\d+\.\d+)",
                    "Mem": r"Mem: (\d+\.\d+)",
                    "Throughput": r"Throughput_tokens: (\d+\.\d+)",
                    "lr": r"Lr: (\d+\.\d+)"}
    for opt in opt_list:
        log_path = f"logs/{opt}_pretrain.log"
        save_path = log_path.replace(".log", ".csv")
        save_path = save_path.replace("logs", "logs/csv")
        data_list = []
        for columns_name, pattern in pretrain_pattern_dict.items():
            data_list.append((columns_name, process_log(log_path, pattern)))
        save_to_csv(save_path, *data_list)
    
    # process llm fine-tune
    #finetune_eval_pattern = r"Eval Loss: (\d+\.\d+)"
    #eval_loss_data = []
    #for opt in opt_list:
        #log_path = f"logs/{opt}_finetune.log"
        #eval_loss_data.append((opt, process_log(f"{log_path}", finetune_eval_pattern)))
    #save_to_csv("logs/csv/finetune_eval_loss.csv", *eval_loss_data)

    #finetune_pattern_dict = {"Loss": r"Loss: (\d+\.\d+)",
                    #"Mem": r"Mem: (\d+\.\d+)",
                    #"lr": r"Lr: (\d+\.\d+)"}
    #for opt in opt_list:
        #log_path = f"logs/{opt}_finetune.log"
        #save_path = log_path.replace(".log", ".csv")
        #save_path = save_path.replace("logs", "logs/csv")
        #data_list = []
        #for columns_name, pattern in finetune_pattern_dict.items():
            #data_list.append((columns_name, process_log(log_path, pattern)))
        #save_to_csv(save_path, *data_list)
    

    # process resnet training
    #resnet_eval_pattern = r"\* Prec@5 (\d+\.\d+)" 
    #eval_loss_data = []
    #for opt in opt_list:
        #log_path = f"logs/{opt}_resnet_train.log"
        #eval_loss_data.append((opt, process_log(f"{log_path}", resnet_eval_pattern)))
    #save_to_csv("logs/csv/resnet_eval_acc.csv", *eval_loss_data)
    #resnet_pattern_dict = {
                           #"train_acc": r"Epoch.*Prec@5: (\d+\.\d+)",
                           #"loss": r"Loss: (\d+\.\d+)",
                        #}
    #for opt in opt_list:
        #log_path = f"logs/{opt}_resnet_train.log"
        #save_path = log_path.replace(".log", ".csv")
        #save_path = save_path.replace("logs", "logs/csv")
        #data_list = []
        #for columns_name, pattern in resnet_pattern_dict.items():
            #data_list.append((columns_name, process_log(log_path, pattern)))
        #save_to_csv(save_path, *data_list)




    