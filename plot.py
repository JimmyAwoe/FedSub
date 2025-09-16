import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


fedavgsgd_finetune = pd.read_csv("logs/csv/fedavgsgd_finetune.csv")
subscafsgd_finetune = pd.read_csv("logs/csv/subscafsgd_finetune.csv")
subscafsgd_full_finetune = pd.read_csv("logs/csv/subscafsgd_full_finetune.csv")
fedavgsgd_pretrain = pd.read_csv("logs/csv/fedavgsgd_pretrain.csv")
subscafsgd_pretrain = pd.read_csv("logs/csv/subscafsgd5_pretrain.csv")
subscafsgd_full_pretrain = pd.read_csv("logs/csv/subscafsgd_full_pretrain.csv")
finetune_eval_loss = pd.read_csv("logs/csv/finetune_eval_loss.csv")
subscafsgd_resnet = pd.read_csv("logs/csv/subscafsgd_resnet_train.csv", usecols=[0,1,2])
subscafsgd_full_resnet = pd.read_csv("logs/csv/subscafsgd_full_resnet_train.csv", usecols=[0,1,2])
fedavgsgd_resnet = pd.read_csv("logs/csv/fedavgsgd_resnet_train.csv", usecols=[0,1,2])
resnet_eval_acc = pd.read_csv("logs/csv/resnet_eval_acc.csv", usecols=[0,1,2])


# pretrain_loss 
pretrain_loss = pd.concat([subscafsgd_pretrain['Loss'], subscafsgd_full_pretrain['Loss'], fedavgsgd_pretrain['Loss']], axis=1)
pretrain_columns = ["Our-CD", r"Our-P$^k$=I", "FedAvg" ]
pretrain_loss.columns = pretrain_columns            

# finetune loss 
finetune_loss = pd.concat([subscafsgd_finetune['Loss'], subscafsgd_full_finetune["Loss"], fedavgsgd_finetune['Loss']], axis=1)
finetune_columns = ["Our-CD", r"Our-P$^k$=I", "FedAvg"]
finetune_loss.columns = finetune_columns            

# fine_tune eval loss
eval_columns = ["Our-CD", r"Our-P$^k$=I", "FedAvg"]
finetune_eval_loss.columns = eval_columns

# resnet train acc
resnet_train_loss = pd.concat([subscafsgd_resnet['train_acc'], subscafsgd_full_resnet['train_acc'], fedavgsgd_resnet['train_acc']], axis=1)
resnet_train_columns = ["Our-CD", r"Our-P$^k$=I", "FedAvg" ]
resnet_train_loss.columns = resnet_train_columns            

# resnet eval acc
resnet_eval_columns = ["Our-CD", r"Our-P$^k$=I", "FedAvg" ]
resnet_eval_acc.columns = resnet_eval_columns   

font_path = '/usr/local/share/fonts/WindowsFonts/simhei.ttf'
font_prop = fm.FontProperties(fname=font_path)
#plt.rcParams['font.family'] = font_prop.get_name()

def intersect(df, col1, col2, start_point=0, end_point=None):
    columns = [col1, col2]
    df = df[columns].dropna(how="any")
    if end_point == None and start_point == 0:
        x = np.arange(len(df[columns]))
    elif end_point != None and start_point == 0:
        x = np.arange(end_point)
    elif end_point == None and start_point != 0:
        x = np.arange(start_point, len(df))
    elif end_point != None and start_point != 0:
        x = np.arange(start_point, end_point)
    df = df.iloc[x]
    ewm1 = pd.Series(df[col1]).ewm(alpha=0.05).mean()
    ewm2 = pd.Series(df[col2]).ewm(alpha=0.05).mean()
    sign = (ewm1 - ewm2).abs()
    sign_step = sign[sign == sign.min()].index
    return sign_step

def plot_assigned_col_linear(df, columns, output_path, title, start_point=0, end_point=None, if_ewm=True, assigned_x=None, xl=None, yl=None):
    plt.figure(figsize=(6, 4))
    #plt.title(title)
    plt.xlabel('Communication Rounds', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    if xl is not None:
        plt.xlabel(xl)
    if yl is not None:
        plt.ylabel(yl)

    #plt.yscale('log')
    plt.grid(True)
    df = df[columns].dropna(how="any")
    if end_point == None and start_point == 0:
        x = np.arange(len(df[columns]))
    elif end_point != None and start_point == 0:
        x = np.arange(end_point)
    elif end_point == None and start_point != 0:
        x = np.arange(start_point, len(df))
    elif end_point != None and start_point != 0:
        x = np.arange(start_point, end_point)
    df = df.iloc[x]
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c']
    i=0
    for name in columns:
        if if_ewm:
            plt.plot(x, df[name], linewidth=1, alpha=0.0, color=color_list[i])
            i+=1
            color = plt.gca().lines[-1].get_color()
            ewm = pd.Series(df[name]).ewm(alpha=0.1).mean()
            print(name, ':', ewm.iat[-1])
            plt.plot(x, ewm, label=name, linewidth=1.5, color=color, alpha=1.0)
            plt.axhline(y=ewm.iat[-1], color=color, linestyle='--', alpha=0.4)
        else:
            if assigned_x is None:
                assigned_x = x
            plt.plot(assigned_x, df[name], linewidth=2, label=name)
            color = plt.gca().lines[-1].get_color()
            print(name, ':', df[name].iat[-1])
            plt.axhline(y=df[name].iat[-1], color=color, linestyle='--', alpha=0.4)
    plt.legend()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()


#plot_assigned_col_linear(
    #pretrain_loss.iloc[::10, :],
    #pretrain_columns,
    #'figures/pretrain_loss.pdf',
    #'Pretrain Loss',
    #start_point=0,
#)

#plot_assigned_col_linear(
    #finetune_loss.iloc[::10,:],
    #finetune_columns,
    #'figures/finetune_loss.pdf',
    #'Training Loss',
#)

plot_assigned_col_linear(
    finetune_eval_loss,
    eval_columns,
    'figures/finetune_Eval_loss.pdf',
    'Eval Loss',
    if_ewm=False,
    assigned_x=[100, 200, 230]
)

#plot_assigned_col_linear(
    #resnet_train_loss,
    #resnet_train_columns,
    #'figures/resnet_train_acc.pdf',
    #'Train acc',
    #if_ewm=False,
    #xl="Steps",
    #yl='Accuracy',
#)

#plot_assigned_col_linear(
    #resnet_eval_acc,
    #resnet_eval_columns,
    #'figures/resnet_eval_acc.pdf',
    #'Resnet Eval Loss',
    #if_ewm=False,
    #assigned_x=[390 * i for i in range(12)],
    #xl="Steps",
    #yl='Accuracy',
#)