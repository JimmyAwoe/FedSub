import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


sgd_finetune = pd.read_csv("logs/sgd_finetune.csv")
subscaf_finetune = pd.read_csv("logs/subscaf_finetune.csv")
sgd_pretrain = pd.read_csv("logs/sgd_pretrain.csv")
subscaf_pretrain = pd.read_csv("logs/subscafsgd_pretrain.csv")
eval_loss = pd.read_csv("logs/eval_loss.csv")


# pretrain_loss 
pretrain_loss = pd.concat([subscaf_pretrain['Loss'], sgd_pretrain['Loss']], axis=1)
pretrain_columns = ["Our-CD", "SGD", ]
pretrain_loss.columns = pretrain_columns            

# finetune loss 
finetune_loss = pd.concat([subscaf_finetune['Loss'], sgd_finetune['Loss']], axis=1)
finetune_columns = ["Our-CD", "SGD", ]
finetune_loss.columns = finetune_columns            

# eval loss
eval_columns = ["Our-CD", "SGD"]
eval_loss.columns = eval_columns

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

def plot_assigned_col(df, columns, output_path, title, start_point=0, end_point=None, if_ewm=True):
    plt.figure(figsize=(6, 4))
    #plt.title(title)
    plt.xlabel('Communication Rounds')
    plt.ylabel('Error')
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
    for name in columns:
        if if_ewm:
            plt.plot(x, df[name], linewidth=1, alpha=0.4)
            color = plt.gca().lines[-1].get_color()
            ewm = pd.Series(df[name]).ewm(alpha=0.05).mean()
            print(name, ':', ewm.iat[-1])
            plt.plot(x, ewm, label=name, linewidth=2, color=color)
            plt.axhline(y=ewm.iat[-1], color=color, linestyle='--', alpha=0.4)
        else:
            plt.plot([1000, 2000, 3000, 4000, 4590], df[name], linewidth=2, label=name)
            color = plt.gca().lines[-1].get_color()
            print(name, ':', df[name].iat[-1])
            plt.axhline(y=df[name].iat[-1], color=color, linestyle='--', alpha=0.4)
    plt.legend()
    plt.savefig(output_path)
    plt.show()
    plt.close()

plot_assigned_col(
    pretrain_loss,
    pretrain_columns,
    'figures/pretrain_loss.png',
    'Pretrain Loss'
)


plot_assigned_col(
    finetune_loss,
    finetune_columns,
    'figures/finetune_loss.png',
    'Finetune Loss'
)

plot_assigned_col(
    eval_loss,
    eval_columns,
    'figures/Eval_loss.png',
    'Eval Loss',
    if_ewm=False
)