import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset,DataLoader

def make_transition(data,batch_size,shuffle):
    if data[-3:]=='csv':
        df = pd.read_csv(data)
    elif data[-7:]=='parquet':
        df = pd.read_parquet(data)
    else:
        "Import csv or parquet data"
        break
    
    s_col = [x for x in df if x[:2]=='s:']
    a_col = [x for x in df if x[:2]=='a:']
    r_col = [x for x in df if x[:2]=='r:']
    dict = {}
    dict['traj'] = {}
    data_len = 0

    s,a,r,s2,t  = [],[],[],[],[]
    
    for traj in tqdm(df.traj.unique()):
        df_traj = df[df['traj'] == traj]
        dict['traj'][traj] = {'s':[],'a':[],'r':[]}
        dict['traj'][traj]['s'] = df_traj[s_col].values.tolist()
        dict['traj'][traj]['a'] = df_traj[a_col].values.tolist()
        dict['traj'][traj]['r'] = df_traj[r_col].values.tolist()

        step_len = len(df_traj) - 1
        for step in range(step_len):
            s.append(dict['traj'][traj]['s'][step])
            a.append(dict['traj'][traj]['a'][step])
            r.append(dict['traj'][traj]['r'][step+1])
            s2.append(dict['traj'][traj]['s'][step+1])
            t.append(0)
            data_len += 1
        s.append(dict['traj'][traj]['s'][step_len])
        a.append(dict['traj'][traj]['a'][step_len])
        r.append(dict['traj'][traj]['r'][step_len+1])
        s2.append(dict['traj'][traj]['s'][step_len+1])
        t.append(1)
        data_len += 1
    
    s  = torch.FloatTensor(np.float32(s))
    a  = torch.LongTensor(np.int64(a))
    r = torch.FloatTensor(np.float32(r))
    s2 = torch.FloatTensor(np.float32(s2))
    t  = torch.FloatTensor(np.float32(t))

    rt = DataLoader(TensorDataset(s, a, r, s2, t),batch_size,shuffle)
    return rt, data_len