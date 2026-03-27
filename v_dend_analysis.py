import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

i = 0
with open(f'output/valid/20260117-192253-dend_sdt-data-cifar10-dvs-t-10-spike-lif/qkv_{i}.pkl', 'rb') as f:
    data = torch.load(f,map_location='cpu')

#print(type(data))
for k in data.keys():
    print(f"key:{k}, shape: {data[k].shape}")

#print(len(data))
t_count = torch.zeros((10,))

q_lif = data["MS_SSA_dend_Conv1_v_lif"]
for t in range(10):
    count_0,count_1,count_2,count_3,count_4 = 0,0,0,0,0
    for i in range(16):
        vec_c = q_lif[t][i][:][0][0]
        count_0 += (q_lif[t][i]==0).sum().item()
        count_1 += (q_lif[t][i]==1).sum().item()
        count_2 += (q_lif[t][i]==2).sum().item()
        count_3 += (q_lif[t][i]==3).sum().item()
        count_4 += (q_lif[t][i]==4).sum().item()

        for c in vec_c:
            if c==4:
                t_count[t]+=1

    
    print(count_0,count_1,count_2,count_3,count_4)

print(t_count)

for i in range(16):
    for t in range(10):
        c0,c1,c2,c3,c4 = 0,0,0,0,0
        vec_c = q_lif[t][i][:][0][0]
        for c in vec_c:
            if c==0:
                c0+=1
            if c==1:
                c1+=1
            if c==2:
                c2+=1  
            if c==3:
                c3+=1
            if c==4:
                c4+=1              
