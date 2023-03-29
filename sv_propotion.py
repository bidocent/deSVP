import numpy as np
from scipy.sparse import coo_matrix
import cooler
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
from scipy.stats import pareto
from scipy.stats import entropy
from scipy.optimize import minimize
import math

k562_sv='''chr11	chr11	5312156	5443877	+-
chr4	chr4	159570188	162695548	+-
chr13	chr9	108009063	131280137	++
chr22	chr9	23290555	130731760	+-
chr10	chr10	101843385	102036376	+-
chr14	chr14	22740100	23338749	-+
chr10	chr3	86089576	48186689	++
chr22	chr22	18965610	22592903	--
chr10	chr17	38919525	26938352	--
chr13	chr13	93371155	107848624	+-
chr14	chr14	96558591	97172429	-+
chr9	chr9	120790471	128341072	-+
chr18	chr18	8110641	23730829	-+
chr9	chr9	120793531	129891596	-+
chr5	chr5	163794106	164558902	-+
chr13	chr13	80535279	89786452	+-
chr7	chr7	79671143	80139199	-+
chr9	chr9	26589490	38430029	-+
chr1	chr1	186318948	187071977	-+
chr13	chr13	80896023	89786346	+-
chr13	chr13	92293810	93200560	--
chr3	chr3	60621153	60767117	+-
chr22	chr9	16819349	131199197	++
chr1	chr18	54726340	25907721	-+
chr5	chr6	51100047	37855743	++
chr17	chr17	28573108	28945866	-+
chr8	chr8	39374555	39529709	+-
chr7	chr7	36282545	36701472	-+
chr18	chr18	3478987	10860074	++
chr14	chr14	66304882	66445742	+-
chr6	chr6	31651286	31865938	-+
chr18	chr18	479945	21925996	+-
chr18	chr18	7481226	25924756	--
chr18	chr18	39431142	39624601	+-
chr15	chr15	76415875	76673167	+-
chr6	chr6	160986406	161246613	-+
chr1	chr1	172184312	172646532	-+
chr13	chr13	80513898	80895800	-+
chr12	chr21	22558289	24132644	--
chr10	chr10	50358858	50524668	+-
chr1	chr1	210154741	210649176	+-
chr6	chr6	16774241	51870915	++
chr9	chr9	128651698	129668798	--
chr7	chr7	111476850	111773294	-+'''
k562_sv = [i.split('\t') for i in k562_sv.split('\n')]
k562_sv_filter=[]
for i in k562_sv:
    if i[0]==i[1] :
        i[3] = int(i[3])
        i[2] = int(i[2])
        k562_sv_filter.append(i)
k562_sv_filter = sorted(k562_sv_filter,key = lambda x:x[0])
k562_sv_filter


def get_four_directions(chr_matrix,sv,patch_size):
    #从左取右时，右边端点为结构变异断点距离要+1！！！不然取不到结构变异断点哦！！
    if sv[2]-patch_size<0 :
        top_left = chr_matrix[0:sv[2],sv[3]-patch_size:sv[3]]
        top_right = chr_matrix[0:sv[2],sv[3]:sv[3]+patch_size]
        down_left = chr_matrix[sv[2]:sv[2]+patch_size,sv[3]-patch_size:sv[3]]
        down_right = chr_matrix[sv[2]:sv[2]+patch_size,sv[3]:sv[3]+patch_size]
        
    elif sv[3]+patch_size > chr_matrix.shape[0]:
        top_left = chr_matrix[sv[2]-patch_size:sv[2],sv[3]-patch_size:sv[3]]
        top_right = chr_matrix[sv[2]-patch_size:sv[2],sv[3]:chr_matrix.shape[0]]
        down_left = chr_matrix[sv[2]:sv[2]+patch_size,sv[3]-patch_size+1:sv[3]+1]
        down_right = chr_matrix[sv[2]:sv[2]+patch_size,sv[3]:chr_matrix.shape[0]]
        
    else:
        top_left = chr_matrix[sv[2]-patch_size:sv[2],sv[3]-patch_size:sv[3]]
        top_right = chr_matrix[sv[2]-patch_size:sv[2],sv[3]:sv[3]+patch_size]
        down_left = chr_matrix[sv[2]:sv[2]+patch_size,sv[3]-patch_size:sv[3]]
        down_right = chr_matrix[sv[2]:sv[2]+patch_size,sv[3]:sv[3]+patch_size]
        
    return top_left,top_right,down_left,down_right

def calculate_P_e(con_matrix,chr_con_sum,sv,patch_size):
    # P_expected
    chr_lenth = con_matrix.shape[0]
    random_sites = np.random.randint(3*patch_size,chr_lenth-3*patch_size,50)
    background_matrixs=[]
    background_results=[]
    chr_con_sum = np.sum(con_matrix)
    for i in random_sites:
        random_matrix = con_matrix[i-patch_size:i,i:i+patch_size]
        if np.sum(random_matrix)==0:
            continue
        random_matrix = random_matrix/chr_con_sum
        diag_index = np.int32(np.linspace(-patch_size+1,patch_size-1,2*patch_size-1))
        diags = map(lambda x: np.diag(random_matrix,x),diag_index)
        diags = np.array([np.mean(i) for i in diags])
        background_matrixs.append(random_matrix)
        background_results.append(diags)
    expected_matrix = np.mean(np.array(background_matrixs),axis=0)
    P_e = np.mean(np.array(background_results),axis=0)
    return P_e

#P_sv
def calculate_P_sv(target_matrix,chr_exp_sum,sv,patch_size,):
    target = target_matrix/chr_exp_sum
    diag_index = np.int32(np.linspace(-patch_size+1,patch_size-1,2*patch_size-1))
    P_sv = map(lambda x: np.diag(target,x),diag_index)
    P_sv = np.array([np.mean(i) for i in P_sv])
    return P_sv

# P_d
def calculate_P_d(target_matrix,chr_con_sum,sv,patch_size):
    target = target_matrix/chr_con_sum
    diag_index = np.int32(np.linspace(-patch_size+1,patch_size-1,2*patch_size-1))
    P_d = map(lambda x: np.diag(target,x),diag_index)
    P_d = np.array([np.mean(i) for i in P_d])
    return P_d

def fun(args):
    P_sv,P_e,P_d = args
    return lambda x: np.sum(np.abs(P_sv-(x*P_e + (1-x)*P_d)))*1e7

def optim(P_sv,P_e,P_d):
    e = 1e-12
    cons=(
        {'type':'ineq','fun':lambda x: x-e},
        {'type':'ineq','fun':lambda x: 1-x-e},
    )
    args = (P_sv,P_e,P_d)  
    x0 = 0.01
    res = minimize(fun(args), x0,constraints=cons)
    print(res.message)
    return res

binsize=10000
con = cooler.Cooler(f'five_merge_multi.mcool::/resolutions/{binsize}')
exp = cooler.Cooler(f'4DNFI18UHVRO_K562.mcool::/resolutions/{binsize}')

tmp = []
results=[]
for sv in tqdm(k562_sv_filter):
    sv[2],sv[3] = math.floor(sv[2]/binsize),math.floor(sv[3]/binsize)
    patch_size = 80
    
    exp_matrix = exp.matrix(balance=False).fetch(sv[0])
    con_matrix = con.matrix(balance=False).fetch(sv[0])

    top_left_exp,top_right_exp,down_left_exp,down_right_exp = get_four_directions(exp_matrix,sv,patch_size)
    top_left_con,top_right_con,down_left_con,down_right_con = get_four_directions(con_matrix,sv,patch_size)

    chr_exp_sum = np.sum(exp_matrix)
    chr_con_sum = np.sum(con_matrix)

    conditions = [sv[4]=='+-',sv[4]=='--',sv[4]=='++',sv[4]=='-+']
    if conditions[0]:
        target_exp = top_right_exp
        target_con = top_right_con
    elif conditions[1]:
        target_exp = np.rot90(down_right_exp)
        target_con = np.rot90(down_right_con)
    elif conditions[2]:
        target_exp = np.rot90(top_left_exp,k=3)
        target_con = np.rot90(top_left_con,k=3)
    elif conditions[3]:
        target_exp = down_left_exp.T
        target_con = down_left_con.T
    print(np.sum(target_exp))
    
    P_e = calculate_P_e(con_matrix,chr_con_sum,sv,patch_size)
    P_sv = calculate_P_sv(target_exp,chr_exp_sum,sv,patch_size,)
    P_d = calculate_P_d(target_con,chr_con_sum,sv,patch_size,)
    
    res = optim(P_sv,P_e,P_d)

    results.append(res)
    print(res.x)
    tmp.append([P_e,P_sv,P_d])
   
X=[]
for i in results:
    X.append(i.x)
fig, ax = plt.subplots(figsize=(8, 6),dpi=300)
ax.bar( np.arange(36),np.array(X).reshape(-1),  color='steelblue')

ax.set_title('K562 SVs Proportion Prediction', fontsize=18)
ax.set_xlabel('SVs', fontsize=14)
ax.set_ylabel('Proportion', fontsize=14)
ax.savefig("your sv propotion.png",dpi=100)