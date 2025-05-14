import numpy as np
from scipy.stats import spearmanr
import math

def cosine_similarity(dict1, dict2):
    list1 = list(dict1.values())
    list2 = list(dict2.values())
    
    return np.dot(list1, list2) / (np.linalg.norm(list1) * np.linalg.norm(list2))

def max_difference(dict1, dict2):
    list1 = list(dict1.values())
    list2 = list(dict2.values())
    max = 0
    for i in range(len(list1)):
        if abs(list1[i]-list2[i]) > max:
            max = abs(list1[i]-list2[i])
    return max


Exact_Shapley = np.array([1, 5, 9, 2, 8, 6, 0, 4, 7, 3])

# TMC_Shapley = np.array([9, 5, 8, 4, 2, 6, 1, 7, 3, 0])

# MAB_Shapley = np.array([7, 6, 9, 1, 4, 2, 3, 0, 8, 5])

GTG_Shapley = np.array([2, 0, 8, 9, 1, 6, 5, 3, 4, 7])

# Delta_Shapley = np.array([8, 5, 6, 9, 0, 3, 4, 2, 1, 7])

CS_Shapley = np.array([1, 2, 9, 5, 6, 8, 7, 4, 0, 3])

 
# 计算斯皮尔曼等级相关性
# coef, p_value = spearmanr(Exact_Shapley, MAB_Shapley) # 0.5515151515151515
# coef, p_value = spearmanr(Exact_Shapley, GTG_Shapley) # 1
coef, p_value = spearmanr(Exact_Shapley, CS_Shapley)
 
print(f"斯皮尔曼等级相关系数: {coef}")
# print(f"p值: {p_value}")

Exact_Shapley_Value = {0: 0.16125465948134657,	1: -0.22824290249023632,	2: -0.07657180027888409,	3: 0.40070392158296364	,4: 0.17497986288415995,	5: -0.15591372784286261,	6: 0.051902265140106776,	7: 0.21205157940941186,	8: 0.00030745243624089383,	9: -0.09567132831092873

}

# Delta_Shapley_Value = {0: 9.020998001098631, 1: 26.661592344443005, 2: 15.947496573130296, 3: 13.56062835454941, 4: 14.308996081352237, 5: -4.464767952760058, 6: -1.3978689114252736, 7: 36.53915731112163, 8: -5.385231991608935, 9: 1.4365010460217797}

# TMC_Shapley_Value = {0: 2.6675816747546204,	1: 2.3853823201855024,	2: 2.4439755974213275,	3: 2.395910966495672,	4: 2.4621516346931465,	5: 3.227918646335602,	6: 2.4477353355288503,	7: 2.4507206476728123,	8: 2.266778006652991,	9: 2.3787773587306345	}

# MAB_Shapley_Value = {0: 2.4355876989780905,	1: 2.3244550833862925,	2: 2.4107831775196016,	3: 2.4141786305204267,	4: 2.388204763853361,	5: 2.4748421450219453,	6: 2.2957111083089354,	7: 2.278295715697228,	8: 2.4404105085938697,	9: 2.308606750293384	


# }  

GTG_Shapley_Value = {0: -0.05523000136017799,	1: 0.08850666706760726,	2: -0.35983998055259386,	3: 0.12494663447141649,	4: 0.1537366772691409,	5: 0.10149000659585,	6: 0.09467667167385417,	7: 0.1937566618124644,	8: 0.034669978668292356,	9: 0.055716676513353984	



}

CS_Shapley_Value = {0: 0.0939766749739647,	1: -0.06914001454909642,	2: -0.04678999707102776,	3: 0.2873766387502352,	4: 0.055406641711791355	,5: -0.010523333152135207,	6: 0.03595331211884816,	7: 0.05416665722926458,	8: 0.0404433568318685,	9: -0.04390335629383723	




}
# 计算余弦相似度
# similarity = cosine_similarity(Exact_Shapley_Value, GTG_Shapley_Value) # 0.9997343798372547
# similarity = cosine_similarity(Exact_Shapley_Value, MAB_Shapley_Value) # 0.9867514726642874   
similarity = cosine_similarity(Exact_Shapley_Value, CS_Shapley_Value) # 
print(f"cos_distance: {similarity}")

# max_diff = max_difference(Exact_Shapley_Value, GTG_Shapley_Value) # 0.07459421109219178
max_diff = max_difference(Exact_Shapley_Value, CS_Shapley_Value) # 1.6448773020848861
# max_diff = max_difference(Exact_Shapley_Value, Delta_Shapley_Value) # 
print(f"max_distance: {max_diff}")