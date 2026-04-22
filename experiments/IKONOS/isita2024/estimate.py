import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.special import logsumexp
from scipy.special import digamma
from scipy.special import beta
import scipy.stats as stats
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import os

import learning

##########################
#       定数の取得        #
##########################
WIDTH = int(learning.WIDTH)
HEIGHT = int(learning.HEIGHT)
MAX_DEPTH = int(learning.MAX_DEPTH)
COLOR = int(learning.COLOR)
N = int(learning.N) #学習データ数
Y_MIN = int(learning.Y_MIN)
Y_MAX = int(learning.Y_MAX)
num_label = int(learning.num_label)
LABEL_SET = [i for i in range(num_label)]

#########################
#      ファイルパス      #
#########################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_folder_path = os.path.join(BASE_DIR, 'test_data', 'images')
gt_folder_path = os.path.join(BASE_DIR, 'test_data', 'labels', 'visualize')
est_folder_path = os.path.join(BASE_DIR, 'test_data', 'labels', 'visualize', 'result')
dif_folder_path = os.path.join(BASE_DIR, 'test_data', 'labels', 'visualize', 'difference')


def _get_common_data_files(image_dir, label_dir):
    valid_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    image_files = {
        name for name in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, name)) and os.path.splitext(name)[1].lower() in valid_ext
    }
    label_files = {
        name for name in os.listdir(label_dir)
        if os.path.isfile(os.path.join(label_dir, name)) and os.path.splitext(name)[1].lower() in valid_ext
    }
    common = sorted(image_files & label_files)
    if not common:
        raise FileNotFoundError('No matching image/label file pairs were found in test_data.')
    return common


test_file_names = _get_common_data_files(image_folder_path, gt_folder_path)
M = len(test_file_names)
os.makedirs(est_folder_path, exist_ok=True)
os.makedirs(dif_folder_path, exist_ok=True)

#パラメータプラグイン
G = [1, 0.8333333333333334, 0.8229166666666667, 0.7333333333333332, 0.6142578124999999, 0.5044921875000001, 0.43749593098958006, 0]
probability = [[0.5, 0.5], [0.5, 0.5], [1.0, 0.0], [0.972972972972973, 0.02702702702702703], [0.7388059701492538, 0.26119402985074625], [0.5162393162393163, 0.48376068376068376], [0.4934445768772348, 0.5065554231227652], [0.5040041782729805, 0.4959958217270195]]
MEAN = [[210, 186, 139], [133, 129, 91]]
STD = [[32, 27, 22], [35, 27, 21]]



#############################
#      四分木のクラス       #
#############################

class Node_of_Full_Rooted_Tree:
    def __init__(self, upper_edge, lower_edge, left_edge, right_edge, depth):
        self.ul_node = None
        self.ur_node = None
        self.ll_node = None
        self.lr_node = None
        self.upper_edge = upper_edge
        self.lower_edge = lower_edge
        self.left_edge = left_edge
        self.right_edge = right_edge
        self.depth = depth
        self.g = 0    

class Node:
    def __init__(self, upper_edge, lower_edge, left_edge, right_edge, depth):
        self.ul_node = None
        self.ur_node = None
        self.ll_node = None
        self.lr_node = None
        self.upper_edge = upper_edge
        self.lower_edge = lower_edge
        self.left_edge = left_edge
        self.right_edge = right_edge
        self.depth = depth
        self.G = G
        self.sep_flag = False
        self.label = None
        self.logq = 0

    
    def pixel_ndarray(self):
        tmp = img_array[self.upper_edge: self.lower_edge, self.left_edge: self.right_edge]
        return tmp
        

    def pixel_array(self):
        tmp = img_array[self.upper_edge: self.lower_edge, self.left_edge: self.right_edge]
        Tmp = tmp.tolist()
        return Tmp
    
    def pixel_array_plusone(self):
        tmp = img_array[self.upper_edge: self.lower_edge, self.left_edge: self.right_edge]
        Tmp = tmp.tolist() 
        for row in range(len(Tmp)):
            for col in range(len(Tmp[row])):
                if COLOR == 1:
                    Tmp[row][col] += 1
                elif COLOR > 1:
                    for color in range(COLOR):
                        Tmp[row][col][color] += 1
        return Tmp


############################################
# 四分木構造のノードを再帰的に生成する関数 #
############################################

def make_full_tree(node):
    if node.depth < MAX_DEPTH:
        node.ul_node = Node_of_Full_Rooted_Tree(node.upper_edge,
                            (node.upper_edge+node.lower_edge)//2,
                            node.left_edge,
                            (node.left_edge+node.right_edge)//2,
                            node.depth+1)
        node.ur_node = Node_of_Full_Rooted_Tree(node.upper_edge,
                            (node.upper_edge+node.lower_edge)//2,
                            (node.left_edge+node.right_edge)//2,
                            node.right_edge,
                            node.depth+1)
        node.ll_node = Node_of_Full_Rooted_Tree((node.upper_edge+node.lower_edge)//2,
                            node.lower_edge,
                            node.left_edge,
                            (node.left_edge+node.right_edge)//2,
                            node.depth+1)
        node.lr_node = Node_of_Full_Rooted_Tree((node.upper_edge+node.lower_edge)//2,
                            node.lower_edge,
                            (node.left_edge+node.right_edge)//2,
                            node.right_edge,
                            node.depth+1)
        make_full_tree(node.ul_node)
        make_full_tree(node.ur_node)
        make_full_tree(node.ll_node)
        make_full_tree(node.lr_node)



def make_tree(node):
    if node.depth < MAX_DEPTH:
        node.ul_node = Node(node.upper_edge,
                            (node.upper_edge+node.lower_edge)//2,
                            node.left_edge,
                            (node.left_edge+node.right_edge)//2,
                            node.depth+1)
        node.ur_node = Node(node.upper_edge,
                            (node.upper_edge+node.lower_edge)//2,
                            (node.left_edge+node.right_edge)//2,
                            node.right_edge,
                            node.depth+1)
        node.ll_node = Node((node.upper_edge+node.lower_edge)//2,
                            node.lower_edge,
                            node.left_edge,
                            (node.left_edge+node.right_edge)//2,
                            node.depth+1)
        node.lr_node = Node((node.upper_edge+node.lower_edge)//2,
                            node.lower_edge,
                            (node.left_edge+node.right_edge)//2,
                            node.right_edge,
                            node.depth+1)
        make_tree(node.ul_node)
        make_tree(node.ur_node)
        make_tree(node.ll_node)
        make_tree(node.lr_node)

def print_tree(node):
    print(node.depth)
    print(node.upper_edge)
    print(node.lower_edge)
    print(node.left_edge)
    print(node.right_edge)
    print("")
    if node.ul_node != None:
        print_tree(node.ul_node)
    if node.ur_node != None:
        print_tree(node.ur_node)
    if node.ll_node != None:
        print_tree(node.ll_node)
    if node.lr_node != None:
        print_tree(node.lr_node)

####################################
#     確率値の自然対数を返す関数   #
####################################

#ラベルkのもとでノードs内のピクセルの画素値v_sが従う分布p(v_s|k)
def log_prob_v_of_k(node, label):
    tmp0 = norm.logpdf(node.pixel_array(), loc=MEAN[label], scale=STD[label])
    tmp1 = norm.cdf(node.pixel_array_plusone(), loc=MEAN[label], scale=STD[label])
    try:
        TMP = np.sum(np.log(tmp1 - tmp0))
    except RuntimeWarning:
        # ゼロ割の警告が発生した場合の処理をここに記述
        TMP = -np.inf
    return TMP

#ラベルk_dの事前分布
def log_prob_k(node, label):
    return np.log(probability[node.depth][label])

#ノードs内のピクセルの画素値v_sの，ラベルkに関する周辺分布p(v_s)
def log_prob_v(node):
    TMP = []
    for l in LABEL_SET:
        if MEAN[l] == [] or STD[l] == []:
            tmp == -np.inf
        else:
            tmp = log_prob_v_of_k(node, label=l) + log_prob_k(node, label=l)
        TMP.append(tmp)
    return logsumexp(TMP)

#ノードsごとのラベルの事後分布p(k|v_s)
def log_prob_k_of_v_s(node, label):
    return log_prob_v_of_k(node, label) + log_prob_k(node, label) - log_prob_v(node)


#ノードs内のピクセルの画素値v_sの，ラベルとハイパーパラメータに関する周辺尤度
def log_Q_v_s(node):
    if node.depth == MAX_DEPTH:
        return log_prob_v(node)
    else:
        if G[node.depth] ==1.0:
            return log_Q_v_s(node.ul_node) + log_Q_v_s(node.ur_node) + log_Q_v_s(node.ll_node) + log_Q_v_s(node.lr_node)
        elif G[node.depth] == 0:
            return log_prob_v(node)
        else:
            tmp1 = np.log(1-G[node.depth]) + log_prob_v(node)
            tmp2 = np.log(G[node.depth]) + log_Q_v_s(node.ul_node) + log_Q_v_s(node.ur_node) + log_Q_v_s(node.ll_node) + log_Q_v_s(node.lr_node)
            return logsumexp([tmp1, tmp2])
    

def calc_logq(node):
    if node.depth < MAX_DEPTH:
        calc_logq(node.ul_node)
        calc_logq(node.ur_node)
        calc_logq(node.ll_node)
        calc_logq(node.lr_node)
    node.logq = log_Q_v_s(node)

#ノードsにおける事後ハイパーパラメータg_{s|v}
def log_Posterior_G(node):
    if node.depth == MAX_DEPTH:
        return -np.inf
    else:
        tmp = np.log(G[node.depth]) + node.ul_node.logq + node.ur_node.logq + node.ll_node.logq + node.lr_node.logq - node.logq
        return tmp
    
##########################
#        MAP計算         #
##########################

def log_phi(node,label,i,j):
    if node.depth == MAX_DEPTH:
        return log_prob_k_of_v_s(node, label)
    elif i < (node.upper_edge + node.lower_edge)//2 and j < (node.left_edge + node.right_edge)//2:
        tmp0 = np.log(1-np.exp(log_Posterior_G(node))) + log_prob_k_of_v_s(node, label)
        tmp1 = log_Posterior_G(node) + log_phi(node.ul_node, label,i,j)
        return logsumexp([tmp0, tmp1])
    elif i < (node.upper_edge + node.lower_edge)//2 and j >= (node.left_edge + node.right_edge)//2:
        tmp0 = np.log(1-np.exp(log_Posterior_G(node))) + log_prob_k_of_v_s(node, label)
        tmp1 = log_Posterior_G(node) + log_phi(node.ur_node, label,i,j)
        return logsumexp([tmp0, tmp1])
    elif i >= (node.upper_edge + node.lower_edge)//2 and j < (node.left_edge + node.right_edge)//2:
        tmp0 = np.log(1-np.exp(log_Posterior_G(node))) + log_prob_k_of_v_s(node, label)
        tmp1 = log_Posterior_G(node) + log_phi(node.ll_node, label,i,j)
        return logsumexp([tmp0, tmp1])
    else:
        tmp0 = np.log(1-np.exp(log_Posterior_G(node))) + log_prob_k_of_v_s(node, label)
        tmp1 = log_Posterior_G(node) + log_phi(node.lr_node, label,i,j)
        return logsumexp([tmp0, tmp1])
    
def argmax_phi_of_label(node, i, j):
    list = []
    for ell in LABEL_SET:
        if MEAN[ell] == [] or STD[ell] == []:
            list.append(-np.inf)
        else: list.append(log_phi(node, ell, i, j))
    max_index = list.index(max(list))
    print(f"phi list={list}")
    print(f"label = {max_index}")
    return max_index

##########################
#        精度計算         #
##########################        
def calculate_metrics(ground_truth, predicted):
    # True Positive (TP), True Negative (TN), False Positive (FP), False Negative (FN) の初期化
    TP = TN = FP = FN = 0
    
    # 推定結果と正解データの比較
    for i in range(len(ground_truth)):
        for j in range(len(ground_truth[0])):
            if ground_truth[i,j] == 255 and predicted[i,j] == 255:
                TP += 1
            elif ground_truth[i,j] == 0 and predicted[i,j] == 0:
                TN += 1
            elif ground_truth[i,j] == 0 and predicted[i,j] == 255:
                FP += 1
            elif ground_truth[i,j] == 255 and predicted[i,j] == 0:
                FN += 1
                
    # Overall Accuracy を計算
    try:
        overall_accuracy = (TP + TN) / (TP + TN + FP + FN)
    except ZeroDivisionError:
        overall_accuracy = 1
    
    # Intersection over Union (IoU) を計算
    intersection = TP
    union = TP + FP + FN
    try:
        IoU = intersection / union
    except ZeroDivisionError:
        IoU = 1
    
    
    return TP, TN, FP, FN, overall_accuracy, IoU



##########################################
#               main                     #
##########################################

OA_Loss = []
IoU_Loss = []

evaluation_file_name = "evaluation_result.txt"
#with open(file_name, "w") as file:
    #file.truncate(0)

for test_file_name in test_file_names:

    stem = os.path.splitext(test_file_name)[0]

    img = Image.open(os.path.join(image_folder_path, test_file_name))
    img_array = np.array(img)
    print(test_file_name)
    #ラベルデータ格納行列
    region_img=np.zeros((HEIGHT, WIDTH))

    #Ground Truthマップ正解画像の読み込み
    gt_image = Image.open(os.path.join(gt_folder_path, test_file_name))
    gt_array = np.array(gt_image)

    #推定したモデルの葉ノードに対して計算されるラベルの事後確率の配列の定義([深さ][ラベル])
    log_prob_label_of_depth = [[0, 0] for _ in range(MAX_DEPTH)]

    root = Node(0, HEIGHT, 0 ,WIDTH,0)

    make_tree(root)
    
    calc_logq(root)

    for i in range(0, HEIGHT):
        for j in range(0, WIDTH):
            region_img[i][j] = (1-argmax_phi_of_label(root, i, j)) * 255
            print(f"({i},{j})={region_img[i][j]}")
            print(gt_array[i][j])
    # 領域マップの画像を作成
    region_img = Image.fromarray(region_img.astype(np.uint8), mode='L')

    # 画像を保存
    result_file_name = f"result_{stem}.png"
    region_img.save(os.path.join(est_folder_path, result_file_name))


    # 領域マップ推定結果画像の読み込み
    result_image = Image.open(os.path.join(est_folder_path, result_file_name))
    result_array = np.array(result_image)


    #精度計算
    TP, TN, FP, FN, OA, IoU = calculate_metrics(gt_array, result_array)

    OA_Loss.append(1-OA)
    IoU_Loss.append(1-IoU)

    print(test_file_name)
    print("OA_Loss = "+str(1-OA))
    print("IoU_Loss = "+str(1-IoU))

    with open(evaluation_file_name, "a") as file:
        file.write(test_file_name+"\n")
        file.write("OA_Loss = "+str(1-OA)+"\n")
        file.write("IoU_Loss = "+str(1-IoU)+"\n")
 

    # ハイライト画像の作成
    correct_mask = (gt_array == result_array)
    highlight_img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    highlight_img[correct_mask] = [0, 0, 255]  # 正解した部分を青色に
    highlight_img[np.logical_not(correct_mask)] = [255, 192, 203]  # 不正解の部分を桃色に

    # 画像の保存
    highlight_image = Image.fromarray(highlight_img)
    highlight_image.save(os.path.join(dif_folder_path, f"difference_{stem}.png"))
    

print("OA_Loss:"+str(OA_Loss))
print("average OA_loss: "+str(np.sum(OA_Loss)/len(OA_Loss)))

print("IoU_Loss:"+str(IoU_Loss))
print("average IoU_loss: "+str(np.sum(IoU_Loss)/len(IoU_Loss)))















