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
import sys

##############################
#   学習データのファイルパス   #
##############################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_folder_path = os.path.join(BASE_DIR, 'train_data', 'images')
gt_folder_path = os.path.join(BASE_DIR, 'train_data', 'labels', 'visualize')


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
        raise FileNotFoundError('No matching image/label file pairs were found in train_data.')
    return common


train_file_names = _get_common_data_files(image_folder_path, gt_folder_path)

##########################
#       定数の取得        #
##########################
#       定数の取得        #
##########################
#1枚の画像を取得
pr_img = Image.open(os.path.join(image_folder_path, train_file_names[0]))

#画像サイズ
WIDTH, HEIGHT = pr_img.size
WIDTH = int(WIDTH)
HEIGHT = int(HEIGHT)
MAX_DEPTH = int(np.log2(WIDTH))

#学習データ数
N = len(train_file_names)

#画像の種類（grayscale: 1, RGB:3）
COLOR = 3

#画素値の範囲
Y_MIN = 0
Y_MAX = 255

#ラベル数
num_label = 2

# ラベル集合
label_set = [i for i in range(num_label)]

#ラベル行列の取りうる値の集合
label_value_set = [0,255]



#############################
#      四分木のクラス       #
#############################

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
        self.label = None
        self.a = 1.0
        self.b = 1.0
        self.split_count = 1
        self.nonsplit_count = 1


    def pixel_ndarray(self, number):
        tmp = img_array[number][self.upper_edge:self.lower_edge, self.left_edge: self.right_edge]
        return tmp
        

    def pixel_array(self, number):
        tmp = img_array[number][self.upper_edge: self.lower_edge, self.left_edge: self.right_edge]
        Tmp = tmp.tolist()
        return Tmp
    
    def pixel_array_plusone(self, number):
        tmp = img_array[number][self.upper_edge: self.lower_edge, self.left_edge: self.right_edge]
        Tmp = tmp.tolist() 
        for row in range(len(Tmp)):
            for col in range(len(Tmp[row])):
                Tmp[row][col] += 1
        return Tmp
    
    def label_ndarray(self,number):
        tmp = label_array[number][self.upper_edge: self.lower_edge, self.left_edge: self.right_edge]
        return tmp
    
    def label_array(self,number):
        tmp = img_array[number][self.upper_edge: self.lower_edge, self.left_edge: self.right_edge]
        Tmp = tmp.tolist()
        return Tmp

############################################
# 四分木構造のノードを再帰的に生成する関数 #
############################################

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




####################################
#     パラメータ推定のための関数　   #
####################################

def is_matrix_all_same(matrix):
    # 行列内のすべての値が同じかどうかを確認
    return (matrix == matrix[0]).all()

count_label_per_depth = [[0 for J in range(num_label)] for I in range(MAX_DEPTH+1)]
def recursive_split(node, image_num):
    if (is_matrix_all_same(node.label_ndarray(image_num))==False) and (node.depth < MAX_DEPTH):
        node.split_count += 1
        recursive_split(node.ul_node, image_num)
        recursive_split(node.ur_node, image_num)
        recursive_split(node.ll_node, image_num)
        recursive_split(node.lr_node, image_num)
    else:
        node.nonsplit_count += 1
        for label_value in label_value_set:
            if label_value == node.label_ndarray(image_num)[0][0]:
                count_label_per_depth[node.depth][label_value_set.index(label_value)] += 1
        
list_g_per_d = [[] for _ in range(MAX_DEPTH+1)]
def print_split_count(node, d):
    if node.depth == d:
        a_s = node.split_count
        b_s = node.nonsplit_count 
        tmp = a_s / (a_s + b_s)
        list_g_per_d[d].append(tmp)
    if node.depth < MAX_DEPTH:
        print_split_count(node.ul_node,d)
        print_split_count(node.ur_node,d)
        print_split_count(node.ll_node,d)
        print_split_count(node.lr_node,d)


##########################
#          main          #
##########################

if __name__ == '__main__':
    img_array = []
    label_array = []
    Pix_value = [[] for i in range(num_label)]
    Pix_value_error2 = [[] for i in range(num_label)]
    Mu = [0 for i in range(num_label)]
    Sigma2 = [0 for i in range(num_label)]
    G=[]
    G_record = [[] for _ in range(MAX_DEPTH+1)]

    #gの推定#
    for image_num, file_name in enumerate(train_file_names):
        #画像の読み込み
        img = Image.open(os.path.join(image_folder_path, file_name))
        region_img = Image.open(os.path.join(gt_folder_path, file_name))
        tmp_array = np.array(img)
        img_array.append(tmp_array)
        region_img_array = np.array(region_img)
        label_array.append(region_img_array)

    #根ノード定義
    root = Node(0, HEIGHT, 0 ,WIDTH, 0)  

    #四分木構造
    make_tree(root)
    
    for i in range(N):
        recursive_split(root, i)

        for d in range(MAX_DEPTH):
            print_split_count(root, d)
            G_record[d].append(sum(list_g_per_d[d])/(4**d))
        list_g_per_d = [[] for _ in range(MAX_DEPTH+1)]
    
    for d in range(MAX_DEPTH):
        G.append(G_record[d][-1])
    G.append(0)
    print("G = "+str(G))


    #pの推定#
    #print(count_label_per_depth)
    record_p=[[0 for J in range(num_label)] for I in range(MAX_DEPTH+1)]
    for d in range(MAX_DEPTH+1):
        tmp = sum(count_label_per_depth[d])
        for i in range(num_label):
            if tmp == 0:
                record_p[d][i] = np.nan
            else:
                record_p[d][i] = count_label_per_depth[d][i]/tmp
    print("probability = "+str(record_p))
    
    #θの推定#
    for image_num, file_name in enumerate(train_file_names):
        #画像の読み込み
        img = Image.open(os.path.join(image_folder_path, file_name))
        region_img = Image.open(os.path.join(gt_folder_path, file_name))
        img_array = np.array(img)
        region_img_array = np.array(region_img)

        for w in range(WIDTH):
            for h in range(HEIGHT):
                for l in label_set:
                    if region_img_array[w][h] == label_value_set[l]:
                        Pix_value[l].append(img_array[w][h])

    for l in range(num_label):
        tmp_list = list(map(list, zip(*Pix_value[l])))
        Mu[l] = [np.sum(tmp)/ len(tmp) for tmp in tmp_list]
    print("MEAN = "+str(Mu))

    for image_num, file_name in enumerate(train_file_names):
        #画像の読み込み
        img = Image.open(os.path.join(image_folder_path, file_name))
        region_img = Image.open(os.path.join(gt_folder_path, file_name))
        img_array = np.array(img)
        region_img_array = np.array(region_img)

        for w in range(WIDTH):
            for h in range(HEIGHT):
                for l in label_set:
                    if region_img_array[w][h] == label_value_set[l]:
                        tmp = (img_array[w][h] - Mu[l])**2
                        Pix_value_error2[l].append(tmp)

    for l in range(num_label):
        tmp_list = list(map(list,zip(*Pix_value_error2[l])))
        Sigma2[l] = [np.sqrt(np.sum(tmp)/ len(tmp)) for tmp in tmp_list]
    print("STD = " +str(Sigma2))    


    print("Finish: Parameter Estimation")