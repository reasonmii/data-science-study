
### mean, variance, std

scores = [10, 20, 30]

n_student = len(scores)

mean = (scores[0] + scores[1] + scores[2])/n_student
square_of_mean = mean**2
mean_of_square = (scores[0]**2 + scores[1]**2 + scores[2]**2)/n_student

variance = mean_of_square - square_of_mean
std = variance**0.5

print("score mean: ", mean)
print("score variance: ", variance)
print("score std: ", std)

scores[0] = (scores[0] - mean)/std
scores[1] = (scores[1] - mean)/std
scores[2] = (scores[2] - mean)/std

mean = (scores[0] + scores[1] + scores[2])/n_student
square_of_mean = mean**2
mean_of_square = (scores[0]**2 + scores[1]**2 + scores[2]**2)/n_student

variance = mean_of_square - square_of_mean
std = variance**0.5

print("score mean: ", mean)
print("score variance: ", variance)
print("score std: ", std)


### Hadamard Product

v1, v2 = [1, 2, 3], [3, 4, 5]

v3 = [v1[0]*v2[0], v1[1]*v2[1], v1[2]*v2[2]]
print(v3)

v3 = [0, 0, 0]   # 초기화
v3[0] = v1[0]*v2[0]
v3[1] = v1[1]*v2[1]
v3[2] = v1[2]*v2[2]
print(v3)

v3 = list()      # 초기화
v3.append(v1[0]*v2[0])
v3.append(v1[1]*v2[1])
v3.append(v1[2]*v2[2])
print(v3)


### Vector Norm

v1 = [1, 2, 3]
norm = (v1[0]**2 + v1[1]**2 + v1[2]**2)**0.5
print(norm)

norm = 0
norm += v1[0]**2
norm += v1[1]**2
norm += v1[2]**2
norm **= 0.5
print(norm)

v1 = [v1[0]/norm, v1[1]/norm, v1[2]/norm]
norm = (v1[0]**2 + v1[1]**2 + v1[2]**2)**0.5  # unit vector
print(norm)


### confusion vector

preds = [0, 1, 0, 2, 1, 2, 0]
labels = [1, 1, 0, 0, 1, 2, 1]
m_class = None

# 몇 개 class가 있는지 구하기
for label in labels:
    if m_class == None or label > m_class:
        m_class = label
m_class += 1

# class 별 맞은 개수
class_cnts, corr_cnts, confusion_vec = list(), list(), list()
for _ in range(m_class):
    class_cnts.append(0)
    corr_cnts.append(0)
    confusion_vec.append(None)

for idx in range(len(preds)):
    pred = preds[idx]     # 예측값
    label = labels[idx]   # 실제값
    
    class_cnts[label] += 1  # 해당 class 개수
    if pred == label:
        corr_cnts[label] += 1
    
for idx in range(m_class):
    confusion_vec[idx] = corr_cnts[idx]/class_cnts[idx]
    
print(confusion_vec)  # class 별 개수 중 맞은 확률
        

### row-wise mean

scores = [[10, 15, 20], [20, 25, 30], [30, 35, 40], [40, 45, 50]]

n_student = len(scores)
n_class = len(scores[0])

sums, means = list(), list()

for _ in range(n_class):
    sums.append(0)

for score in scores:
    for idx in range(n_class):
        sums[idx] += score[idx]
        
for idx in range(n_class):
    means.append(sums[idx]/n_student)


### mean subtraction

for idx1 in range(n_student):
    for idx2 in range(n_class):
        scores[idx1][idx2] -= means[idx2]

sums, means = list(), list()        
        
for _ in range(n_class):
    sums.append(0)

for score in scores:
    for idx in range(n_class):
        sums[idx] += score[idx]
        
for idx in range(n_class):
    means.append(sums[idx]/n_student)


### variance, std, standardization

scores = [[10, 15, 20], [20, 25, 30], [30, 35, 40], [40, 45, 50]]

n_student = len(scores)
n_class = len(scores[0])

class_sums = list()
class_means = list()
class_square_sums = list()

class_variance = list()
class_std = list()

for _ in range(n_class):
    class_sums.append(0)
    class_square_sums.append(0)
        
for student in scores:
    for idx in range(n_class):
        class_sums[idx] += student[idx]
        class_square_sums[idx] += student[idx]**2

for idx in range(n_class):
    class_means.append(class_sums[idx]/n_student)

# variance, std
for idx in range(n_class):
    mos = class_square_sums[idx]/n_student
    som = class_means[idx]**2
    
    variance = mos - som
    std = variance**0.5
    
    class_variance.append(variance)
    class_std.append(std)

print(class_variance)
print(class_std)

# standardization

for s_idx in range(n_student):
    for c_idx in range(n_class):
        score = scores[s_idx][c_idx]
        mean = class_means[c_idx]
        std = class_std[c_idx]
        
        scores[s_idx][c_idx] = \
            (score - mean)/std


###
v = [[1, 11, 21],
     [2, 12, 22],
     [3, 13, 23],
     [4, 14, 24]]

n_dim = len(v)
n_vec = len(v[0])

# Hadmard Product
h_prod = list()
for dim in v:
    prod = 1
    for val in dim:
        prod *= val
    h_prod.append(prod)

print(h_prod)        

# vector norm
norm = list()
for _ in range(n_vec):
    norm.append(0)
print(norm)

for dim in v:
    for idx in range(n_vec):
        norm[idx] += dim[idx]**2
print(norm)

for idx in range(n_vec):
    norm[idx] **= 0.5
print(norm)

# unit vector
for dim_idx in range(n_dim):
    for vec_idx in range(n_vec):
        v[dim_idx][vec_idx] /= norm[vec_idx]

print(norm)


### 과목별 최고점, 최우수 학생 구하기

scores = [[10, 40, 20],
          [50, 20, 60],
          [70, 40, 30],
          [30, 80, 40]]

n_student = len(scores)  # 4
n_class = len(scores[0]) # 3

m_class = scores[0]
m_idx = list()

for s_idx in range(n_student):
    student_scores = scores[s_idx]
    
    for c_idx in range(n_class):
        score = student_scores[c_idx]
        if score > m_class[c_idx]:
            m_class[c_idx] = score
            m_idx[c_idx] = s_idx
    

### One-hot Encoding

labels = [0, 1, 2, 1, 0, 3]

n_label = len(labels)
n_class = 0

for label in labels:
    if label > n_class:
        n_class = label
        
n_class += 1  # class 개수

one_hot_mat = list()

for label in labels:
    one_hot_vec = list()
    for _ in range(n_class):
        one_hot_vec.append(0)
    one_hot_vec[label] = 1
    one_hot_mat.append(one_hot_vec)

print(one_hot_mat)

# [[1, 0, 0, 0], [0, 1, 0, 0],
#  [0, 0, 1, 0], [0, 1, 0, 0],
#  [1, 0, 0, 0], [0, 0, 0, 1]]


###
preds = [[1, 0, 0, 0], [0, 0, 1, 0],
         [0, 0, 1, 0], [1, 0, 0, 0],
         [1, 0, 0, 0], [0, 0, 0, 1]]

n_pred = len(preds)      # 6
n_class = len(preds[0])  # 4

acc = 0
for p_idx in range(n_pred):
    pred = preds[p_idx]
    label = one_hot_mat[p_idx]
    
    correct = 0
    for c_idx in range(n_class):
        if pred[c_idx] == label[c_idx]:
            correct += 1
    
    if correct == n_class:
        acc += 1

acc /= n_pred
print("accuracy (%)", acc*100, ("%"))


### matrix addition

mat1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
mat2 = [[11, 12, 13], [14, 15, 16], [17, 18, 19]]

n_row = len(mat1)
n_col = len(mat1[0])

mat_add = list()
for r_idx in range(n_row):
    add_tmp = list()
    for c_idx in range(n_col):
        add_tmp.append(mat1[r_idx][c_idx] + mat2[r_idx][c_idx])
    
    mat_add.append(add_tmp)


### matrix-vector multiplication

mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
vec = [10, 20, 30]

n_row = len(mat)
n_col = len(mat[0])

mat_vec_mul = list()
for r_idx in range(n_row):
    mat_vec = mat[r_idx]
    dot_product = 0
    for c_idx in range(n_col):
        dot_product += mat_vec[c_idx]*vec[c_idx]
    
    mat_vec_mul.append(dot_product)

print(mat_vec_mul)  # [140, 320, 500]


### matrix-matrix multiplication

mat1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
mat2 = [[11, 12, 13], [14, 15, 16], [17, 18, 19]]

n_row = len(mat1)
n_col = len(mat1[0])

mat_mul = list()

for r_idx in range(n_row):
    vec1 = mat1[r_idx]
    dot_prods = list()
    
    for c_idx in range(n_col):
        vec2 = list()
        for idx in range(n_col):
            vec2.append(mat2[idx][c_idx])
        #print(vec2)
        
        dot_prod = 0
        for idx in range(n_col):
            dot_prod += vec1[idx]*vec2[idx]
        dot_prods.append(dot_prod)
        #print(dot_prods)
    
    mat_mul.append(dot_prods)

print(mat_mul)
# [[90, 96, 102], [216, 231, 246], [342, 366, 390]]


### Transposed Matrix
mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
n_row = len(mat)
n_col = len(mat[0])

mat_t = list()
for c_idx in range(n_col):
    vec = list()
    for r_idx in range(n_row):
        vec.append(mat[r_idx][c_idx])
    mat_t.append(vec)
        
mat_t
# [[1, 4, 7], [2, 5, 8], [3, 6, 9]]


