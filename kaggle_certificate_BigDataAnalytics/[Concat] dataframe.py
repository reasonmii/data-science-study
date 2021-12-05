
'''
수치형 컬럼과 범주형 컬럼 각각
modeling에 맞게 변환 후 한 테이블로 합치기

n_train : 수치형 컬럼
c_train : 범주형 컬럼
'''

train = pd.concat([n_train, c_train], axis=1)

