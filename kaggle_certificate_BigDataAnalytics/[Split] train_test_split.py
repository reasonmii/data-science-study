
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(train.drop('TravelInsurance', axis=1), train['TravelInsurance'], test_size=0.1, random_state=1204)

