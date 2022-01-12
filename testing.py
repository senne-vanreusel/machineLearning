import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

data = [15, 16, 18, 19, 22, 24, 29, 30, 34]

# print("mean ", np.mean(data))
# print("median ", np.median(data))
# print("50th percentile ", np.percentile(data, 50))
# print("25th percentile ", np.percentile(data, 25))
# print("75th percentile ", np.percentile(data, 75))
# print("standard deviation ", np.std(data))
# print("variance ", np.var(data))

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
# print(df.head())
# print(df.describe())
col = df['Fare']
# print(col)
small_df = df[['Age', 'Sex', 'Survived']]
# print(small_df.head())
df['male'] = df['Sex'] == 'male'
# print(df.head())
# print(col.values)
# print(small_df.values)
# print(small_df.shape)
# print(small_df.values[:,2])
arr = df[['Pclass', 'Fare', 'Age']].values
# print(arr[:, 2])
mask = arr[:, 2] < 18
# print(arr[arr[:, 2] < 18])
# print(mask.sum())
#
# plt.scatter(df['Age'], df['Fare'], c=df['Pclass'])
# plt.xlabel('Age')
# plt.ylabel('Fare')
# plt.plot([0, 80], [85, 5])
# plt.show()
#
# print(df['Age'].values)

plt.scatter(df['Fare'], df['Age'], c=df['Survived'])
plt.xlabel('Fare')
plt.ylabel('Age')
plt.show()

x = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

#
# model = LogisticRegression()
#
# model.fit(x, y)
#  print(model.coef_,model.intercept_)
#
# y_predict = model.predict(x)
# print((y == y_predict).sum()/y.shape[0])
# print(model.score(x,y))




cancer_data = load_breast_cancer()
print(cancer_data.keys())
# print(cancer_data['DESCR'])

df = pd.DataFrame(cancer_data['data'],
                  columns=cancer_data['feature_names'])
df['target'] = cancer_data['target']
print(df.head())
print(cancer_data['target_names'])

x=df[cancer_data.feature_names].values
y=df['target'].values
model = LogisticRegression(solver='liblinear')
model.fit(x,y)

print(model.score(x,y))