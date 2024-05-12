import pandas as pd

df = pd.read_csv('train.csv', encoding='utf-8')

df.drop(['bdate', 'life_main', 'people_main', 'occupation_name', 'city', 'career_end', 'career_start'], axis=1,
        inplace=True)

df['sex'] = df['sex'].apply(lambda s: 1 if s == 1 else 0)

df['education_form'] = df['education_form'].fillna('Full-time')

df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df.drop('education_form', axis=1, inplace=True)

df[list(pd.get_dummies(df['education_status']).columns)] = pd.get_dummies(df['education_status'])
df.drop('education_status', axis=1, inplace=True)

df['langs'] = df['langs'].apply(lambda x: len(x.split(';')))

df['last_seen'] = df['last_seen'].apply(lambda x: int(x[0:4]))

df['occupation_type'] = df['occupation_type'].fillna('university')
print(df['occupation_type'].value_counts())

df[list(pd.get_dummies(df['occupation_type']).columns)] = pd.get_dummies(df['occupation_type'])
df.drop('occupation_type', axis=1, inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

y = df['result']
x = df.drop('result', axis=1)

'''x_train, y_train - для обучения
x_test, y_test - для проверки правильности прогноза'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print(accuracy_score(y_test, y_pred) * 100)

teat_df = df = pd.read_csv('test.csv', encoding='utf-8')
teat_df.info()
df.drop(['bdate', 'life_main', 'people_main', 'occupation_name', 'city', 'career_end', 'career_start'], axis=1,
        inplace=True)

df['sex'] = df['sex'].apply(lambda s: 1 if s == 1 else 0)

df['education_form'] = df['education_form'].fillna('no_study')

df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df.drop('education_form', axis=1, inplace=True)

df[list(pd.get_dummies(df['education_status']).columns)] = pd.get_dummies(df['education_status'])
df.drop('education_status', axis=1, inplace=True)

df['langs'] = df['langs'].apply(lambda x: len(x.split(';')))

df['last_seen'] = df['last_seen'].apply(lambda x: int(x[0:4]))

df['occupation_type'] = df['occupation_type'].fillna('РАБОТАТЬ НАДО')
print(df['occupation_type'].value_counts())

df[list(pd.get_dummies(df['occupation_type']).columns)] = pd.get_dummies(df['occupation_type'])
df.drop('occupation_type', axis=1, inplace=True)

x_test = sc.transform(x_test)
y_pred = classifier.predict(x_test)

print(y_pred)

result1 = pd.DataFrame(columns=['id', 'result'])
result1.id = df['id']
result1.result = y_pred
df['result'] = y_pred
df.to_csv('results.csv', index=False)
y_pred.to.cvs('results.cvs')
