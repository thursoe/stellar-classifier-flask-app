from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import warnings, os

warnings.filterwarnings('ignore')

# Load the dataset
rel_path = os.path.join(os.getcwd(), "assets", "star240_balanced.csv")
data = pd.read_csv(rel_path)

# Encoding
label_encoder_specclass = LabelEncoder()
data['Spectral Class'] = label_encoder_specclass.fit_transform(data['Spectral Class'])

# Feature selection
X = data[['Temp', 'L', 'R', 'Abs Mag', 'Spectral Class']]
y = data['Target Type']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dimensionality reduction with PCA
pca = PCA(n_components=0.95)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
n_components = X_train.shape[1]

logistic_regression = LogisticRegression(multi_class='ovr', max_iter=1000, C=0.001)
svc = SVC(kernel='linear', probability=True)
voting_clf = VotingClassifier(estimators=[
    ('lr', logistic_regression),
    ('svc', svc)
], voting='soft')

param_grid = {'lr__C': [0.1, 1.0, 10],'svc__C': [0.1, 1.0, 10]}
grid_search = GridSearchCV(estimator=voting_clf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Function to predict new data
def make_predict(model, new_data):
    target_names = ['Brown Dwarf', 'Red Dwarf', 'White Dwarf',
                'Main Sequence', 'Supergiant', 'Hypergiant']
    new_data_df = pd.DataFrame([new_data], columns=X.columns)
    new_data_df['Spectral Class'] = label_encoder_specclass.transform(new_data_df['Spectral Class'])

    new_data_scaled = scaler.transform(new_data_df)
    new_data_pca = pca.transform(new_data_scaled)

    prediction = model.predict(new_data_pca)

    return target_names[prediction[0]]

# Model Serialization
import joblib
joblib.dump(best_model, 'model_stellar.pkl')