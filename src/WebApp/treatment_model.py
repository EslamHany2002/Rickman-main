import warnings
import joblib
from sklearn.preprocessing import LabelEncoder

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize LabelEncoders
tumor_type_encoder = LabelEncoder()
level_encoder = LabelEncoder()

# Load model
model = joblib.load(r'C:\Users\abdel\Downloads\tesst\models\model.pkl')

# Fit LabelEncoders during startup
tumor_type_mapping = {1: "Glioma", 2: "Pituitary", 3: "Meningioma"}
level_mapping = {0: "low grade", 1: "high grade"}
tumor_type_encoder.fit(list(tumor_type_mapping.values()))
level_encoder.fit(list(level_mapping.values()))

# Set feature names for DecisionTreeClassifier (assuming it's your model)
model.feature_names = ['age', 'previous_treatments', 'encoded_tumor_type', 'encoded_level', 'spread_of_tumor']

# Predict treatment protocol function
def predict_protocol(age, previous_treatments, tumor_type, level, spread_of_tumor):
    try:
        previous_treatments = int(previous_treatments)
        spread_of_tumor = int(spread_of_tumor)

        # Encode input variables
        encoded_tumor_type = tumor_type_encoder.transform([tumor_type])[0]
        encoded_level = level_encoder.transform([level])[0]

        data = [[age, previous_treatments, encoded_tumor_type, encoded_level, spread_of_tumor]]

        # Perform prediction
        protocol = model.predict(data)[0]
        return protocol
    except Exception as e:
        return str(e)
