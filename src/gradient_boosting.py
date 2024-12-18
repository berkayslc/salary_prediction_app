import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Veri Yükleme
df = pd.read_csv("../data/salary.csv")

# 2. Salary Dönüşüm Fonksiyonu
def salary_to_numeric(salary_range):
    """
    Maaş aralıklarını sayısal ortalama değere dönüştürür.
    Örneğin: '41.000 - 45.000 TL' -> 43000.0
    """
    if isinstance(salary_range, str):
        numbers = re.findall(r'\d+\.\d+|\d+', salary_range)
        numbers = [int(num.replace('.', '')) for num in numbers]
        if len(numbers) == 2:
            return sum(numbers) / 2  # Ortalama değer
        elif len(numbers) == 1:
            return numbers[0]  # Tek sayı varsa olduğu gibi al
    return None

# Salary sütununu dönüştürme
df['Salary_numeric'] = df['Salary'].apply(salary_to_numeric)

# Salary dönüşüm fonksiyonunu kaydetme
joblib.dump(salary_to_numeric, '../encoders/salary_to_numeric_gb.pkl')
print("Salary dönüşüm fonksiyonu kaydedildi.")

# 3. Kategorik Verilerin Dönüşümü (Label Encoding)
label_encoders = {}
categorical_columns = ['Position', 'Level', 'Experience', 'Technology', 
                       'Location', 'Way_of_working', 'Employees_number', 'Salary_type']

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Encoder'ları kaydetme
joblib.dump(label_encoders, '../encoders/label_encoders_gb.pkl')
print("Label encoders başarıyla kaydedildi.")

# 4. Bağımsız ve Bağımlı Değişkenlerin Tanımlanması
X = df.drop(columns=['Salary', 'Salary_numeric', 'Time'])  # Kullanılmayan sütunlar
y = df['Salary_numeric']

# 5. Veriyi Eğitim ve Test Olarak Bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Özelliklerin Ölçeklendirilmesi (StandardScaler)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Scaler'ı kaydetme
joblib.dump(scaler, '../scalers/scaler_gb.pkl')
print("Scaler başarıyla kaydedildi.")

# 7. GradientBoostingRegressor Modelinin Eğitilmesi
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Modeli kaydetme
joblib.dump(model, '../models/gradient_boosting_model.pkl')
print("GradientBoostingRegressor modeli başarıyla kaydedildi.")

# 8. Model Performansının Değerlendirilmesi
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Sonuçları dosyaya kaydetme
with open("../results/gradient_boosting_results.txt", "w", encoding="utf-8") as f:
    f.write("\nModel Performansı:")
    f.write(f"\nMean Absolute Error (MAE): {mae:.2f}")
    f.write(f"\nMean Squared Error (MSE): {mse:.2f}")
    f.write(f"\nR2 Score: {r2:.4f}")

print("Model performans sonuçları başarıyla kaydedildi.")
