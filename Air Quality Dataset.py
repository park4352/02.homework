import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate_model(file_path: object) -> object:
    # CSV 파일에서 데이터 읽기
    data = pd.read_csv(file_path, header=None)  # 헤더가 없는 경우에는 header=None으로 설정합니다.

    # 데이터 전처리
    X = data.iloc[:, :-1]  # 센서 출력 값을 feature로 사용
    y = data.iloc[:, -1]   # 공기질을 나타내는 값은 마지막 열로 가정합니다.

    # 데이터를 학습 세트와 테스트 세트로 나누기
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 머신러닝 모델 선택 및 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 테스트 세트로 예측 수행
    y_pred = model.predict(X_test)

    # 정확도 및 분류 보고서 출력
    accuracy = accuracy_score(y_test, y_pred)
    print(f"모델 정확도: {accuracy}")
    print("분류 보고서:\n", classification_report(y_test, y_pred))

    return model

def predict_air_quality(model, new_data):
    # 새로운 데이터에 대한 예측 수행
    predictions = model.predict(new_data)
    return predictions

if __name__ == "__main__":

    training_file_path = "/Users/admin/PycharmProjects/pythonProject12/dataset.csv"

    # 모델 학습
    trained_model = train_and_evaluate_model(training_file_path)

    # 사용자에게 입력 받은 새로운 데이터 (센서 출력 값) -
    new_data = pd.DataFrame([[670, 696, 1252, 1720, 1321, 2431]],
                            columns=['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5', 'sensor6'])

    # 공기질 예측
    predictions = predict_air_quality(trained_model, new_data)

    # 결과 출력
    print("예측된 공기질:", predictions[0])

