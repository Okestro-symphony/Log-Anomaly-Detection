def train_isolation_forest(df, padding_data):
    '''
    * Isolation Forest model setting
    - n_estimators=100
    - max_samples='auto'
    - n_jobs=-1
    - max_features=2
    - contamination=0.01
    '''
    #padding한 data load
    data_df = padding_data

    # model 정의
    model = IsolationForest(n_estimators=100, max_samples='auto', n_jobs=-1,
                            max_features=2, contamination=0.01)

    try:
        model = model.fit(data_df)
    except Exception as ex:
        print('모델 실행 실패 : ', ex)

   
    try:
        # score & anomaly 판단 여부 값 추가
        score = model.decision_function(data_df)
        anomaly = model.predict(data_df)

    except Exception as ex:
        print('이상징후 판별 실패 : ', ex)

    # anomaly_data = df.loc[df['is_anomaly'] == -1]  # 이상값은 -1으로 나타낸다.
    return df
