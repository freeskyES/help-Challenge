
import pandas as pd
import os
import numpy as np
import logging
logging.basicConfig(level = logging.INFO)

d_vital_sign = dict() # dictionary
d_vital_sign['DBP'] = ["ABPd","ARTd","NBPd"]
d_vital_sign['MBP'] = ["ABPm","ARTm","NBPm"]
d_vital_sign['SBP'] = ["ABPs","ARTs","NBPs"]
d_vital_sign['PR'] = ["HR","Pulse","SPO2-R"]
d_vital_sign['RR'] = ["RR","Resp"]
d_vital_sign['SPO2'] = ["SpO2-%","SPO2-%", "SpO2","SpO2T"]
d_vital_sign['BT'] = ["Temp", "Trect"]

__measurement_file_name = 'MEASUREMENT_NICU.csv'
__condition_occurrence_file_name ='CONDITION_OCCURRENCE_NICU.csv'
__outcome_cohort_file_name = 'OUTCOME_COHORT.csv'
__person_file_name = 'PERSON_NICU.csv'

map_vital_sign = dict()
for key in d_vital_sign.keys():
    vals = d_vital_sign[key]
    for val in vals:
        map_vital_sign[val] = key

def get_vital_sign_name(name):
    return map_vital_sign[name]

def load_data_set(directory):
    person_table = pd.read_csv(os.path.join(directory,__person_file_name), encoding = 'windows-1252')
    outcome_cohort_table = pd.read_csv(os.path.join(directory,__outcome_cohort_file_name), encoding = 'windows-1252')
    condition_occurrence_table = pd.read_csv(os.path.join(directory,__condition_occurrence_file_name), encoding = 'windows-1252')
    measurement_table = pd.read_csv(os.path.join(directory,__measurement_file_name), encoding = 'windows-1252')
    return person_table, condition_occurrence_table, outcome_cohort_table, measurement_table

def preprocess_measurement(df):
    df.MEASUREMENT_DATETIME = pd.to_datetime(df.MEASUREMENT_DATETIME.values)
    logging.info('MEASUREMENT_DATETIME %s', df.MEASUREMENT_DATETIME)

    # ~: 비트 보수 연산자, 값 -x / 논리값을 반대로 바꿈 > 데이터 집합을 정리하고 NAN 없이 열을 반환 시킬 수 있음

    df = df[(~np.isnan(df.VALUE_AS_NUMBER.values))] # VALUE_AS_NUMBER 값에 NaN (Not a Number) 없이 열을 반환시킴.
    df.sort_values(by = 'MEASUREMENT_DATETIME', inplace = True) # 정렬, MEASUREMENT_DATETIME 기준으로 ASC 정렬, 정렬된 리스트로 적용

    df['VITAL_SIGN_NAME'] = [None] * len(df) # 새로운 Key VITAL_SIGN_NAME value 값에 list로 None 을 df(사전) 의 총 길이만큼 넣기 [None, None, ...]

    for sub_name in map_vital_sign:
        key = map_vital_sign[sub_name]
        idx_list = df.MEASUREMENT_SOURCE_VALUE.values == sub_name  # MEASUREMENT_SOURCE_VALUE 값과 같은 데이터들을
        df.loc[idx_list,'VITAL_SIGN_NAME'] = key  # [행 인덱싱값, 열 인덱싱값]
        
        
    df['MEASUREMENT_DATETIME_TOTAL_SECONDS'] = df.MEASUREMENT_DATETIME.values.astype(np.int64)
    df.set_index(keys=['PERSON_ID','MEASUREMENT_DATETIME_TOTAL_SECONDS'], drop=False, inplace=True) #인덱스 셋팅
    df.sort_index(inplace = True) #정렬
    return df

def generate_x(person_id, cohort_start_time, cohort_end_time, measurement_df):
    x_input = []
    end_time_int = pd.to_datetime(cohort_end_time).value
    start_time_int = pd.to_datetime(cohort_start_time).value

    sub_measurement_df = measurement_df.loc[person_id,:].loc[start_time_int:end_time_int,:]

    for vital_sign_name in d_vital_sign:
        x= sub_measurement_df.loc[(sub_measurement_df.VITAL_SIGN_NAME == vital_sign_name),['VALUE_AS_NUMBER','MEASUREMENT_DATETIME_TOTAL_SECONDS']].values
        x = x[:,0]
        x = x[~np.isnan(x)]
        if len(x) == 0:
            x_input.append(0.0)
        else:
            x_input.append(np.mean(x)) # x 값들의 평균
    return x_input


def train_model(model, person_df, condition_occurrence_df, measurement_df, outcome_cohort_df):
    x_train = []
    y_train = []
    for _, row in outcome_cohort_df.iterrows():
        x = generate_x(row.SUBJECT_ID, row.COHORT_START_DATE, row.COHORT_END_DATE, measurement_df)
        x_train.append(x)
        y_train.append(row.LABEL)
    
    model.fit(x_train,y_train)
    return model

def predict(model, person_df, condition_occurrence_df, measurement_df, outcome_cohort_df):
    x_test = []
    for _, row in outcome_cohort_df.iterrows():
        x = generate_x(row.SUBJECT_ID, row.COHORT_START_DATE, row.COHORT_END_DATE, measurement_df)
        x_test.append(x)

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)
    return y_pred, y_proba