import os
import util
from time import sleep
from sklearn import ensemble
import pickle
import pandas as pd
OUTPUT_DIR = '/data/output'
TEST_DIR = '/data/test'
VOL_DIR = '/data/volume'

def inference():
  test_model = pickle.load(open(os.path.join(VOL_DIR,'model.dat'),'rb'))
  person_table, condition_occurrence_table, outcome_cohort_table, measurement_table = util.load_data_set(TEST_DIR)
  measurement_table = util.preprocess_measurement(measurement_table)
  y_pred, y_proba = util.predict(test_model,person_table, condition_occurrence_table,measurement_table,outcome_cohort_table)
  predict_result = pd.DataFrame({'LABEL': y_pred, 'LABEL_PROBABILITY':y_proba})
  predict_result.to_csv(os.path.join(OUTPUT_DIR,'output.csv'),index= False)
if __name__ == "__main__":
  inference()