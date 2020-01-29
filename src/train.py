import os
import util
from time import sleep
from sklearn import ensemble
import pickle
ID = os.environ['ID']

TRAIN_DIR = '/data/train'
LOG_DIR = '/data/volume/logs'
VOL_DIR = '/data/volume'
TRAINING_EPOCHS = 300

def train():
  test_model = ensemble.GradientBoostingClassifier()
  person_table, condition_occurrence_table, outcome_cohort_table, measurement_table = util.load_data_set(TRAIN_DIR)
  measurement_table = util.preprocess_measurement(measurement_table)
  test_model = util.train_model(test_model,person_table, condition_occurrence_table, measurement_table, outcome_cohort_table)
  pickle.dump(test_model, open(os.path.join(VOL_DIR,'model.dat'),'wb')) # 데이터 입력
if __name__ == "__main__":
  train()