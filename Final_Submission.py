from Utils.load_data_2020 import load_URM, load_ICM
from New_Splitting_function.CrossKValidator import CrossKValidator
from Hybrid_recommender.ItemKNNScoresHybridRecommender import ItemKNNScoresHybridThreeRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from BO.bayes_opt import BayesianOptimization
from Base.Evaluation.Evaluator import EvaluatorHoldout
from BO.bayes_opt import SequentialDomainReductionTransformer
from BO.bayes_opt.logger import JSONLogger
from BO.bayes_opt.event import Events
from Notebooks_utils.data_splitter import train_test_holdout
from MatrixFactorization.IALS_implicit_Recommender import IALSRecommender_implicit
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
import pandas as pd
import csv
import numpy as np
import scipy.sparse as sps
# ---------------------------------------------------------------------------------------------------------
# Loading URM and ICM
URM_all = load_URM("Data/data_train.csv")
ICM_all = load_ICM("Data/data_ICM_title_abstract.csv")

# Feature combination
URM_stack_all = sps.vstack([URM_all, ICM_all.T])

######### ----------------------------------------------------------------------------------------------------------
# Recommender 1 - iALS implicit
recommender_IALS = IALSRecommender_implicit(URM_stack_all)
recommender_IALS.fit(n_factors=int(469.47983415825166), regularization=5.550461180441473e-06,
                         iterations=int(39.2008537931411), alpha_val=42.991885722226804, use_gpu=False)

# Recommender 2 - Graph based Rp3Beta
recommender_rp3Beta = RP3betaRecommender(URM_stack_all)
recommender_rp3Beta.fit(topK=int(366.00024582537884), alpha=0.38591507235778016, beta=0.13109580041155922, implicit=True)

# Recommender 3 - Slim Elastic Net
recommender_elastic = SLIMElasticNetRecommender(URM_stack_all)
elastic_param = {'topK': 414, 'l1_ratio': 0.0006293695094936683, 'alpha': 0.0014914689856932379}
recommender_elastic.fit(**elastic_param)

# Final Hybrid Recommender
recommender = ItemKNNScoresHybridThreeRecommender(URM_train=URM_stack_all,
                                                             Recommender_1=recommender_IALS,
                                                             Recommender_2=recommender_rp3Beta,
                                                             Recommender_3=recommender_elastic)
recommender.fit(alpha=0.269754707380376, beta=0.9146463918776708, gamma=0.5887379007121888)

# Writing submission
from Utils.writing_submission import write_submission
write_submission(recommender, "Final_Submission.csv")
