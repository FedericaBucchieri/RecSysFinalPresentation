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

# ---------------------------------------------------------------------------------------------------------
# Defining a cross validator to retrieve k evaluators, URM_train and URM_validation

k = 5
cross_validator = CrossKValidator(URM_all=URM_all, k=k)
evaluator_list, URM_train_list, URM_validation_list = cross_validator.creating_k_evaluator()

# Feature combination
URM_stack_list = []
for i in range(k):
    URM_stack_list.append(sps.vstack([URM_train_list[i], ICM_all.T]))

# ----------------------------------------------------------------------------------------------------------
# Defining hyperparameters tuning intervals

tuning_params = {
    "alpha": (0.0, 1.0),
    "beta": (0.0, 1.0),
    "gamma": (0.0, 1.0)
}

# ----------------------------------------------------------------------------------------------------------
# Defining recommenders and fitting them with k-splits

recommender_IALS_list = []
recommender_elastic_list = []
recommender_rp3Beta_list = []

for i in range(k):
    # Recommender 1 - iALS implicit
    recommender_IALS = IALSRecommender_implicit(URM_stack_list[i])
    recommender_IALS.fit(n_factors=int(469.47983415825166), regularization=5.550461180441473e-06,
                         iterations=int(39.2008537931411), alpha_val=42.991885722226804, use_gpu=False)
    recommender_IALS_list.append(recommender_IALS)

    # Recommender 2 - Graph based Rp3Beta
    recommender_rp3Beta = RP3betaRecommender(URM_stack_list[i])
    recommender_rp3Beta.fit(topK=int(366.00024582537884), alpha=0.38591507235778016,
                            beta=0.13109580041155922, implicit=True)
    recommender_rp3Beta_list.append(recommender_rp3Beta)

    # Recommender 3 - Slim Elastic Net
    recommender_elastic = SLIMElasticNetRecommender(URM_stack_list[i])
    elastic_param = {'topK': 414, 'l1_ratio': 0.0006293695094936683, 'alpha': 0.0014914689856932379}
    recommender_elastic.fit(**elastic_param)
    recommender_elastic_list.append(recommender_elastic)



# ----------------------------------------------------------------------------------------------------------
# Performing Hyperparameter Tuning with Bayesian Optimization

def BO_func(alpha, beta, gamma):
    cumulative_Map = 0

    for i in range(k):
        recommender = ItemKNNScoresHybridThreeRecommender(URM_train=URM_stack_list[i],
                                                             Recommender_1=recommender_IALS_list[i],
                                                             Recommender_2=recommender_rp3Beta_list[i],
                                                             Recommender_3=recommender_elastic_list[i])
        recommender.fit(alpha, beta, gamma)

        MAP = evaluator_list[i].evaluateRecommender(recommender)[0][10]["MAP"]
        print("MAP: " + str(MAP) + " -> alpha: " + str(alpha) + " beta: " + str(beta) + " gamma: " + str(gamma) + "\n")
        cumulative_Map += MAP

    print("cumulative MAP: " + str(cumulative_Map / k))
    return cumulative_Map / k


# ----------------------------------------------------------------------------------------------------------
# Defining optimizers attributes

optimizer = BayesianOptimization(
    f=BO_func,
    pbounds=tuning_params,
    verbose=5
)

# Defining a logger to save the tuning
logger = JSONLogger(path="./Stack_Hybrid_IALS_Rp3Beta_Elastic_WithCross.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(
    init_points=60,
    n_iter=70,
)

# ----------------------------------------------------------------------------------------------------------
# printing the final best result using optimizer.max
print("\n\nRESULT\n\n")
print(optimizer.max)
