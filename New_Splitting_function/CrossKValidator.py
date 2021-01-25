from Data_manager.split_functions.split_train_validation_k_fold import split
from Base.Evaluation.Evaluator import EvaluatorHoldout
import scipy.sparse as sp


class CrossKValidator:

    def __init__(self, URM_all, k):
        self.URM_all = URM_all
        self.k = k

    def creating_k_evaluator(self):
        URM_train_list, URM_validation_list = split(self.URM_all, self.k)

        evaluator_list = []

        for URM_validation in URM_validation_list:
            evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
            evaluator_list.append(evaluator_validation)

        return evaluator_list, URM_train_list, URM_validation_list


class CrossKValidator_With_ICM_stack:

    def __init__(self, URM_all, ICM_all, k):
        self.URM_all = URM_all
        self.k = k
        self.ICM_all = ICM_all

    def creating_k_evaluator_with_ICM_stack_URM(self):
        URM_train_list, URM_validation_list = split(self.URM_all, self.k)
        evaluator_list = []
        ICM_stack_list = []

        for i in range(self.k):
            URM_train_t = URM_train_list[i].T
            ICM_and_URM = sp.hstack([self.ICM_all, URM_train_t], format="csr")
            evaluator_validation = EvaluatorHoldout(URM_validation_list[i], cutoff_list=[10])
            evaluator_list.append(evaluator_validation)
            ICM_stack_list.append(ICM_and_URM)

        return evaluator_list, URM_train_list, URM_validation_list, ICM_stack_list

