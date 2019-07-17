import numpy as np
from distutils.version import LooseVersion
from shap.common import assert_import, DenseData
from shap.explainers.tree import TreeExplainer, TreeEnsemble, Tree, XGBTreeModelLoader, CatBoostTreeModelLoader
import scipy
import xgboost
output_transform_codes = {
    "identity": 0,
    "logistic": 1,
    "logistic_nlogloss": 2,
    "squared_loss": 3
}

feature_dependence_codes = {
    "independent": 0,
    "tree_path_dependent": 1,
    "global_path_dependent": 2
}

class MultiTreeExplainer(TreeExplainer):
    """Uses Tree SHAP algorithms to explain the output of ensemble tree models.

    Tree SHAP is a fast and exact method to estimate SHAP values for tree models and ensembles of trees,
    under several different possible assumptions about feature dependence. It depends on fast C++
    implementations either inside an externel model package or in the local compiled C extention.

    Parameters
    ----------
    model : model object
        The tree based machine learning model that we want to explain. XGBoost, LightGBM, CatBoost,
        and most tree-based scikit-learn models are supported.

    data : numpy.array or pandas.DataFrame
        The background dataset to use for integrating out features. This argument is optional when
        feature_dependence="tree_path_dependent", since in that case we can use the number of training
        samples that went down each tree path as our background dataset (this is recorded in the model object).

    feature_dependence : "tree_path_dependent" (default) or "independent"
        Since SHAP values rely on conditional expectations we need to decide how to handle correlated
        (or otherwise dependent) input features. The default "tree_path_dependent" approach is to just
        follow the trees and use the number of training examples that went down each leaf to represent
        the background distribution. This approach repects feature dependecies along paths in the trees.
        However, for non-linear marginal transforms (like explaining the model loss)  we don't yet
        have fast algorithms that respect the tree path dependence, so instead we offer an "independent"
        approach that breaks the dependencies between features, but allows us to explain non-linear
        transforms of the model's output. Note that the "independent" option requires a background
        dataset and its runtime scales linearly with the size of the background dataset you use. Anywhere
        from 100 to 1000 random background samples are good sizes to use.

    model_output : "margin", "probability", or "log_loss"
        What output of the model should be explained. If "margin" then we explain the raw output of the
        trees, which varies by model (for binary classification in XGBoost this is the log odds ratio).
        If "probability" then we explain the output of the model transformed into probability space
        (note that this means the SHAP values now sum to the probability output of the model). If "log_loss"
        then we explain the log base e of the model loss function, so that the SHAP values sum up to the
        log loss of the model for each sample. This is helpful for breaking down model performance by feature.
        Currently the probability and log_loss options are only supported when feature_dependence="independent".
    """

    def __init__(self, model, data=None, model_output="margin", feature_dependence="tree_path_dependent"):
        if str(type(data)).endswith("pandas.core.frame.DataFrame'>"):
            self.data = data.values
        elif isinstance(data, DenseData):
            self.data = data.data
        else:
            self.data = data
        self.data_missing = None if self.data is None else np.isnan(self.data)
        self.model_output = model_output
        self.feature_dependence = feature_dependence
        self.expected_value = None
        self.model = MultiTreeEnsemble(model, self.data, self.data_missing)

        assert feature_dependence in feature_dependence_codes, "Invalid feature_dependence option!"

        # check for unsupported combinations of feature_dependence and model_outputs
        if feature_dependence == "tree_path_dependent":
            assert model_output == "margin", "Only margin model_output is supported for feature_dependence=\"tree_path_dependent\""
        else:
            assert data is not None, "A background dataset must be provided unless you are using feature_dependence=\"tree_path_dependent\"!"

        if model_output != "margin":
            if self.model.objective is None and self.model.tree_output is None:
                raise Exception("Model does have a known objective or output type! When model_output is " \
                                "not \"margin\" then we need to know the model's objective or link function.")

        # A bug in XGBoost fixed in v0.81 makes XGBClassifier fail to give margin outputs
        if str(type(model)).endswith("xgboost.sklearn.XGBClassifier'>") and model_output != "margin":
            assert_import("xgboost")
            assert LooseVersion(xgboost.__version__) >= LooseVersion('0.81'), \
                "A bug in XGBoost fixed in v0.81 makes XGBClassifier fail to give margin outputs! Please upgrade to XGBoost >= v0.81!"

        # compute the expected value if we have a parsed tree for the cext
        if self.model_output == "logloss":
            self.expected_value = self.__dynamic_expected_value
        elif data is not None:
            self.expected_value = self.model.predict(self.data, output=model_output).mean(0)
        elif hasattr(self.model, "node_sample_weight"):
            self.expected_value = self.model.values[:, 0].sum(0)
            self.expected_value += self.model.base_offset


class MultiTreeEnsemble(TreeEnsemble):
    """ An ensemble of decision trees.

    This object provides a common interface to many different types of models.
    """

    def __init__(self, model, data=None, data_missing=None):
        self.model_type = "internal"
        self.trees = None
        less_than_or_equal = True
        self.base_offset = 0
        self.objective = None  # what we explain when explaining the loss of the model
        self.tree_output = None  # what are the units of the values in the leaves of the trees
        self.dtype = np.float64  # for sklearn we need to use np.float32 to always get exact matches to their predictions
        self.data = data
        self.data_missing = data_missing
        self.fully_defined_weighting = True  # does the background dataset land in every leaf (making it valid for the tree_path_dependent method)
        self.tree_limit = None  # used for limiting the number of trees we use by default (like from early stopping)

        # we use names like keras
        objective_name_map = {
            "mse": "squared_error",
            "friedman_mse": "squared_error",
            "reg:linear": "squared_error",
            "regression": "squared_error",
            "regression_l2": "squared_error",
            "mae": "absolute_error",
            "gini": "binary_crossentropy",
            "entropy": "binary_crossentropy",
            "binary:logistic": "binary_crossentropy",
            "binary_logloss": "binary_crossentropy",
            "binary": "binary_crossentropy"
        }

        tree_output_name_map = {
            "regression": "raw_value",
            "regression_l2": "squared_error",
            "reg:linear": "raw_value",
            "binary:logistic": "log_odds",
            "binary_logloss": "log_odds",
            "binary": "log_odds"
        }

        if type(model) == list and type(model[0]) == Tree:
            self.trees = model
        elif str(type(model)).endswith("sklearn.ensemble.forest.RandomForestRegressor'>"):
            self.dtype = np.float32
            scaling = 1.0 / len(model.estimators_)  # output is average of trees
            self.trees = [Tree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in
                          model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif str(type(model)).endswith("skopt.learning.forest.RandomForestRegressor'>"):
            self.dtype = np.float32
            scaling = 1.0 / len(model.estimators_)  # output is average of trees
            self.trees = [Tree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in
                          model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif str(type(model)).endswith("sklearn.ensemble.forest.ExtraTreesRegressor'>"):
            self.dtype = np.float32
            scaling = 1.0 / len(model.estimators_)  # output is average of trees
            self.trees = [Tree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in
                          model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif str(type(model)).endswith("skopt.learning.forest.ExtraTreesRegressor'>"):
            self.dtype = np.float32
            scaling = 1.0 / len(model.estimators_)  # output is average of trees
            self.trees = [Tree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in
                          model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif str(type(model)).endswith("sklearn.tree.tree.DecisionTreeRegressor'>"):
            self.dtype = np.float32
            self.trees = [Tree(model.tree_, data=data, data_missing=data_missing)]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif str(type(model)).endswith("sklearn.tree.tree.DecisionTreeClassifier'>"):
            self.dtype = np.float32
            self.trees = [Tree(model.tree_, normalize=True, data=data, data_missing=data_missing)]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        elif str(type(model)).endswith("sklearn.ensemble.forest.RandomForestClassifier'>"):
            self.dtype = np.float32
            scaling = 1.0 / len(model.estimators_)  # output is average of trees
            self.trees = [Tree(e.tree_, normalize=True, scaling=scaling, data=data, data_missing=data_missing) for e in
                          model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        elif str(type(model)).endswith(
                "sklearn.ensemble.forest.ExtraTreesClassifier'>"):  # TODO: add unit test for this case
            self.dtype = np.float32
            scaling = 1.0 / len(model.estimators_)  # output is average of trees
            self.trees = [Tree(e.tree_, normalize=True, scaling=scaling, data=data, data_missing=data_missing) for e in
                          model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        elif str(type(model)).endswith("sklearn.ensemble.gradient_boosting.GradientBoostingRegressor'>"):
            self.dtype = np.float32

            # currently we only support the mean and quantile estimators
            if str(type(model.init_)).endswith("ensemble.gradient_boosting.MeanEstimator'>"):
                self.base_offset = model.init_.mean
            elif str(type(model.init_)).endswith("ensemble.gradient_boosting.QuantileEstimator'>"):
                self.base_offset = model.init_.quantile
            elif str(type(model.init_)).endswith("sklearn.dummy.DummyRegressor'>"):
                self.base_offset = model.init_.constant_[0]
            else:
                assert False, "Unsupported init model type: " + str(type(model.init_))

            self.trees = [Tree(e.tree_, scaling=model.learning_rate, data=data, data_missing=data_missing) for e in
                          model.estimators_[:, 0]]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif str(type(model)).endswith("sklearn.ensemble.gradient_boosting.GradientBoostingClassifier'>"):
            self.dtype = np.float32

            # TODO: deal with estimators for each class
            if model.estimators_.shape[1] > 1:
                assert False, "GradientBoostingClassifier is only supported for binary classification right now!"

            # currently we only support the logs odds estimator
            if str(type(model.init_)).endswith("ensemble.gradient_boosting.LogOddsEstimator'>"):
                self.base_offset = model.init_.prior
                self.tree_output = "log_odds"
            elif str(type(model.init_)).endswith("sklearn.dummy.DummyClassifier'>"):
                self.base_offset = scipy.special.logit(
                    model.init_.class_prior_[1])  # with two classes the trees only model the second class
                self.tree_output = "log_odds"
            else:
                assert False, "Unsupported init model type: " + str(type(model.init_))

            self.trees = [Tree(e.tree_, scaling=model.learning_rate, data=data, data_missing=data_missing) for e in
                          model.estimators_[:, 0]]
            self.objective = objective_name_map.get(model.criterion, None)
        elif str(type(model)).endswith("xgboost.core.Booster'>"):
            assert_import("xgboost")
            self.original_model = model
            self.model_type = "xgboost"
            xgb_loader = XGBTreeModelLoader(self.original_model)
            self.trees = xgb_loader.get_trees(data=data, data_missing=data_missing)
            self.base_offset = xgb_loader.base_score
            less_than_or_equal = False
            self.objective = objective_name_map.get(xgb_loader.name_obj, None)
            self.tree_output = tree_output_name_map.get(xgb_loader.name_obj, None)
        elif str(type(model)).endswith("xgboost.sklearn.XGBClassifier'>") or \
                str(type(model)).endswith("multi_xgboost.MultiXGBClassifier'>"):
            assert_import("xgboost")
            self.dtype = np.float32
            self.model_type = "xgboost"
            self.original_model = model.get_booster()
            xgb_loader = XGBTreeModelLoader(self.original_model)
            self.trees = xgb_loader.get_trees(data=data, data_missing=data_missing)
            self.base_offset = xgb_loader.base_score
            less_than_or_equal = False
            self.objective = objective_name_map.get(xgb_loader.name_obj, None)
            self.tree_output = tree_output_name_map.get(xgb_loader.name_obj, None)
            self.tree_limit = getattr(model, "best_ntree_limit", None)
        elif str(type(model)).endswith("xgboost.sklearn.XGBRegressor'>"):
            assert_import("xgboost")
            self.original_model = model.get_booster()
            self.model_type = "xgboost"
            xgb_loader = XGBTreeModelLoader(self.original_model)
            self.trees = xgb_loader.get_trees(data=data, data_missing=data_missing)
            self.base_offset = xgb_loader.base_score
            less_than_or_equal = False
            self.objective = objective_name_map.get(xgb_loader.name_obj, None)
            self.tree_output = tree_output_name_map.get(xgb_loader.name_obj, None)
            self.tree_limit = getattr(model, "best_ntree_limit", None)
        elif str(type(model)).endswith("xgboost.sklearn.XGBRanker'>"):
            assert_import("xgboost")
            self.original_model = model.get_booster()
            self.model_type = "xgboost"
            xgb_loader = XGBTreeModelLoader(self.original_model)
            self.trees = xgb_loader.get_trees(data=data, data_missing=data_missing)
            self.base_offset = xgb_loader.base_score
            less_than_or_equal = False
            # Note: for ranker, leaving tree_output and objective as None as they
            # are not implemented in native code yet
            self.tree_limit = getattr(model, "best_ntree_limit", None)
        elif str(type(model)).endswith("lightgbm.basic.Booster'>"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            self.original_model = model
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [Tree(e, data=data, data_missing=data_missing) for e in tree_info]
            except:
                self.trees = None  # we get here because the cext can't handle categorical splits yet

            self.objective = objective_name_map.get(model.params.get("objective", "regression"), None)
            self.tree_output = tree_output_name_map.get(model.params.get("objective", "regression"), None)

        elif str(type(model)).endswith("lightgbm.sklearn.LGBMRegressor'>"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            self.original_model = model.booster_
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [Tree(e, data=data, data_missing=data_missing) for e in tree_info]
            except:
                self.trees = None  # we get here because the cext can't handle categorical splits yet
            self.objective = objective_name_map.get(model.objective, None)
            self.tree_output = tree_output_name_map.get(model.objective, None)
            if model.objective is None:
                self.objective = "squared_error"
                self.tree_output = "raw_value"
        elif str(type(model)).endswith("lightgbm.sklearn.LGBMRanker'>"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            self.original_model = model.booster_
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [Tree(e, data=data, data_missing=data_missing) for e in tree_info]
            except:
                self.trees = None  # we get here because the cext can't handle categorical splits yet
            # Note: for ranker, leaving tree_output and objective as None as they
            # are not implemented in native code yet
        elif str(type(model)).endswith("lightgbm.sklearn.LGBMClassifier'>"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            self.original_model = model.booster_
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [Tree(e, data=data, data_missing=data_missing) for e in tree_info]
            except:
                self.trees = None  # we get here because the cext can't handle categorical splits yet
            self.objective = objective_name_map.get(model.objective, None)
            self.tree_output = tree_output_name_map.get(model.objective, None)
            if model.objective is None:
                self.objective = "binary_crossentropy"
                self.tree_output = "log_odds"
        elif str(type(model)).endswith("catboost.core.CatBoostRegressor'>"):
            assert_import("catboost")
            self.model_type = "catboost"
            self.original_model = model
        elif str(type(model)).endswith("catboost.core.CatBoostClassifier'>"):
            assert_import("catboost")
            self.model_type = "catboost"
            self.original_model = model
            self.dtype = np.float32
            cb_loader = CatBoostTreeModelLoader(model)
            self.trees = cb_loader.get_trees(data=data, data_missing=data_missing)
            self.tree_output = "log_odds"
            self.objective = "binary_crossentropy"
        elif str(type(model)).endswith("catboost.core.CatBoost'>"):
            assert_import("catboost")
            self.model_type = "catboost"
            self.original_model = model
        elif str(type(model)).endswith("imblearn.ensemble._forest.BalancedRandomForestClassifier'>"):
            self.dtype = np.float32
            scaling = 1.0 / len(model.estimators_)  # output is average of trees
            self.trees = [Tree(e.tree_, normalize=True, scaling=scaling, data=data, data_missing=data_missing) for e in
                          model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        else:
            raise Exception("Model type not yet supported by TreeExplainer: " + str(type(model)))

        # build a dense numpy version of all the tree objects
        if self.trees is not None:
            max_nodes = np.max([len(t.values) for t in self.trees])
            assert len(np.unique([t.values.shape[1] for t in
                                  self.trees])) == 1, "All trees in the ensemble must have the same output dimension!"
            ntrees = len(self.trees)
            self.n_outputs = self.trees[0].values.shape[1]

            # important to be -1 in unused sections!! This way we can tell which entries are valid.
            self.children_left = -np.ones((ntrees, max_nodes), dtype=np.int32)
            self.children_right = -np.ones((ntrees, max_nodes), dtype=np.int32)
            self.children_default = -np.ones((ntrees, max_nodes), dtype=np.int32)
            self.features = -np.ones((ntrees, max_nodes), dtype=np.int32)

            self.thresholds = np.zeros((ntrees, max_nodes), dtype=self.dtype)
            self.values = np.zeros((ntrees, max_nodes, self.trees[0].values.shape[1]), dtype=self.dtype)
            self.node_sample_weight = np.zeros((ntrees, max_nodes), dtype=self.dtype)

            for i in range(ntrees):
                l = len(self.trees[i].features)
                self.children_left[i, :l] = self.trees[i].children_left
                self.children_right[i, :l] = self.trees[i].children_right
                self.children_default[i, :l] = self.trees[i].children_default
                self.features[i, :l] = self.trees[i].features
                self.thresholds[i, :l] = self.trees[i].thresholds
                self.values[i, :l, :] = self.trees[i].values
                self.node_sample_weight[i, :l] = self.trees[i].node_sample_weight

                # ensure that the passed background dataset lands in every leaf
                if np.min(self.trees[i].node_sample_weight) <= 0:
                    self.fully_defined_weighting = False

            # If we should do <= then we nudge the thresholds to make our <= work like <
            if not less_than_or_equal:
                self.thresholds = np.nextafter(self.thresholds, -np.inf)

            self.num_nodes = np.array([len(t.values) for t in self.trees], dtype=np.int32)
            self.max_depth = np.max([t.max_depth for t in self.trees])