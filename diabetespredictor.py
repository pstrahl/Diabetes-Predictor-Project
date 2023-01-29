import pandas as pd
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score


class Diabetes_Predictor():
    """
    This class uses various models to predict diabetes in a patient.

    The scoring used is recall, since we want the model to be sensitive
    to patients with diabetes (minimize False Negatives).

    Several of the features in the dataset provided have skewed distributions,
    so to handle this we take the log(1 + X) transform of these features
    (on all of X, before splitting into training and testing sets).  This will
    not cause leakage since anytime we are given new data, we always have the
    ability to apply this transform.

    From here, we use the KNNImputer to impute missing values in the BloodPressure
    and SkinThickness features, which previously were represented as zeros.
    This occurs inside a pipeline while using GridSearchCV to prevent data leakage.

    Depending on the model, this either completes the preprocessing in each fold,
    or we further apply the MinMaxScaler to obtain data which is all on the
    same scale (between 0 and 1).  This is not important for the
    RandomForestClassifier, but it does make a difference for the
    SupportVectorClassifer.

    Finally, when fitting the models, we pass in the class_weight = balanced attribute
    since it is known that there is class imbalance.  This allows us to effectively
    oversample the minority class.
    """

    def __init__(self, random_state=0):
        """
        Create the dataframe from which X and y will be constructed.

        Returns:
            None
        """
        self.df = pd.read_csv(r"""C:\Users\pkstr\PycharmProjects\Diabetes_predictor
                              \DiabetesPredictorProject\diabetes.csv"""
                              )
        self.X = None
        self.y = None
        self.random_state = random_state

    def get_X_and_y(self):
        """
        Create X and y from df.

        Returns:
            None
        """
        self.X = self.df.drop(labels="Outcome", axis=1)
        self.y = self.df["Outcome"].copy()

    def replace_zero_w_nan(self):
        """
        Replace the zeros in BloodPressure and SkinThickness with np.nan.

        The values of zero in BloodPressure and SkinThickness indicate
        missing data, as these values are not possible.
        During the GridSearchCV, the KNNImputer in sklearn will be used to
        impute these values.

        Returns:
            None
        """
        self.X.loc[:, ["BloodPressure", "SkinThickness"]].replace(to_replace=0,
                                                                  value=np.nan,
                                                                  inplace=True
                                                                  )

    def remove_skew(self):
        """
        Apply log(1 + X) transform to skewed features in X.

        Insulin, Age, DiabetesPedigreeFunction, and Pregnancies all
        have skewed distributions.  To correct for this, we apply
        log(1 + X) to each column.  This does not introduce any
        data leakage, for given new data, we always have the ability to
        take log(1 + X) of each of these entries in the sample.

        Returns:
            None
        """
        features = ["Insulin", "Age", "DiabetesPedigreeFunction", "Pregnancies"]
        for feature in features:
            self.X[feature] = self.X[feature].apply(lambda x: np.log(1 + x))

    def get_training_and_test_data(self):
        """
        Create the training and test data splits.

        Returns:
            None
        """
        self.get_X_and_y()
        self.replace_zero_w_nan()
        self.remove_skew()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                random_state=self.random_state,
                                                                                stratify=self.y)

    def RandomForest(self, n_estimators, max_depth, cv, random_state):
        """
        Construct a Random Forest Classifier and compute the recall score.

        This method constructs the GridSearchCV object with a pipeline consisting
        of a KNNImputer and a RandomForestClassifier.
        In each cross validation split, first the KNNImputer is run on the training data
        in that fold to impute the missing values in BloodPressure, SkinThickness.
        From there, the RandomForestClassifier is trained on the training data with
        a combination of the hyperparameters specified in the param_grid (which consists
        of n_estimators and max_depth).  Finally, the KNNImputer and RandomForestClassifier
        (both fit on the training data) are used to predict the missing values in the test data
        and, from that, predict the class of the new test data.  The best model is determined
        the recall score.

        The Best Estimator, Cross Validation results, Feature Importances, Predictions, and
        Recall Score are stored in the RF, RF_CV_results, RF_feature_importances,
        RF_predictions, and RF_recall_score attributes, respectively.

        Arguments:
            n_estimators (list(int)): This is the number of trees in the forest, specified as a list.
                of integers.

            max_depth (list(int)): This is the maximum depth of each tree in the forest, specified as a list of
               integers

            cv (int): This is the number of folds to use in the cross validation.

            random_state (int): This is the seed to allow reproducibility.

        Returns:
            None
        """
        self.get_training_and_test_data()
        param_grid = {"RF__n_estimators": n_estimators, "RF__max_depth": max_depth}
        pipeline = Pipeline([("imputer", KNNImputer()),
                             ("RF", RandomForestClassifier(class_weight="balanced", random_state=random_state)
                              )
                             ]
                            )
        grid_search = GridSearchCV(pipeline,
                                   param_grid=param_grid,
                                   cv=cv, scoring="recall",
                                   )
        grid_search.fit(self.X_train, self.y_train)

        self.RF = grid_search.best_estimator_
        self.RF_CV_results = pd.DataFrame(grid_search.cv_results_)
        self.RF_predictions = self.RF.predict(self.X_test)
        self.RF_Recall_score = recall_score(y_true=self.y_test,
                                            y_pred=self.RF_predictions)

    def SupportVectorMachine(self, gamma, C, cv, random_state):
        """
        Construct a Support Vector Classifier with the rbf kernel and compute the f1 score.

        This method constructs the GridSearchCV object using the estimator
        consisting of the pipeline whose first step is the specified scaler and
        whose second step is the SVC.  Just as with the RandomForest method, the
        Best Estimator, Cross-Validation results, Feature Importances, Predictions,
        and Recall Score are stored in the SVM, SVM_CV_results, SVM_feature_importances,
        SVM_predictions, and SVM_f1_score attributes, respectively.

        Arguments:

            gamma (list(float)): This is the list of possible values of gamma for the GridSearchCV
                to search over. gamma (>0) controls the width of the Gaussian kernel.  The Gaussian
                kernel is k(x,y) = e^{-gamma ||x - y||^2}.  If gamma is close to 0, then the model is
                less complex, and the decision boundary is more smooth.  The larger gamma gets, the more
                complex the model becomes, and the more intricate the decision boundary becomes.

            C (list(float)): This is the list of possible values of C for the GridSearchCV to search over.
                C (>0) is a regularization parameter which controls the size of the coefficient in the SVM.

            cv (int): This is the number of folds in the cross validation.

            random_state (int): This is the seed to allow reproducibility.

        Returns:
            None
        """
        self.get_training_and_test_data()
        pipeline = Pipeline([("imputer", KNNImputer()),
                             ("scaler", MinMaxScaler()),
                             ("SVM", SVC(class_weight="balanced",
                                         random_state=random_state)
                              )
                             ]
                            )
        param_grid = {"SVM__gamma": gamma, "SVM__C": C}
        grid_search = GridSearchCV(pipeline, param_grid=param_grid,
                                   cv=cv, scoring="f1")
        grid_search.fit(self.X_train, self.y_train)

        self.SVM = grid_search.best_estimator_
        self.SVM_CV_results = pd.DataFrame(grid_search.cv_results_)
        self.SVM_predictions = self.SVM.predict(self.X_test)
        self.SVM_Recall_score = recall_score(y_true=self.y_test,
                                             y_pred=self.SVM_predictions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Fits models and makes prediction as to
                                     whether or not a sample has diabetes"""
                                     )
    parser.add_argument('n_estimators', type = list(int), help = """This is the number of trees in the
                        RandomForestClassifier, specified as a list for the GridSearchCV to search
                        over."""
                        )
    parser.add_argument('max_depth', type = list(int), help = """This is the maximum depth of each tree
                        in the RandomForestClassifier, specified as a list for the GridSearchCV to search
                        over."""
                        )
    parser.add_argument('cv_rf', type = int, help = """This is the number of folds in the cross validation
                        for the RandomForestClassifier.""")
    parser.add_argument('random_state_rf', type = int, help = """This is the seed of the randomization when
                        constructing the splits in the cross validation for the RandomForestClassifier."""
                        )
    parser.add_argument('gamma', type = list(float), help = """This is the list of possible values of 
                        gamma for the GridSearchCV to search over. gamma (>0) controls the width of 
                        the Gaussian kernel."""
                        )
    parser.add_argument('C', type = list(float), help = """This is the list of possible values of C 
                        for the GridSearchCV to search over.  C (>0) is a regularization parameter 
                        which controls the size of the coefficient in the SVM."""
                        )
    parser.add_argument('cv_svm', type = int, help = """This is the number of folds in the cross validation
                        for the SupportVectorClassifier."""
                        )
    parser.add_argument('random_state_svm', type = int, help = """This is the seed of the randomization when
                        constructing the splits in the cross validation for the SupportVectorClassifier."""
                        )
    args = parser.parse_args()
    d = diabetes_predictor()
    RF = d.RandomForest(args.n_estimators, args.max_depth, args.cv_rf, args.random_state_rf)
    print("X_train:{} \n".format(d.X_train))
    print("y_train:{} \n".format(d.y_train))
    print("X_test:{} \n".format(d.X_test))
    print("Random Forest best_estimator:{}".format(d.RF))
    print("Random Forest Cross-Validation results:{}".format(d.RF_CV_results))
    print("Random Forest Predictions:{}, length:{}".format(d.RF_predictions, len(d.RF_predictions)))
    print("y_test:{}".format(d.y_test))
    print("Random Forest Recall Score:{}".format(d.RF_Recall_score))
    SVM = d.SupportVectorMachine(args.gamma, args.C, args.cv_svm, args.random_state_svm)
    print("Support Vector Machine best_estimator:{}".format(d.SVM))
    print("Support Vector Machine Cross-Validation results:{}".format(d.SVM_CV_results))
    print("Random Forest Predictions:{}, length:{}".format(d.SVM_predictions, len(d.SVM_predictions)))
    print("y_test:{}".format(d.y_test))
    print("Support Vector Machine Recall Score:{}".format(d.SVM_Recall_score))



