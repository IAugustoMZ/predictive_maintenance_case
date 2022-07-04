"""
script for definition of utilities class for Machine Learning
modeling, experimentation and validation
"""
import os
import mlflow
import warnings
import pickle
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_absolute_error, r2_score,\
    confusion_matrix, recall_score, precision_score, fbeta_score,\
    make_scorer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor,\
    ExtraTreesRegressor, ExtraTreesClassifier, GradientBoostingClassifier,\
    GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from imblearn import pipeline as imb_pipe
from imblearn.over_sampling import SMOTE

# ignore warnings
warnings.filterwarnings('ignore')
np.random.seed(2)


class MLutils:

    # class attributes
    beta = 0.5          # beta for fbeta score
    REG_SCORE = 'r2'    # regression model selection score
    CLASS_SCORE = make_scorer(fbeta_score, beta=beta)  # classification model
    # selection score
    RANDOM_SEED = 2     # random seed
    model_alias = {
        'rlu': {
            'rf': RandomForestRegressor(random_state=RANDOM_SEED),
            'ext': ExtraTreesRegressor(random_state=RANDOM_SEED),
            'gb': GradientBoostingRegressor(random_state=RANDOM_SEED),
            'ada': AdaBoostRegressor(random_state=RANDOM_SEED),
            'xgb': XGBRegressor()
        },
        'class': {
            'lr': LogisticRegression(solver='saga'),
            'rf': RandomForestClassifier(random_state=RANDOM_SEED),
            'ext': ExtraTreesClassifier(random_state=RANDOM_SEED),
            'gb': GradientBoostingClassifier(random_state=RANDOM_SEED),
            'ada': AdaBoostClassifier(random_state=RANDOM_SEED),
            'xgb': XGBClassifier()
        },

    }                # model alias dictionary
    search_grid = {
        'rf': {
            'model__n_estimators': np.random.randint(10, 1000, 200),
            'model__max_depth': np.random.randint(2, 10, 5),
            'model__min_samples_leaf': np.random.uniform(0.01, 0.50, 100),
        },
        'ext': {
            'model__n_estimators': np.random.randint(10, 1000, 200),
            'model__max_depth': np.random.randint(2, 10, 5),
            'model__min_samples_leaf': np.random.uniform(0.01, 0.50, 100),
        },
        'gb': {
            'model__learning_rate': np.random.uniform((0.01, 0.1, 100)),
            'model__n_estimators': np.random.randint(10, 100, 200),
            'model__max_depth': np.random.randint(2, 10, 5)
        },
        'ada': {
            'model__learning_rate': np.random.uniform((0.01, 0.1, 100)),
            'model__n_estimators': np.random.randint(10, 1000, 200),
        },
        'xgb': {
            'model__eta': np.random.uniform((0.01, 0.1, 100)),
            'model__max_depth': np.random.randint(2, 10, 100),
            'model__lambda': np.random.uniform((0.01, 10, 100)),
            'model__alpha': np.random.uniform((0.01, 10, 100)),
        },
        'lr': {
            'model__penalty': ['l1', 'l2', 'elastic'],
            'model__alpha': np.random.uniform((0.01, 10, 100)),
            'model__l1_ratio': np.random.uniform(0.01, 1, 100)
        }

    }                # model hyperparameter dictionary

    def __init__(self) -> None:
        """
        methods for machine learning training, experimentation
        and validation
        """
        pass

    @staticmethod
    def calculate_regression_metrics(yreal: pd.DataFrame,
                                     ypred: np.array) -> list:
        """
        calculates the regression performance metrics

        Parameters
        ----------
        yreal : pd.DataFrame
            expected target
        ypred : np.array
            predicted target

        Returns
        -------
        list
            regression performance metrics list
        """

        # mean absolute error
        mae_train = mean_absolute_error(yreal, ypred)

        # r2 score
        r2 = r2_score(yreal, ypred)

        return mae_train, r2

    def calculate_classification_metrics(self,
                                         yreal: pd.DataFrame,
                                         ypred: np.array) -> list:
        """
        calculates the classification performance metrics

        Parameters
        ----------
        yreal : pd.DataFrame
            expected target
        ypred : np.array
            predicted target

        Returns
        -------
        list
            classification performance metrics list
        """

        # precision score
        precision = precision_score(yreal, ypred)

        # recall score
        recall = recall_score(yreal, ypred)

        # fbeta score
        fbeta = fbeta_score(yreal, ypred, beta=self.beta)

        return precision, recall, fbeta

    @staticmethod
    def regression_performance_fig(yreal: pd.DataFrame,
                                   ypred: np.array) -> object:
        """
        builds a figure of model's performance evaluation
        for regression

        Parameters
        ----------
        yreal : pd.DataFrame
            expected target
        ypred : np.array
            predicted target

        Returns
        -------
        object
            matplotlib figure
        """

        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(yreal, yreal, 'r-', label='Ideal Predictions')
        ax.plot(yreal, ypred, 'k.', label='True Predictions')
        ax.set_xlabel('Expected Values', size=14)
        ax.set_ylabel('Predicted Values', size=14)
        ax.set_title('Predicted vs Expected - Regression', size=16)
        ax.legend(loc='best', prop={'size': 14})

        return fig

    @staticmethod
    def classification_performance_fig(yreal: pd.DataFrame,
                                       ypred: np.array) -> object:
        """
        builds a figure of classification model performance

        Parameters
        ----------
        yreal : pd.DataFrame
            expected targets
        ypred : np.array
            predicted targets

        Returns
        -------
        fig
            matplotlib figure with confusion matrix and ROCAUC
        """

        # create figure
        fig = plt.figure(figsize=(8, 8))

        # create confusion matrix
        ax = fig.add_subplot(1, 2, 1)
        sns.heatmap(
            confusion_matrix(yreal, ypred),
            annot=True,
            cmap='Blues',
            ax=ax
        )
        ax.set_title('Confusion Matrix', size=16)

        return fig

    def select_features(self,
                        data: pd.DataFrame,
                        target: str,
                        n_folds: int = 10) -> None:
        """
        applies the recursive feature elimination
        with cross validation

        Parameters
        ----------
        data : pd.DataFrame
            train dataset
        target : str
            name of target variable
        n_folds : int, optional
            number of folds, by default 10
        """

        # saves the target name
        self.target = target

        if self.target == 'rlu':  # regression problem

            # create Kfold object
            self.cvs = KFold(n_splits=n_folds, random_state=self.RANDOM_SEED)

            # create selector
            selector = RFECV(
                estimator=LinearRegression(normalize=True),
                step=1,
                cv=self.cvs,
                scoring=self.REG_SCORE,
                verbose=0
            )

        else:  # classification problem
            self.cvs = StratifiedKFold(
                n_splits=n_folds,
                random_state=self.RANDOM_SEED
            )

            # create selector
            selector = RFECV(
                estimator=RandomForestClassifier(
                    random_state=self.RANDOM_SEED
                ),
                step=1,
                cv=self.cvs,
                scoring=self.CLASS_SCORE,
                verbose=0
            )

        # split predictors and targets
        self.x = data.drop([self.target], axis=1)
        self.y = data[[self.target]]

        # fit selector
        selector.fit(self.x, self.y)

        # save selected features
        self.selected_features = self.x.loc[:, selector.support_].columns

    def create_model_pipeline(self,
                              model_alias: str) -> None:
        """
        creates the modeling pipeline for the selected
        model based on an alias

        Parameters
        ----------
        model_alias : str
            model alias
        """

        # save model alias
        self.alias = model_alias

        # configure column transformer to project columns
        proj_cols = ColumnTransformer([
            ('column_proj', 'passthrough', self.selected_features)
        ], remainder='drop')

        # select the estimator based on the alias
        estimator = self.model_alias[self.target][self.alias]

        # configure the adequate type of pipeline
        if self.target == 'rlu':       # regression problem

            self.model_pipe = Pipeline([
                ('column_proj', proj_cols),
                ('scaler', RobustScaler(())),
                ('model', estimator)
            ])

        else:

            self.model_pipe = imb_pipe.Pipeline([
                ('column_proj', proj_cols),
                ('scaler', RobustScaler()),
                ('resample', SMOTE()),
                ('model', estimator)
            ])

    def model_fitting(self,
                      experiment_name: str) -> None:
        """
        trains the selected model and selects its hyperparameters
        by using cross-validation. The experimentation is
        integrated with mlflow

        Parameters
        ----------
        experiment_name : str
            name of experiment
        """

        # select scoring
        if self.target == 'rlu':
            scoring = self.REG_SCORE
        else:
            scoring = self.CLASS_SCORE

        # set experiment name
        mlflow.set_experiment(experiment_name)

        # start run for the selected model
        with mlflow.start_run(run_name=f'{self.alias}_model_{self.target}'):

            # select search grid
            serch_grid = self.search_grid[self.alias]

            # create hyperparameter tuner
            tuner = RandomizedSearchCV(
                estimator=self.model_pipe,
                param_distributions=serch_grid,
                n_iter=50,
                cv=self.cvs,
                scoring=scoring,
                random_state=self.RANDOM_SEED,
                n_jobs=-1,
                refit=True
            )

            # fit tuner
            tuner.fit(self.x, self.y.values.ravel())

            # log best metric
            mlflow.log_metric(
                f'best_{scoring}',
                tuner.best_score_
            )

            # log params
            mlflow.log_params(tuner.best_params_)

            # log model
            sign = infer_signature(
                self.x.head(1),
                tuner.best_estimator_.predict(self.x.head(1))
            )

            mlflow.sklearn.log_model(
                tuner.best_estimator_,
                'model',
                input_example=self.x.head(1),
                signature=sign
            )

            # calculate performance metrics
            yhat = tuner.best_estimator_.predict(self.x)

            if self.target == 'rlu':

                mae, r2 = self.calculate_regression_metrics(
                    yreal=self.y,
                    ypred=yhat
                )

                # log metrics
                mlflow.log_metrics({
                    'mae': mae,
                    'r2': r2
                })

                print(f'Mean Absolute Error: {mae}')
                print(f'R2 Score: {r2}')

                # create figure of model performance
                fig = self.regression_performance_fig(
                    yreal=self.y,
                    ypred=yhat)

                # log performance figure
                mlflow.log_figure(
                    fig,
                    f'performance_{self.alias}_{self.target}.png'
                )

            else:

                recall, prec, fbeta = self.calculate_classification_metrics(
                    yreal=self.y,
                    ypred=yhat
                )

                # log metrics
                mlflow.log_metrics({
                    'recall': recall,
                    'precision': prec,
                    'fbeta': fbeta
                })

                print(f'Recall Score: {recall}')
                print(f'Precision Score: {prec}')
                print(f'Fbeta: {fbeta}')

                # create figure of model performance
                fig = self.classification_performance_fig(
                    yreal=self.y,
                    ypred=yhat)

                # log performance figure
                mlflow.log_figure(
                    fig,
                    f'performance_{self.alias}_{self.target}.png'
                )

        return fig
