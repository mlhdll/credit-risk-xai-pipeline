"""
Credit Risk Prediction - Data Preprocessing & Pipeline Architecture
Zero Data Leakage guarantee using scikit-learn ColumnTransformer & Pipeline.
"""

from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-specific Financial Feature Engineering:
    - cred_hist_to_age_ratio: Proportion of adult life with credit history
    - income_per_emp_year: Income stability indicator
    - loan_to_income_calc: Calculated loan amount to annual income ratio
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # Credit history length relative to adult age (age - 18)
        adult_age = np.maximum(X_out['person_age'] - 18, 1)
        X_out['cred_hist_to_age_ratio'] = np.clip(
            X_out['cb_person_cred_hist_length'] / adult_age, 0, 1
        )
        
        # Income per employment year (income stability)
        emp_len = np.maximum(X_out['person_emp_length'].fillna(0), 1)
        X_out['income_per_emp_year'] = X_out['person_income'] / emp_len
        
        # Loan to annual income ratio
        X_out['loan_to_income_calc'] = X_out['loan_amnt'] / np.maximum(X_out['person_income'], 1)

        return X_out


def get_preprocessor() -> Tuple[ColumnTransformer, List[str], List[str], List[str]]:
    """
    Builds a robust, zero-leakage ColumnTransformer for Credit Risk dataset.
    """
    numeric_features = [
        'person_age',
        'person_income',
        'person_emp_length',
        'loan_amnt',
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_cred_hist_length',
        'cred_hist_to_age_ratio',
        'income_per_emp_year',
        'loan_to_income_calc'
    ]

    ordinal_features = ['loan_grade']
    loan_grade_order = [['A', 'B', 'C', 'D', 'E', 'F', 'G']]

    nominal_features = [
        'person_home_ownership',
        'loan_intent',
        'cb_person_default_on_file'
    ]

    # Numeric Pipeline: Median Imputation + Standard Scaling
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Ordinal Pipeline: Grade encoding A->0 ... G->6
    ordinal_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=loan_grade_order))
    ])

    # Nominal Categorical Pipeline: Most frequent imputation + OneHotEncoding
    nominal_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('ord', ordinal_pipeline, ordinal_features),
            ('cat', nominal_pipeline, nominal_features)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    return preprocessor, numeric_features, ordinal_features, nominal_features
