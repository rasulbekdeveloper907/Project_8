import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# --------------------------
# 1. Missing Value Imputer
# --------------------------
class MissingValueImputer:
    def __init__(self, df, target_col='kilometer'):
        self.df = df
        self.target_col = target_col

    def fill(self):
        # Kategorik ustunlar
        categorical_cols = self.df.select_dtypes(include='object').columns
        for col in categorical_cols:
            mode_value = self.df[col].mode()[0]
            self.df[col].fillna(mode_value, inplace=True)

        # Raqamli ustunlar (target ustun ham inobatga olinadi)
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            if self.df[col].isnull().sum() > 0:
                mean_value = self.df[col].mean()
                self.df[col].fillna(mean_value, inplace=True)
        return self

    def get_df(self):
        return self.df

# --------------------------
# 2. Encoder
# --------------------------
class Encoder:
    def __init__(self, df):
        self.df = df
        self.encoder = LabelEncoder()

    def encodla(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                if self.df[col].nunique() <= 5:
                    # One-hot encoding
                    dummies = pd.get_dummies(self.df[col], prefix=col, dtype=int)
                    self.df = pd.concat([self.df.drop(columns=[col]), dummies], axis=1)
                else:
                    # Label encoding
                    self.df[col] = self.encoder.fit_transform(self.df[col])
        return self

    def get_df(self):
        return self.df 

# --------------------------
# 3. Scaler (MinMaxScaler)
# --------------------------
class Scaler:
    def __init__(self, df, target_col='kilometer'):
        self.df = df
        self.scaler = MinMaxScaler()
        self.target_col = target_col

    def scaling_qil(self):
        # Target ustunni scal qilmaslik
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)

        self.df[numeric_cols] = pd.DataFrame(
            self.scaler.fit_transform(self.df[numeric_cols]),
            columns=numeric_cols,
            index=self.df.index
        )
        return self

    def get_df(self):
        return self.df
