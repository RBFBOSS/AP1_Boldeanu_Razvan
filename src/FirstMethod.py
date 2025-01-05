import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class FirstMethod:
    @staticmethod
    def run(train_data, test_data, scale=False, iteration_tag=""):
        print(f"\n[Method 1] Predicting Sold[MW] - (ID3 + Bayesian) [Tag: {iteration_tag}]")

        train_df = train_data[['Data', 'Sold[MW]']].copy()
        test_df = test_data[['Data', 'Sold[MW]']].copy()

        for df in [train_df, test_df]:
            df['hour'] = df['Data'].dt.hour
            df['day'] = df['Data'].dt.day
            df['month'] = df['Data'].dt.month
            df['dayofweek'] = df['Data'].dt.dayofweek

        x_train = train_df[['hour', 'day', 'month', 'dayofweek']]
        x_test = test_df[['hour', 'day', 'month', 'dayofweek']]
        y_test = test_df['Sold[MW]'].values

        num_bins = 5
        train_df['Sold_Bucket'] = (
            train_df['Sold[MW]']
            .pipe(lambda s: s.dropna())
            .pipe(lambda s: s.rank(method='first'))
        )

        train_df['Sold_Bucket'] = \
            train_df['Sold[MW]'].pipe(lambda s: s.dropna())

        train_df['Sold_Bucket'] = \
            train_df['Sold[MW]'].transform(
                lambda s: pd.qcut(s, q=num_bins, labels=False, duplicates='drop')
            )

        bin_edges = pd.qcut(train_df['Sold[MW]'], q=num_bins, retbins=True, duplicates='drop')[1]
        test_df['Sold_Bucket'] = pd.cut(test_df['Sold[MW]'], bins=bin_edges, labels=False, include_lowest=True)
        test_df['Sold_Bucket'] = test_df['Sold_Bucket'].fillna(0).astype(int)
        y_train_buckets = train_df['Sold_Bucket']

        # Optional scaling
        if scale:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

        # ID3
        print("\n[Method 1] Using default ID3 parameters...")
        id3_model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)

        id3_model.fit(x_train, y_train_buckets)
        pred_buckets_id3 = id3_model.predict(x_test)

        bucket_means = train_df.groupby('Sold_Bucket')['Sold[MW]'].mean().to_dict()
        pred_values_id3 = np.array([bucket_means.get(b, bucket_means[0]) for b in pred_buckets_id3])

        # Bayesian
        bayes_model = GaussianNB()
        bayes_model.fit(x_train, y_train_buckets)
        pred_buckets_bayes = bayes_model.predict(x_test)
        pred_values_bayes = np.array([bucket_means.get(b, bucket_means[0]) for b in pred_buckets_bayes])

        # Evaluate
        mse_id3 = mean_squared_error(y_test, pred_values_id3)
        rmse_id3 = mse_id3 ** 0.5
        mae_id3 = mean_absolute_error(y_test, pred_values_id3)
        r2_id3 = r2_score(y_test, pred_values_id3)

        mse_bayes = mean_squared_error(y_test, pred_values_bayes)
        rmse_bayes = mse_bayes ** 0.5
        mae_bayes = mean_absolute_error(y_test, pred_values_bayes)
        r2_bayes = r2_score(y_test, pred_values_bayes)

        print(f"\n[Method 1] ID3 Performance:")
        print(f"RMSE: {rmse_id3:.2f}, MAE: {mae_id3:.2f}, R²: {r2_id3:.2f}")

        print(f"\n[Method 1] Bayesian Performance:")
        print(f"RMSE: {rmse_bayes:.2f}, MAE: {mae_bayes:.2f}, R²: {r2_bayes:.2f}")

        # Save plot
        plt.figure(figsize=(14, 7))
        plt.plot(test_df['Data'], y_test, label='Real Values', color='blue')
        plt.plot(test_df['Data'], pred_values_id3, label='ID3 Predictions', alpha=0.7, color='orange')
        plt.plot(test_df['Data'], pred_values_bayes, label='Bayesian Predictions', alpha=0.7, color='green')
        plt.xlabel('Date')
        plt.ylabel('Sold[MW]')
        plt.title(f'[Method 1] Real vs. ID3 & Bayesian Predictions - {iteration_tag}')
        plt.legend()

        plt.show()

        return rmse_id3, rmse_bayes, mae_id3, mae_bayes
