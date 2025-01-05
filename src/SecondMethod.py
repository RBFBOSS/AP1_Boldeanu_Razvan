import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class SecondMethod:
    """
    Method 2:
    - Predict each component (Consum[MW], Productie[MW], etc.) with ID3 & Bayesian
    - Then compute Sold[MW] = Productie[MW] - Consum[MW]
    """
    @staticmethod
    def run(train_data, test_data, scale=False, iteration_tag=""):
        print(f"\n[Method 2] Predicting Sold[MW] - (ID3 + Bayesian) [Tag: {iteration_tag}]")

        train_df = train_data[['Data', 'Sold[MW]']].copy()
        test_df = test_data[['Data', 'Sold[MW]']].copy()

        for df in [train_df, test_df]:
            df['hour'] = df['Data'].dt.hour
            df['day'] = df['Data'].dt.day
            df['month'] = df['Data'].dt.month
            df['dayofweek'] = df['Data'].dt.dayofweek

        special_columns = [
            'Consum[MW]', 'Productie[MW]', 'Carbune[MW]',
            'Hidrocarburi[MW]', 'Ape[MW]', 'Nuclear[MW]',
            'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]'
        ]

        predicted_columns_id3 = {}
        predicted_columns_bayes = {}
        rmse_id3 = rmse_bayes = mae_id3 = mae_bayes = 9999

        actual_sold = test_data['Sold[MW]'].values

        # ID3 predictions
        for target in special_columns:
            print(f"\nTraining ID3 for column: {target}")

            x_train = train_df[['hour', 'day', 'month', 'dayofweek']]
            y_train = train_data[target]
            x_test = test_df[['hour', 'day', 'month', 'dayofweek']]

            if scale:
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)

            num_bins = 5
            try:
                train_data[f'{target}_Bucket'] = pd.qcut(y_train, q=num_bins, labels=False, duplicates='drop')
            except ValueError as e:
                print(f"Error in bucketing for {target}: {e}")
                continue

            bin_edges = pd.qcut(train_data[target], q=num_bins, retbins=True, duplicates='drop')[1]
            test_data[f'{target}_Bucket'] = pd.cut(
                test_data[target], bins=bin_edges, labels=False, include_lowest=True
            )
            test_data[f'{target}_Bucket'] = test_data[f'{target}_Bucket'].fillna(0).astype(int)
            y_train_buckets = train_data[f'{target}_Bucket']

            id3_model = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42)

            id3_model.fit(x_train, y_train_buckets)
            pred_buckets_id3 = id3_model.predict(x_test)

            bucket_means = train_data.groupby(f'{target}_Bucket')[target].mean().to_dict()
            pred_values_id3 = np.array([bucket_means.get(b, bucket_means[0]) for b in pred_buckets_id3])

            predicted_columns_id3[target] = pred_values_id3

        # Bayesian predictions
        for target in special_columns:
            print(f"\nTraining Bayesian for column: {target}")

            x_train = train_df[['hour', 'day', 'month', 'dayofweek']]
            y_train = train_data[target]
            x_test = test_df[['hour', 'day', 'month', 'dayofweek']]

            if scale:
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)

            num_bins = 5
            try:
                train_data[f'{target}_Bucket'] = pd.qcut(y_train, q=num_bins, labels=False, duplicates='drop')
            except ValueError as e:
                print(f"Error in bucketing for {target}: {e}")
                continue

            bin_edges = pd.qcut(train_data[target], q=num_bins, retbins=True, duplicates='drop')[1]
            test_data[f'{target}_Bucket'] = pd.cut(
                test_data[target], bins=bin_edges, labels=False, include_lowest=True
            )
            test_data[f'{target}_Bucket'] = test_data[f'{target}_Bucket'].fillna(0).astype(int)
            y_train_buckets = train_data[f'{target}_Bucket']

            bayes_model = GaussianNB()
            bayes_model.fit(x_train, y_train_buckets)
            pred_buckets_bayes = bayes_model.predict(x_test)

            bucket_means = train_data.groupby(f'{target}_Bucket')[target].mean().to_dict()
            pred_values_bayes = np.array([bucket_means.get(b, bucket_means[0]) for b in pred_buckets_bayes])

            predicted_columns_bayes[target] = pred_values_bayes

        if 'Productie[MW]' in predicted_columns_id3 and 'Consum[MW]' in predicted_columns_id3:
            predicted_sold_id3 = (
                    predicted_columns_id3['Productie[MW]'] - predicted_columns_id3['Consum[MW]']
            )
            mse_id3 = mean_squared_error(actual_sold, predicted_sold_id3)
            rmse_id3 = mse_id3 ** 0.5
            mae_id3 = mean_absolute_error(actual_sold, predicted_sold_id3)
            r2_id3 = r2_score(actual_sold, predicted_sold_id3)
            print(f"\n[Method 2] ID3 => Sold[MW]: RMSE={rmse_id3:.2f}, MAE={mae_id3:.2f}, R²={r2_id3:.2f}")
        else:
            print("Error: Missing 'Productie[MW]' or 'Consum[MW]' in ID3 predictions.")
            predicted_sold_id3 = None

        if 'Productie[MW]' in predicted_columns_bayes and 'Consum[MW]' in predicted_columns_bayes:
            predicted_sold_bayes = (
                    predicted_columns_bayes['Productie[MW]'] - predicted_columns_bayes['Consum[MW]']
            )
            mse_bayes = mean_squared_error(actual_sold, predicted_sold_bayes)
            rmse_bayes = mse_bayes ** 0.5
            mae_bayes = mean_absolute_error(actual_sold, predicted_sold_bayes)
            r2_bayes = r2_score(actual_sold, predicted_sold_bayes)
            print(f"[Method 2] Bayesian => Sold[MW]: RMSE={rmse_bayes:.2f}, MAE={mae_bayes:.2f}, R²={r2_bayes:.2f}")
        else:
            print("Error: Missing 'Productie[MW]' or 'Consum[MW]' in Bayesian predictions.")
            predicted_sold_bayes = None

        plt.figure(figsize=(14, 7))
        plt.plot(test_df['Data'], actual_sold, label='Real', color='blue')
        plt.plot(test_df['Data'], predicted_sold_id3, label='ID3', alpha=0.7, color='red')
        plt.plot(test_df['Data'], predicted_sold_bayes, label='Bayesian', alpha=0.7, color='green')
        plt.xlabel('Date')
        plt.ylabel('Sold[MW]')
        plt.title(f'[Second method] Predictions - {iteration_tag}')
        plt.legend()

        plt.show()

        return rmse_id3, rmse_bayes, mae_id3, mae_bayes
