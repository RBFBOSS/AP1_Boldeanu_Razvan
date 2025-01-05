import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def data_init():
    train_files = [
        r"../data/2014-2016.xlsx",
        r"../data/2017-2019.xlsx",
        r"../data/2020-2022.xlsx",
        r"../data/2023-2024-no-dec.xlsx"
    ]
    test_file = r"../data/2024-dec.xlsx"

    train_dfs = []
    for fpath in train_files:
        df = pd.read_excel(fpath, sheet_name="Grafic SEN")
        train_dfs.append(df)

    train_data = pd.concat(train_dfs, ignore_index=True)
    test_data = pd.read_excel(test_file, sheet_name="Grafic SEN")

    print("TEST(passed): First rows (TRAIN):")
    print(train_data.head())
    print("\nTRAIN Info:")
    print(train_data.info())

    print("\nTEST(passed): First rows (TEST):")
    print(test_data.head())
    print("\nTEST Info:")
    print(test_data.info())

    numeric_columns = [
        'Consum[MW]', 'Productie[MW]', 'Carbune[MW]',
        'Hidrocarburi[MW]', 'Ape[MW]', 'Nuclear[MW]',
        'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]', 'Sold[MW]'
    ]
    for col in numeric_columns:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
        test_data[col] = pd.to_numeric(test_data[col], errors='coerce')

    train_data['Data'] = pd.to_datetime(train_data['Data'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
    test_data['Data'] = pd.to_datetime(test_data['Data'], format='%d-%m-%Y %H:%M:%S', errors='coerce')

    if 'Medie Consum[MW]' in train_data.columns:
        train_data['Medie Consum[MW]'] = (
            train_data['Medie Consum[MW]'].astype(str).str.replace(',', '.')
        )
        train_data['Medie Consum[MW]'] = pd.to_numeric(train_data['Medie Consum[MW]'], errors='coerce')

    if 'Medie Consum[MW]' in test_data.columns:
        test_data['Medie Consum[MW]'] = (
            test_data['Medie Consum[MW]'].astype(str).str.replace(',', '.')
        )
        test_data['Medie Consum[MW]'] = pd.to_numeric(test_data['Medie Consum[MW]'], errors='coerce')

    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    print("\nPost preprocessing TRAIN:")
    print(train_data.info())
    print(train_data.head())

    print("\nPost preprocessing TEST:")
    print(test_data.info())
    print(test_data.head())

    return train_data, test_data


if __name__ == "__main__":
    training_data, testing_data = data_init()
