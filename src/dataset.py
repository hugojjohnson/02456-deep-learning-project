import glob
import os
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Returns a dictionary with frames ['wind_speed_19_n', 'wind_speed_13_n', 'wind_speed_11_n', 'wind_speed_15_n', 'wind_speed_17_n'].
def load_dataframes():
    pickle_path = os.path.join('data', 'interim', 'dataframes.pkl')
    if os.path.exists(pickle_path):
        # Load the dictionary from the file
        with open(pickle_path, "rb") as f:
            loaded_dict = pickle.load(f)
        print("Loaded from pickle")
        return loaded_dict


    # Find all CSV files in the specified path
    csv_files = glob.glob(os.path.join('data', 'raw', 'wind_speed_*.csv'))

    # Read each CSV file into a DataFrame and store in a dictionary
    dataframes = {}
    for file in csv_files:
        df_name = os.path.basename(file).replace(".csv", "")  # Extract file name without extension
        dataframes[df_name] = pd.read_csv(file)

    # Display keys (file names) to ensure everything loaded correctly
    # print("Loaded datasets:", list(dataframes.keys()))
    with open(pickle_path, "wb") as f:
        pickle.dump(dataframes, f)
    print("Generated and saved to pickle")
    return dataframes

def filter_df(input_features, output_features):
    # Læs og kombiner alle datasæt i én DataFrame
    dataframes = load_dataframes()
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Filtrer data til kun at indeholde de ønskede kolonner
    filtered_df = combined_df[input_features + output_features]
    return filtered_df

def normalize_data():
    # Udvælg de input features, du ønsker at anvende
    input_features = ['beta1', 'beta2', 'beta3', 'Theta', 'omega_r', 'Vwx']
    # Udvælg de output features, du ønsker at forudsige
    output_features = ['Mz1', 'Mz2', 'Mz3']

    filtered_df = filter_df(input_features, output_features)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()


    X = scaler_X.fit_transform(filtered_df[input_features])
    y = scaler_y.fit_transform(filtered_df[output_features])
    return X, y

def load_and_split_data():
    # # =====
    # # Normaliser dataene
    # scaler = MinMaxScaler()
    # scaled_data = scaler.fit_transform(filtered_df)
    # combined_df_scaled = pd.DataFrame(scaled_data, columns=filtered_df.columns)

    # X = combined_df_scaled[input_features]
    # y = combined_df_scaled[output_features]
    # # =====
    X, Y = load_and_split_data()

    # Split data i træning og test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Konverter til 3D-format for LSTM (samples, timesteps, features)
    # Her antages en enkelt timestep, men du kan øge det, hvis du vil have flere tidssteg
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y