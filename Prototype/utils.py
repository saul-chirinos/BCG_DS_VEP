import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns

sns.set()


def __convert_date_dtype(data: pd.DataFrame):
    date_cols = [col for col in data if 'date' in col]
    data[date_cols] = data[date_cols].apply(lambda x: x.astype('datetime64'))
    return data


def extract_date_features(data: pd.DataFrame, col: str):
    data[f'{col}_year'] = data[col].dt.year
    data[f'{col}_month'] = data[col].dt.month
    data[f'{col}_dow'] = data[col].dt.day_of_week
    return data

def feature_eng(data: pd.DataFrame):
    data['total_cons'] = data[['cons_12m', 'cons_gas_12m']].sum(axis=1)
    data = __feature_eng_prices(data)

    # Getting bins for features with highest variance
    data = __get_percentiles(data, 'total_cons', 10)
    data = __get_percentiles(data, 'cons_12m', 10)
    data = __get_percentiles(data, 'cons_last_month', 2)
    data = __get_percentiles(data, 'forecast_cons_year', 2)
    data = __get_percentiles(data, 'forecast_cons_12m', 10)
    data = __get_percentiles(data, 'imp_cons', 2)
    data = __get_percentiles(data, 'net_margin', 10)
    data = __get_percentiles(data, 'forecast_meter_rent_12m', 4)
    data = __get_percentiles(data, 'margin_gross_pow_ele', 4)
    data = __get_percentiles(data, 'margin_net_pow_ele', 4)
    data = __get_percentiles(data, 'pow_max', 4)

    data = __forage_feature_eng(data)
    data = __feature_eng_dates(data)
    return data


def __feature_eng_dates(data: pd.DataFrame):
    # Get date durations
    data['contract_duration'] = data.date_end - data.date_activ
    data['next_renewal'] = data.date_renewal - data.date_activ
    data['days_since_modif_prod'] = data.date_modif_prod - data.date_activ

    data['contract_duration'] = data.contract_duration.dt.days
    data['next_renewal'] = data.next_renewal.dt.days
    data['days_since_modif_prod'] = data.days_since_modif_prod.dt.days

    data = extract_date_features(data, 'date_activ')
    data = extract_date_features(data, 'date_end')
    data = extract_date_features(data, 'date_modif_prod')
    data = extract_date_features(data, 'date_renewal')
    data = extract_date_features(data, 'price_date')
    return data


def __feature_eng_prices(data: pd.DataFrame):
    data['price_var'] = data[['price_off_peak_var', 'price_peak_var', 'price_mid_peak_var']].mean(axis=1)
    data['price_fix'] = data[['price_off_peak_fix', 'price_peak_fix', 'price_mid_peak_fix']].mean(axis=1)

    # Get price difference across periods
    # Energy
    data['price_p1_p2_var_diff'] = data.price_peak_var - data.price_off_peak_var
    data['price_p1_p3_var_diff'] = data.price_mid_peak_var - data.price_off_peak_var
    data['price_p2_p3_var_diff'] = data.price_mid_peak_var - data.price_peak_var

    # Power
    data['price_p1_p2_fix_diff'] = data.price_peak_fix - data.price_off_peak_fix
    data['price_p1_p3_fix_diff'] = data.price_mid_peak_fix - data.price_off_peak_fix
    data['price_p2_p3_fix_diff'] = data.price_mid_peak_fix - data.price_peak_fix

    return data


def __forage_feature_eng(data: pd.DataFrame):
    # Group off-peak prices by companies and month
    monthly_price_by_id = data.groupby(['id', 'price_date']).agg({
        'price_off_peak_var': 'mean',
        'price_off_peak_fix': 'mean'
    }).reset_index()

    # Get january and december prices
    jan_prices = monthly_price_by_id.groupby('id').first().reset_index()
    dec_prices = monthly_price_by_id.groupby('id').last().reset_index()

    # Calculate the difference
    dec_prices.rename(columns={'price_off_peak_var': 'dec_1' ,'price_off_peak_fix': 'dec_2'}, inplace=True)
    diff = pd.merge(dec_prices, jan_prices.drop(columns='price_date'), on='id')
    diff['offpeak_diff_dec_january_energy'] = diff['dec_1'] - diff['price_off_peak_var']
    diff['offpeak_diff_dec_january_power'] = diff['dec_2'] - diff['price_off_peak_fix']
    diff = diff[['id', 'offpeak_diff_dec_january_energy', 'offpeak_diff_dec_january_power']]

    data = pd.merge(data, diff, on='id')
    data.drop('id', axis=1, inplace=True)
    return data


def __get_percentiles(data: pd.DataFrame, col: str, q: int):
    data[f'{col}_binned'] = pd.qcut(data[col], q=q)
    return data


def get_sample_df():
    sample_df = pd.read_csv('Data/sample_df.csv')
    return sample_df


def load_model():
    model = joblib.load('Task_3/best_model.joblib')
    return model


def plot_probabilities(data: pd.DataFrame, col: st.columns):
    fig, ax = plt.subplots()
    ax.hist(data['Churn Probability'], bins=40)

    ax.set_title('Churn Probability Distribution')
    ax.set_xlabel('Probability')
    ax.set_ylabel('# Clients')
    col.pyplot(fig)


def preprocess(data: pd.DataFrame):
    copy_df = data.copy()
    sample_ids = copy_df['id']
    copy_df = __convert_date_dtype(copy_df)

    # Obj to category
    obj_cols = copy_df.select_dtypes(include='object').columns
    copy_df[obj_cols] = copy_df[obj_cols].apply(lambda x: x.astype('category'))

    copy_df['has_gas'] = copy_df['has_gas'].replace({'f': 0, 't': 1})

    return copy_df, sample_ids


if __name__ == '__main__':
    pass