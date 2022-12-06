import pandas as pd
import streamlit as st

import utils

st.set_page_config(
    page_title='Churn Prediction App',
    page_icon=':zap:',
    layout='wide',
    initial_sidebar_state='expanded'
)


def main():
    st.title('PowerCo :zap:')
    st.sidebar.subheader('Energy Client Churn Predictor')

    with st.sidebar.expander('Business problem'):
        st.write("""
                 This project aims to help PowerCo, a fictional company, retain their customers.
                 PowerCo is a major gas and electricity utility that supplies to corporate, SME
                 (Small & Medium Enterprise), and residential customers. A fair hypothesis is that
                 price changes affect customer churn. Therefore, it is helpful to know the
                 probability a customer is to churn at their current price, for which a good
                 predictive model could be useful. For the customers at risk of churning, i.e.
                 customers above the probability threshold, a discount of 20% will be offered to
                 help retain them.
                 """)

    with st.sidebar.expander('How to use'):
        st.write("""
                 Select a probability threshold below to indicate the cutoff point for discounts
                 to be given too.
                 """)

    prob_thresh = st.sidebar.slider('Probability threshold', 0.0, 1.0)

    data_col, feat_eng_col = st.columns(2)

    # Load data
    X = utils.get_sample_df()
    with data_col.expander('Client data :bar_chart:'):
        st.dataframe(X)

    # Preprocess and feature engineering
    X_modif, ids = utils.preprocess(X)
    X_modif = utils.feature_eng(X_modif)

    with feat_eng_col.expander('Features engineered :wrench:'):
        num_features = len(X_modif.columns[31:])
        st.markdown(f'**{num_features} new features** were created.')
        st.dataframe(X_modif.iloc[:5, 31:])
        st.caption('Showing only first 5 rows')

    # Predict
    model = utils.load_model()
    probabilities = model.predict_proba(X_modif)

    predictions_df = pd.DataFrame({'client_id':ids, 'Churn Probability': probabilities[:, 1]})\
        .groupby('client_id').mean().sort_values('client_id', ascending=False)

    st.markdown('---')
    st.subheader('Predictions')

    pred_col, at_risk_metrics_col, plot_col= st.columns(3)

    pred_col.dataframe(predictions_df)

    # Clients at risk metrics
    clients = utils.get_client_prices(predictions_df)
    clients_at_risk = clients[clients['Churn Probability'] >= prob_thresh]

    at_risk_metrics_col.subheader('Clients at risk')
    if len(clients_at_risk) < 1:
        at_risk_metrics_col.metric('Count', 0)
        at_risk_metrics_col.metric('Average churn probability', '-')
        at_risk_metrics_col.metric('Min churn probability', '-')
        at_risk_metrics_col.metric('Max churn probability', '-')
    else:
        avg_churn_prob = clients_at_risk['Churn Probability'].mean()*100
        avg_energy_price = clients_at_risk['price_var'].mean()
        avg_power_price = clients_at_risk['price_fix'].mean()

        at_risk_metrics_col.metric('Count', f'{len(clients_at_risk):,}')
        at_risk_metrics_col.metric('Average churn probability', f'{avg_churn_prob:.1f} %')
        at_risk_metrics_col.metric('Average price paid of energy', f'$ {avg_energy_price:.2f}')
        at_risk_metrics_col.metric('Average price paid of power', f'$ {avg_power_price:.2f}')

    # Probability plot
    utils.plot_probabilities(predictions_df, prob_thresh, plot_col)


if __name__ == '__main__':
    main()
