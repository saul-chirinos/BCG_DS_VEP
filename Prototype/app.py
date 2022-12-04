import pandas as pd
import streamlit as st
import utils

st.set_page_config(layout='wide')


def main():
    st.sidebar.title('PowerCo :zap:')
    st.sidebar.subheader('Energy Client Churn Predictor')

    st.subheader('Business Problem')
    st.write("""
             This project aims to help PowerCo, a fictional company, retain their customers.
             PowerCo is a major gas and electricity utility that supplies to corporate, SME
             (Small & Medium Enterprise), and residential customers. A fair hypothesis is that
             price changes affect customer churn. Therefore, it is helpful to know the
             probability a customer is to churn at their current price, for which a good
             predictive model could be useful. For the customers at risk of churning, a
             discount of 20% will be offered to help retain them.
             """)

    st.subheader('New Data')
    X = utils.get_sample_df()
    X = X.sort_values(['id', 'price_date'], ascending=[0, 1]).reset_index(drop=True)
    st.dataframe(X)

    if st.sidebar.button('Predict'):
        # Preprocess and feature engineering
        X_modif, ids = utils.preprocess(X)
        X_modif = utils.feature_eng(X_modif)

        col1, col2 = st.columns(2)

        col1.subheader('Features Engineered :wrench:')
        num_features = len(X_modif.columns[31:])
        col1.markdown(f'**{num_features} new features** were created.')
        col1.dataframe(X_modif.iloc[:, 31:])

        st.sidebar.subheader('Model Predictions :dart:')
        model = utils.load_model()
        probabilities = model.predict_proba(X_modif)

        predictions_df = pd.DataFrame({'client_id':ids, 'Churn Probability': probabilities[:, 1]})\
            .groupby('client_id').mean().sort_values('client_id', ascending=False)
        st.sidebar.dataframe(predictions_df)

        # Probability plot
        utils.plot_probabilities(predictions_df, col2)


if __name__ == '__main__':
    main()
