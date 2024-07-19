import streamlit as st
import pandas as pd
import joblib

skincare_df = pd.read_csv('/Users/amina/projects/sephora-recommendation/skincare_df.csv', low_memory=False)
model = joblib.load('/Users/amina/projects/sephora-recommendation/sephora_model_1.pkl')

def get_recommendations(user_skin_type, user_skin_tone, user_budget_category):
    filtered_data = skincare_df[
        (skincare_df['price_category'] == user_budget_category) &
        (skincare_df['skin_type'] == user_skin_type) &
        (skincare_df['skin_tone'] == user_skin_tone)
    ].copy()

    if filtered_data.empty:
        return pd.DataFrame({'message': ['No recommendations found for the given criteria.']})

    feature_columns = ['skin_type_combination', 'skin_type_dry', 'skin_type_oily',
                       'price_category_low', 'price_category_medium', 'price_category_high',
                       'skin_tone_dark', 'skin_tone_deep', 'skin_tone_ebony', 'skin_tone_fair',
                       'skin_tone_fairLight', 'skin_tone_light', 'skin_tone_lightMedium',
                       'skin_tone_medium', 'skin_tone_mediumTan', 'skin_tone_notSureST',
                       'skin_tone_olive', 'skin_tone_porcelain', 'skin_tone_rich', 'skin_tone_tan']
    
    for col in feature_columns:
        if col not in filtered_data.columns:
            filtered_data[col] = 0


    filtered_data['predicted_rating'] = model.predict(filtered_data[feature_columns])

    recommendations = filtered_data.sort_values(by='predicted_rating', ascending=False).head(3)

    return recommendations[['product_name', 'predicted_rating']]


st.title('Skincare Product Recommendation System')

user_skin_type = st.selectbox('Select your skin type', ['Combination', 'Dry', 'Oily'])
user_skin_tone = st.selectbox('Select your skin tone', ['Fair', 'Medium', 'Deep'])
user_budget_category = st.selectbox('Select your budget category', ['Low', 'Medium', 'High'])

if st.button('Get Recommendations'):
    recommendations = get_recommendations(user_skin_type, user_skin_tone, user_budget_category)
    if 'message' in recommendations.columns:
        st.write(recommendations['message'][0])
    else:
        st.write('Top 3 Recommendations:')
        st.write(recommendations[['product_name', 'predicted_rating']])
