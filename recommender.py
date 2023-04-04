import streamlit as st
import numpy as np
import pandas as pd
import chardet
from sklearn.metrics.pairwise import cosine_similarity

word_embeddings = np.load('word_embeddings.npy')

with open("Articles.csv", 'rb') as f:
    result = chardet.detect(f.read())  
articles_df = pd.read_csv("Articles.csv", encoding=result['encoding'])


#  Recommending the Top 5 similar books

def recommendations(title):    
     
    # finding cosine similarity for the vectors
    cosine_similarities = cosine_similarity(word_embeddings, word_embeddings)
    # taking the title and book image link and store in new data frame called books
    books = articles_df[['Heading', 'NewsType', 'Article', 'Date']]
    #Reverse mapping of the index
    indices = pd.Series(articles_df.index, index = articles_df['Heading']).drop_duplicates()         
    idx = indices[title]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:11]
    book_indices = [i[0] for i in sim_scores]
    recommend = books.iloc[book_indices]
    return recommend
    for index, row in recommend.iterrows():
        response = row['Heading']
        print(response)


# Web UI
st.set_page_config(page_title="News Stories Recommender System", layout="wide")
# st.title(":blue[News Stories Recommender System]")
st.markdown("<h1 style='text-align: center; color: blue;'>News Stories Recommender System</h1>", unsafe_allow_html=True)

# st.subheader("Your one stop hub for the best news article recommendations")
st.markdown("<h3 style='text-align: center; color: blue;'>Your one stop hub for the best news article recommendations</h3>", unsafe_allow_html=True)


text_search = st.text_input("***Enter a story you would like recommendations for:***", value="")
st.markdown("""---""")

if text_search:
    recommendation_list = recommendations(text_search)
    for n_row, row in recommendation_list.iterrows():
        with st.container():
            
            st.caption(f"{row['Date'].strip()}")
            st.markdown(f"**{row['NewsType'].strip()}**")
            st.markdown(f"*{row['Heading'].strip()}*")
            with st.expander("Read article"):
                st.write(row['Article'].strip())
            st.markdown("""---""")

# # Show the cards
# N_cards_per_row = 3
# if text_search:
#     recommendation_list = recommendations(text_search)
#     for n_row, row in recommendation_list.iterrows():
#         i = n_row%N_cards_per_row
#         if i==0:
#             st.write("---")
#         cols = st.columns(N_cards_per_row, gap="large")
#         # draw the card
#         with cols[n_row%N_cards_per_row]:
#             st.caption(f"{row['Date'].strip()}")
#             st.markdown(f"**{row['NewsType'].strip()}**")
#             st.markdown(f"*{row['Heading'].strip()}*")
#             # st.markdown(f"**{row['Video']}**")

