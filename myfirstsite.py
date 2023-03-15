# import the streamlit library
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import textwrap
warnings.filterwarnings('ignore')
from gensim.corpora import Dictionary
from gensim.models.lsimodel import LsiModel
from gensim.similarities import MatrixSimilarity

# importing the requried dataset
df_diet = pd.read_csv("Recommendation_data.csv")

# This code creates three dataframes containing recipes for different diet types - non-vegetarian, vegetarian, and vegan
df_diet_non_veg = df_diet[df_diet['diet_type'] == 'nveg'].reset_index(drop = True).copy()
df_diet_veg = df_diet[df_diet['diet_type'] == 'veg'].reset_index(drop = True).copy()
df_diet_vegan = df_diet[df_diet['diet_type'] == 'vegan'].reset_index(drop = True).copy()

# This code loads pre-trained LSI models, similarity indices, and dictionaries for four different diet types (0: general, 1: non-vegetarian, 2: vegetarian, 3: vegan)
# that were saved previously using the save() method.
model_0 = LsiModel.load('models /model_0')
index_0 = MatrixSimilarity.load('models /index_0')
ingredient_dict_0 = Dictionary.load('models /ingredient_dict_0.dict')
model_1 = LsiModel.load('models /model_1')
index_1 = MatrixSimilarity.load('models /index_1')
ingredient_dict_1 = Dictionary.load('models /ingredient_dict_1.dict')
model_2 = LsiModel.load('models /model_2')
index_2 = MatrixSimilarity.load('models /index_2')
ingredient_dict_2 = Dictionary.load('models /ingredient_dict_2.dict')
model_3 = LsiModel.load('models /model_3')
index_3 = MatrixSimilarity.load('models /index_3')
ingredient_dict_3 = Dictionary.load('models /ingredient_dict_3.dict')

# This dictionary will be used to select the appropriate model for recommendation
model_dict = {"suprise_me":0,"non-veg":1,"veg":2,"vegan":3}

# Define a function to get the top n similar recipes based on user ingredients
def get_recipes_by_ingredients(ingredients, n,model):
    """
    Given a list of ingredients, return a dataframe of n recommended recipes based on the similarity 
    of the ingredients to those in the dataset, for a specific diet type.

    Args:
    - ingredients (list): A list of ingredients.
    - n (int): The number of recommended recipes to return.
    - model (str): A string indicating the diet type to consider. 
                   Possible values: 'suprise', 'non-veg', 'veg', 'vegan'.

    Returns:
    - recommends (DataFrame): A Pandas DataFrame containing the n recommended recipes.
    """
    if model_dict.get(model) == 0:
        # Convert the user ingredients to bag-of-words representation
        ingredient_bow = ingredient_dict_0.doc2bow(ingredients)

        # Get the topic distribution for the user ingredients
        user_topics = model_0[ingredient_bow]

        # Get the similarity scores for all recipes based on user ingredients
        sim_scores = index_0[user_topics]

        # Sort the recipes by similarity score
        sim_scores = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)

        # Get the top n similar recipes
        sim_scores = sim_scores[0:n]
        recipe_indices = [i[0] for i in sim_scores]
        recommends = df_diet.iloc[recipe_indices]
        recommends = recommends.reset_index(drop = True)
        return recommends
    if model_dict.get(model) == 1:
        # Convert the user ingredients to bag-of-words representation
        ingredient_bow = ingredient_dict_1.doc2bow(ingredients)

        # Get the topic distribution for the user ingredients
        user_topics = model_1[ingredient_bow]

        # Get the similarity scores for all recipes based on user ingredients
        sim_scores = index_1[user_topics]

        # Sort the recipes by similarity score
        sim_scores = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)

        # Get the top n similar recipes
        sim_scores = sim_scores[0:n]
        recipe_indices = [i[0] for i in sim_scores]
        recommends = df_diet_non_veg.iloc[recipe_indices]
        recommends = recommends.reset_index(drop = True)
        return recommends
    if model_dict.get(model) == 2:
        # Convert the user ingredients to bag-of-words representation
        ingredient_bow = ingredient_dict_2.doc2bow(ingredients)

        # Get the topic distribution for the user ingredients
        user_topics = model_2[ingredient_bow]

        # Get the similarity scores for all recipes based on user ingredients
        sim_scores = index_2[user_topics]

        # Sort the recipes by similarity score
        sim_scores = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)

        # Get the top n similar recipes
        sim_scores = sim_scores[0:n]
        recipe_indices = [i[0] for i in sim_scores]
        recommends = df_diet_veg.iloc[recipe_indices]
        recommends = recommends.reset_index(drop = True)
        return recommends
    if model_dict.get(model) == 3:
        # Convert the user ingredients to bag-of-words representation
        ingredient_bow = ingredient_dict_3.doc2bow(ingredients)

        # Get the topic distribution for the user ingredients
        user_topics = model_3[ingredient_bow]

        # Get the similarity scores for all recipes based on user ingredients
        sim_scores = index_3[user_topics]

        # Sort the recipes by similarity score
        sim_scores = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)

        # Get the top n similar recipes
        sim_scores = sim_scores[0:n]
        recipe_indices = [i[0] for i in sim_scores]
        recommends = df_diet_vegan.iloc[recipe_indices]
        recommends = recommends.reset_index(drop = True)
        return recommends

"""
1.The code begins by importing Streamlit and defining the title of the web app as "The Sentient Chef".

2.The user can input a list of ingredients separated by spaces and select a diet type from a dropdown menu.

3.The function get_recipes_by_ingredients is called with the list of ingredients and the selected diet type, and it returns a DataFrame with 10 recipe suggestions.

4.If the user clicks the "Submit" button, the code displays the title, description, and URL of each recipe suggestion in a separate container using Streamlit's container function.

5.The title and description are displayed using Streamlit's header and text functions, respectively, and the URL is displayed as a hyperlink using Streamlit's markdown syntax.

Overall, the code provides a simple interface for users to get recipe suggestions based on their ingredients and dietary preferences.
"""

st.title('The Sentient Chef')

weight = st.text_input("input ingerdients with spaces ")
diet = st.selectbox("Diet Type: ",
                     ['suprise_me', 'non-veg', 'veg','vegan'])

list_ = weight.split(" ")
df = get_recipes_by_ingredients(list_, n=10,model =diet)

if(st.button('Submit')):
    with st.container():
        st.header(df.title[0])
        st.text(textwrap.fill(df.description[0], width=80))
        st.write(f"We can teleport you to the webpage if you like this recipe [teleport me]({df.URL[0]})")
    with st.container():
        st.header(df.title[1])
        st.text(textwrap.fill(df.description[1], width=80))
        st.write(f"We can teleport you to the webpage if you like this recipe [teleport me]({df.URL[1]})")
    with st.container():
        st.header(df.title[2])
        st.text(textwrap.fill(df.description[2], width=80))
        st.write(f"We can teleport you to the webpage if you like this recipe [teleport me]({df.URL[2]})")
    with st.container():
        st.header(df.title[3])
        st.text(textwrap.fill(df.description[3], width=80))
        st.write(f"We can teleport you to the webpage if you like this recipe [teleport me]({df.URL[3]})")
    with st.container():
        st.header(df.title[4])
        st.text(textwrap.fill(df.description[4], width=80))
        st.write(f"We can teleport you to the webpage if you like this recipe [teleport me]({df.URL[4]})")
    with st.container():
        st.header(df.title[5])
        st.text(textwrap.fill(df.description[5], width=80))
        st.write(f"We can teleport you to the webpage if you like this recipe [teleport me]({df.URL[5]})")
    with st.container():
        st.header(df.title[6])
        st.text(textwrap.fill(df.description[6], width=80))
        st.write(f"We can teleport you to the webpage if you like this recipe [teleport me]({df.URL[6]})")
    with st.container():
        st.header(df.title[7])
        st.text(textwrap.fill(df.description[7], width=80))
        st.write(f"We can teleport you to the webpage if you like this recipe [teleport me]({df.URL[7]})")
    with st.container():
        st.header(df.title[8])
        st.text(textwrap.fill(df.description[8], width=80))
        st.write(f"We can teleport you to the webpage if you like this recipe [teleport me]({df.URL[8]})")
    with st.container():
        st.header(df.title[9])
        st.text(textwrap.fill(df.description[9], width=80))
        st.write(f"We can teleport you to the webpage if you like this recipe [teleport me]({df.URL[9]})")


    

