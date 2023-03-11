# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import collections
import time
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("/home/karthikmrathod1999/my projects/data/df_diet_type.csv")
df.shape

df

df.title.nunique()

df.to_csv("data_for_recommendations.csv",index = False)

df_diet = df[['title','ingredients','diet_type','URL','description']]

df_diet.head(10)

# df_diet['title_encoded'] = df_diet.index

df_diet.head()

# title_codes = {}
# # n = 0
# for i in list(df_diet.title):
#     title_codes[i] = n
#     n+=1

def convert_tolist(x):
    y = eval(x)
    return y

df_diet.ingredients = df_diet.ingredients.apply(convert_tolist)

list_ingeredients = []
for i in df_diet['ingredients']:
    for j in range(len(i)):
        list_ingeredients.append(i[j])
len(list_ingeredients)

# import nltk


# # Use NLTK to tag each word with its part of speech
# pos_tags = nltk.pos_tag(list_ingeredients)

# # Filter out only the nouns from the list
# nouns = [word for (word, pos) in pos_tags if pos.startswith('N')]
# # list_nouns.append(nouns)
# len(nouns)

filtered_words = [word for word in list_ingeredients if len(word) > 2]

len(filtered_words)

counts = collections.Counter(filtered_words)

counts

def remove_ingerdients(x):
    filtered_list = []
    for i in x:
        try:
            if counts.get(i) > 2:
                filtered_list.append(i)
        except:
            pass
    return filtered_list

df_diet.ingredients = df_diet.ingredients.apply(remove_ingerdients)

df_diet.ingredients

def list_to_string(ingredient_list):
    return ' '.join(ingredient_list)

df_diet.ingredients = df_diet.ingredients.apply(list_to_string)

from gensim.corpora import Dictionary
from gensim.models.lsimodel import LsiModel
from gensim.similarities import MatrixSimilarity

df_diet_non_veg = df_diet[df_diet['diet_type'] == 'nveg'].reset_index(drop = True).copy()
df_diet_veg = df_diet[df_diet['diet_type'] == 'veg'].reset_index(drop = True).copy()
df_diet_vegan = df_diet[df_diet['diet_type'] == 'vegan'].reset_index(drop = True).copy()

"""# SUPRISE-ME"""

# Create a dictionary from the one-hot encoded ingredients
ingredient_dict = Dictionary(df_diet['ingredients'].apply(lambda x: x.split()))

# Create a bag-of-words representation of the ingredient data
ingredient_bow = [ingredient_dict.doc2bow(recipe.split()) for recipe in df_diet['ingredients']]

# Create an LSI topic model from the ingredient data
model_0 = LsiModel(corpus=ingredient_bow, id2word=ingredient_dict, num_topics=50)

# Create a matrix similarity index for the topic model
index_0 = MatrixSimilarity(model_0[ingredient_bow])

"""# NON-VEG"""

# Create a dictionary from the one-hot encoded ingredients
ingredient_dict = Dictionary(df_diet_non_veg['ingredients'].apply(lambda x: x.split()))

# Create a bag-of-words representation of the ingredient data
ingredient_bow = [ingredient_dict.doc2bow(recipe.split()) for recipe in df_diet_non_veg['ingredients']]

# Create an LSI topic model from the ingredient data
model_1 = LsiModel(corpus=ingredient_bow, id2word=ingredient_dict, num_topics=50)

# Create a matrix similarity index for the topic model
index_1 = MatrixSimilarity(model_1[ingredient_bow])

"""# VEG"""

# Create a dictionary from the one-hot encoded ingredients
ingredient_dict = Dictionary(df_diet_veg['ingredients'].apply(lambda x: x.split()))

# Create a bag-of-words representation of the ingredient data
ingredient_bow = [ingredient_dict.doc2bow(recipe.split()) for recipe in df_diet_veg['ingredients']]

# Create an LSI topic model from the ingredient data
model_2 = LsiModel(corpus=ingredient_bow, id2word=ingredient_dict, num_topics=50)

# Create a matrix similarity index for the topic model
index_2 = MatrixSimilarity(model_2[ingredient_bow])

"""# VEGAN"""

# Create a dictionary from the one-hot encoded ingredients
ingredient_dict = Dictionary(df_diet_vegan['ingredients'].apply(lambda x: x.split()))

# Create a bag-of-words representation of the ingredient data
ingredient_bow = [ingredient_dict.doc2bow(recipe.split()) for recipe in df_diet_vegan['ingredients']]

# Create an LSI topic model from the ingredient data
model_3 = LsiModel(corpus=ingredient_bow, id2word=ingredient_dict, num_topics=50)

# Create a matrix similarity index for the topic model
index_3 = MatrixSimilarity(model_3[ingredient_bow])

model_dict = {"suprise":0,"non-veg":1,"veg":2,"vegan":3}

# Define a function to get the top n similar recipes based on user ingredients
def get_recipes_by_ingredients(ingredients, n,model):
    if model_dict.get(model) == 0:
        # Convert the user ingredients to bag-of-words representation
        ingredient_bow = ingredient_dict.doc2bow(ingredients)

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
        ingredient_bow = ingredient_dict.doc2bow(ingredients)

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
        ingredient_bow = ingredient_dict.doc2bow(ingredients)

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
        ingredient_bow = ingredient_dict.doc2bow(ingredients)

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

# Test the function with a list of user ingredients and get the top 3 similar recipes

"""## to check ingerdiends
1. 
"""

ingredients = ['chicken', 'cheese','butter','garlic']
get_recipes_by_ingredients(ingredients, n=10,model ='suprise')

ingredients = ['butter','milk']
get_recipes_by_ingredients(['butter','milk'], n=10,model ='veg')

