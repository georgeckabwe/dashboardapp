
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import base64
from pandas._config.config import options
import streamlit as st
df = pd.read_csv('Mississauga Recent Listings last pull 09072021 Wrangled Data.csv')

st.set_page_config(layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:300px !important;
}
</style>
""", unsafe_allow_html=True)

# st.markdown('<p class="big-font">Hello World !!</p>', unsafe_allow_html=True)

st.write("""
# RECENT SALES DASHBOARD
""")


# def get_user_input():
#     #soldPrice = st.text_input(label = 'soldPrice', min_value = 50000,max_value = 9000000), 
#     beds = st.sidebar.slider(label = 'beds',min_value = 0, max_value = 8, value =1)
#     baths = st.sidebar.slider(label = 'baths',min_value = 0, max_value = 10, value = 1)
#     rooms = st.sidebar.slider(label = 'rooms',min_value = 0, max_value = 15, value = 5)
#     prop_style = st.sidebar.multiselect(label = 'Property Style', options = sorted(df['details.style'].unique()),default =sorted(df['details.style'].unique())[3] )
#     neighborhood = st.sidebar.selectbox(label = 'Neighborhoods', options = sorted(df['address.neighborhood'].unique()),format_func = str, index = 3)

#     #streetName = st.selectbox(label = 'streetName', options = sorted(df['address.streetName'].unique()))

#     # Store a dictionary into a variable
#     user_data = {#'soldPrice':soldPrice,
#                 'beds' : beds,
#                 'baths' : baths,
#                 'rooms' : rooms,
#                 'neighborhood':neighborhood,
#                 'prop_style' : prop_style
#                 #'streetName':streetName
#                 }

#     # Transform the data into a DataFrame
#     features = pd.DataFrame(data= user_data, )

#     return features

# # Show and initiate the input
# user_input = get_user_input()



# st.subheader('User Input:')
# st.write(user_input)


######################################
# Filtering Data
######################################
#beds = st.sidebar.radio(label = 'Beds',options =(df['details.numBedrooms']).unique(),index=0,format_func=int )
#beds = st.checkbox(label = 'beds',min_value = int(min(df['details.numBedrooms'])), max_value = int(max(df['details.numBedrooms'])), value =1)
beds =BEDS= st.sidebar.slider(label = 'beds',min_value = int(min(df['details.numBedrooms'])), max_value = int(max(df['details.numBedrooms'])), value =1)
baths =BATHS= st.sidebar.slider(label = 'baths',min_value = int(min(df['details.numBathrooms'])), max_value = int(max(df['details.numBathrooms'])), value = 1)
rooms =ROOMS= st.sidebar.slider(label = 'rooms',min_value = int(min(df['details.numRooms'])), max_value = int(max(df['details.numRooms'])), value = 5)
multi_style = STYLE = st.sidebar.multiselect(label = 'Property Style multi', options = sorted(df['details.style'].unique()),default =sorted(df['details.style'].unique())[3] )
multi_neighborhood =NEIGHBORHOOD = st.sidebar.multiselect(label = 'neighborhood multi', options = sorted(df['address.neighborhood'].unique()),default =sorted(df['address.neighborhood'].unique())[3] )
df_filtered = df[(df['details.style'].isin(multi_style))&(df['address.neighborhood'].isin(multi_neighborhood))&
                (df['details.numBedrooms'].isin(list(str(beds))))&(df['details.numBathrooms'].isin(list(str(baths))))&
                (df['details.numRooms'].isin(list(str(rooms))))]
st.header('Display Stats of Selected Property Style(s) and Neighborhood(s)')
st.write('Data Dimension: ' + str(df_filtered.shape[0]) + ' rows and ' + str(df_filtered.shape[1]) + ' columns.')

df_filtered = df_filtered[["address.neighborhood",'soldPrice',"address.streetName",'address.streetNumber','Date','ave_sqft','dollar_per_sqft',"details.numBedrooms","details.numBathrooms","details.numRooms"]]
df_filtered = df_filtered.rename(columns= {"address.neighborhood":'neighborhood',"address.streetName":'streetName',
                                'address.streetNumber':'streetNumber',"details.numBedrooms":'beds',"details.numBathrooms":'baths',
                                "details.numRooms":'rooms','ave_sqft':'sqft','dollar_per_sqft':'$/sqft'})
df_filtered = df_filtered.sort_values(by=(['streetNumber','soldPrice']), ascending=[True,True])
st.write(df_filtered, height = 0, width = 6)


# def filedownload(df_filtered):
#     csv = df_filtered.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
#     href = f'<a href="data:file/csv;base64,{b64}" download="{STYLE[0]} Properties in {NEIGHBORHOOD[0]} with {BEDS}-beds {BATHS}-baths {BEDS}-rooms.csv">Download CSV File</a>'
#     return href

# st.markdown(filedownload(df_filtered), unsafe_allow_html=True)

# print(user_input.reset_index(drop=True))
# # Filter out
# beds = user_input['beds']
# baths = user_input['baths']
# rooms = user_input['rooms']
# neighborhood = str(user_input['neighborhood'][0])
# prop_style = str(user_input['prop_style'][0])
# #streetName = str(user_input['streetName'])


# #ave_sqft 
# st.subheader('Filtered:')
# df_filtered = df[df['details.numBedrooms']==beds[0]]
# df_filtered = df_filtered[df_filtered['details.numBathrooms']==baths[0]]
# df_filtered = df_filtered[df_filtered['details.numRooms']==rooms[0]]
# df_filtered = df_filtered[df_filtered['address.neighborhood']==neighborhood]
# df_filtered = df_filtered[df_filtered['details.style']==prop_style]
# df_filtered = df_filtered[["address.neighborhood",'soldPrice',"address.streetName",'address.streetNumber','Date','ave_sqft','dollar_per_sqft',"details.numBedrooms","details.numBathrooms","details.numRooms"]]
# df_filtered = df_filtered.rename(columns= {"address.neighborhood":'neighborhood',"address.streetName":'streetName',
#                                 'address.streetNumber':'streetNumber',"details.numBedrooms":'beds',"details.numBathrooms":'baths',
#                                 "details.numRooms":'rooms','ave_sqft':'sqft','dollar_per_sqft':'$/sqft'})
# df_sorted = df_filtered.sort_values(by=(['streetNumber','soldPrice']), ascending=[True,True])
# st.write(df_sorted, height = 0, width = 6)


# Create a group 


df_group = df_filtered.groupby([df_filtered['streetName'],df_filtered['streetNumber']])
df_group_mean = df_group.mean()
df_group_count =df_group['soldPrice'].count()
df_group_mean['Count']= df_group['soldPrice'].count()

# Move column to front
df_group_mean = df_group_mean[ ['Count'] + [ col for col in df_group_mean.columns if col != 'Count' ] ]

st.subheader('Averages:')
st.write(df_group_mean)



#df_filtered = df[df['ave_sqft']==ave_sqft]
st.subheader('Descriptive Statistics:')
st.write(df_filtered.describe())

######################################
# Heatmap
######################################
st.set_option('deprecation.showPyplotGlobalUse', False)
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df_filtered.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(10, 2))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot()