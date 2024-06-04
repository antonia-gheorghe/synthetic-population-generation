import random
import numpy as np
import geopandas as gpd
import pandas as pd
import os
import json
import plotly as py
import math
from collections import Counter
import itertools
import InputData as ID

def getkeys(df):
    groups = list(df.columns)[2:] #drop first two columns: areacode and total
    return groups

def get_weights(df, area):
    df = df[df['geography code'] == area]
    groups = list(df.columns)[2:] #drop first two columns: areacode and total
    values = df.values.flatten().tolist()[1:]
    total = values.pop(0)
    weights = [x / total for x in values]
    return weights
    # return np.random.choice(groups, size=size, replace=True, p=weights).tolist()

def get_weighted_samples_by_age_sex(df, age, sex, size):
    df = df[[col for col in df.columns if age in col]]
    df = df[[col for col in df.columns if sex in col]]
    groups = list(df.columns)
    values = df.values.flatten().tolist()
    total = sum(values)
    weights = [x / total for x in values]
    return np.random.choice(groups, size=size, replace=True, p=weights).tolist()

def get_weight(key, df):
    value = getdictionary(df).get(key)
    total = df.values.flatten().tolist()[1:].pop(0)
    return value/total

# def getdictionary(df, area):
#     df = df[df['geography code'] == area]
#     if 'total' in df.columns:
#         df = df.iloc[:, 1:] #drop total column
#     dic = {}
#     for index, row in df.iterrows():
#         for index, column in enumerate(df.columns):
#             if index==0:
#                 continue
#             dic[column] = int(row[column])
#     return dic
def getdictionary(df, area):
    df = df[df['geography code'] == area]
    dic = df.iloc[0].to_dict()
    dic.pop('geography code')
    dic.pop('total')
    return dic

def aggregate_age(data):
    # Define age brackets
    child_ages = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19']
    adult_ages = ['20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64']
    elder_ages = ['65_69', '70_74', '75_79', '80_84', '85+']

    aggregate = {}
    for key in data.keys():
        aggregate[key.split(' ')[0] + ' child ' + key.split(' ')[2]] = 0
        aggregate[key.split(' ')[0] + ' adult ' + key.split(' ')[2]] = 0
        aggregate[key.split(' ')[0] + ' elder ' + key.split(' ')[2]] = 0

    for key, val in data.items():
        k = key.split(' ')
        if k[1] in child_ages:
            aggregate[k[0] + ' child ' + k[2]] +=val
        elif k[1] in adult_ages:
            aggregate[k[0] + ' adult ' + k[2]] +=val
        elif k[1] in elder_ages:
            aggregate[k[0] + ' elder ' + k[2]] +=val
    return aggregate

def plot_crosstable(data, title):
    import plotly.express as px
    # Creating the bar plot
    fig = px.bar(x=list(data.keys()), y=list(data.values()), labels={'x': title, 'y': 'Count'})
    fig.update_layout(title=title, xaxis_tickangle=-45)
    return fig

def generate_combinations(left, right, total):
    # generate all possible combinations
    combinations = itertools.product(left, right)
    return [combo for combo in combinations if combo[0].split(' ')[:2] == combo[1].split(' ')[:2]]


def convert_marital_cross_table(old_cross_table):
    new_age_categories = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24',
                          '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59',
                          '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
    marital_statuses = ['Single', 'Married', 'Partner', 'Separated', 'Divorced', 'Widowed']

    # Initialize new cross table with all zeros
    new_cross_table = {'M ' + age + ' ' + status: 0 for age in new_age_categories for status in marital_statuses}
    new_cross_table.update({'F ' + age + ' ' + status: 0 for age in new_age_categories for status in marital_statuses})

    for key, value in old_cross_table.items():
        parts = key.split(' ')
        gender, age_range, marital_status = parts[0], parts[1], ' '.join(parts[2:])

        new_key = f'{gender} {age_range} {marital_status}'
        if new_key in new_cross_table:
            new_cross_table[new_key] += value

    return new_cross_table


def convert_qualification_cross_table(old_cross_table):
    new_age_categories = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24',
                          '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59',
                          '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
    qualification_levels = ['no', 'level1', 'level2', 'apprent', 'level3', 'level4+', 'other']
    genders = ['M', 'F']

    # Initialize new cross table with all zeros
    new_cross_table = {}
    for gender in genders:
        for age in new_age_categories:
            for level in qualification_levels:
                new_cross_table[f'{gender} {age} {level}'] = 0

    # Map the old age ranges to the new age categories
    age_range_mapping = {
        '0_4': ['0_4'],
        '5_7': ['5_7'],
        '8_9': ['8_9'],
        '10_14': ['10_14'],
        '15': ['15'],
        '16_24': ['16_17', '18_19', '20_24'],
        '25_34': ['25_29', '30_34'],
        '35_49': ['35_39', '40_44', '45_49'],
        '50_64': ['50_54', '55_59', '60_64'],
        '65+': ['65_69', '70_74', '75_79', '80_84', '85+']
    }

    # Distribute counts uniformly
    for key, value in old_cross_table.items():
        parts = key.split(' ')
        gender, old_age_range, qual_level = parts[0], parts[1], parts[2]

        if old_age_range in age_range_mapping:
            new_ages = age_range_mapping[old_age_range]
            count_per_age = round(value / len(new_ages))  # Uniform distribution

            for new_age in new_ages:
                new_key = f'{gender} {new_age} {qual_level}'
                new_cross_table[new_key] += count_per_age

    return new_cross_table

def convert_composition_sex_age_cross_table(old_cross_table):
    new_age_categories = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24',
                          '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59',
                          '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
    compositions = ['SP-Elder', 'SP-Adult', 'OF-Elder', 'OF-Married', 'OF-Married-0C', 'OF-Married-1C', 'OF-Married-2C', 'OF-Married-ND', 'OF-Cohabiting', 'OF-Cohabiting-0C', 'OF-Cohabiting-1C', 'OF-Cohabiting-2C', 'OF-Cohabiting-ND', 'OF-LoneParent', 'OF-Lone-1C', 'OF-Lone-2C', 'OF-Lone-ND', 'OH-1C', 'OH-2C', 'OH-Student', 'OH-Elder', 'OH-Adult']
    genders = ['M', 'F']

    # Initialize new cross table with all zeros
    new_cross_table = {}
    for gender in genders:
        for age in new_age_categories:
            for composition in compositions:
                new_cross_table[f'{gender} {age} {composition}'] = 0

    # Map the old age ranges to the new age categories
    age_range_mapping = {
        '0_15': ['0_4', '5_7', '8_9', '10_14', '15'],
        '16_24': ['16_17', '18_19', '20_24'],
        '25_34': ['25_29', '30_34'],
        '35_49': ['35_39', '40_44', '45_49'],
        '50+': ['50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
    }

    # Distribute counts uniformly
    for key, value in old_cross_table.items():
        parts = key.split(' ')
        gender, old_age_range, comp = parts[0], parts[1], parts[2]

        if old_age_range in age_range_mapping:
            new_ages = age_range_mapping[old_age_range]
            count_per_age = round(value / len(new_ages))  # Uniform distribution

            for new_age in new_ages:
                new_key = f'{gender} {new_age} {comp}'
                new_cross_table[new_key] += count_per_age

    return new_cross_table


def convert_household_cross_table(old_cross_table):
    new_age_categories = ['0_4', '5_7', '8_9', '10_14', '15', '16_17', '18_19', '20_24',
                          '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59',
                          '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
    household_compositions = ['1E_0C', '1A_0C', '2E_0C', '3A_1C', '3A_0C', '2A_0C', '2A_1C', '2A_3C', '1A_1C']
    genders = ['M', 'F']

    # Initialize new cross table with all zeros
    new_cross_table = {}
    for gender in genders:
        for age in new_age_categories:
            for comp in household_compositions:
                new_cross_table[f'{gender} {age} {comp}'] = 0

    # Map the old age ranges to the new age categories
    age_range_mapping = {
        '0_15': ['0_4', '5_7', '8_9', '10_14', '15'],
        '16_24': ['16_17', '18_19', '20_24'],
        '25_34': ['25_29', '30_34'],
        '35_49': ['35_39', '40_44', '45_49'],
        '50+': ['50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85+']
    }

    # Distribute counts uniformly
    for key, value in old_cross_table.items():
        parts = key.split(' ')
        gender, old_age_range, household_comp = parts[0], parts[1], parts[2]

        if old_age_range in age_range_mapping:
            new_ages = age_range_mapping[old_age_range]
            count_per_age = round(value / len(new_ages))  # Uniform distribution

            for new_age in new_ages:
                new_key = f'{gender} {new_age} {household_comp}'
                new_cross_table[new_key] += count_per_age

    return new_cross_table

def assign_size(composition, hh_size_weights):
    household_sizes = {
        '1PE': '1', '1PA': '1', '1FE': '1',
        '1FM-0C': '2', '1FM-1C': '3', '1FM-nC': '4+', '1FM-nA': '3+',
        '1FS-0C': '2', '1FS-1C': '3', '1FS-nC': '4+', '1FS-nA': '3+',
        '1FC-0C': '2', '1FC-1C': '3', '1FC-nC': '4+', '1FC-nA': '3+',
        '1FL-1C': '2', '1FL-nC': '2+', '1FL-nA': '2+',
        '1H-1C': '3+', '1H-nC': '3+', '1H-nA': '3+', '1H-nE': '3+'
    }
    expected_size = household_sizes[composition]
    # if expected size has +
    if '+' in expected_size:
        expected_size = int(expected_size.replace('+', ''))
        # get random choice from the expected size till 8
        expected_sizes = list(range(expected_size, 9))
        return random.choices(expected_sizes, weights=[hh_size_weights[str(size)] for size in expected_sizes])[0]

    else:
        return int(expected_size)
def get_hh_comp_by_size_crosstable(area):
    hh_comp_dict = ID.getdictionary(ID.HHcomdf, area)  # household composition
    hh_size_dict = ID.getdictionary(ID.HHsizedf, area)  # household size
    hh_total = ID.get_total(ID.HHcomdf, area)

    # create a blank hh_df and assign household composition to each household
    hh_df = pd.DataFrame(columns=['household_ID', 'composition'])
    hh_df['household_ID'] = range(1, hh_total + 1)
    id = 1
    for key, value in hh_comp_dict.items():
        for i in range(value):
            hh_df.loc[hh_df['household_ID'] == id, 'composition'] = key
            id += 1
    # assign 1PA to nan values
    hh_df['composition'] = hh_df['composition'].fillna('1PA')
    # calculate weights for each household size
    hh_size_weights = {key: value / sum(hh_size_dict.values()) for key, value in hh_size_dict.items()}
    hh_df['size'] = hh_df['composition'].apply(lambda x: int(assign_size(x, hh_size_weights)))
    # create a cross table of household composition and household size using hh_df
    hh_comp_by_size_dict = hh_df.groupby(['composition', 'size']).size().to_dict()
    #replace tuple keys with string keys
    hh_comp_by_size_dict = {f'{key[0]} {key[1]}': value for key, value in hh_comp_by_size_dict.items()}
    return hh_comp_by_size_dict


path = os.path.join(os.path.dirname(os.getcwd()), 'Diff-SynPoP')
sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'sex_by_age_5yrs.csv'))
cols = [(col.split(' ')[1] + ' ' + col.split(' ')[0]) for col in sex_by_age.columns[2:]]
sex_by_age.columns = ['geography code', 'total'] + cols

substring_mapping_rel = {
    'OR': 'O',
    'NR': 'N'
}

religion_by_sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'religion_by_sex_by_age.csv'))
for col in religion_by_sex_by_age.columns:
    for old_substring, new_substring in substring_mapping_rel.items():
        if old_substring in col:
            new_col = col.replace(old_substring, new_substring)
            religion_by_sex_by_age.rename(columns={col: new_col}, inplace=True)
            break
religion_by_sex_by_age = religion_by_sex_by_age.drop(columns=[col for col in religion_by_sex_by_age.columns if 'All' in col])

ethnicities = ['W0', 'M0', 'B0', 'A0', 'O0']
ethnic_by_sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'ethnic_by_sex_by_age.csv'))

ethnic_by_religion = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'ethnic_by_religion.csv'))
ethnic_by_religion = ethnic_by_religion.drop(columns=[col for col in ethnic_by_religion.columns[2:] if  col.split(' ')[0] not in ethnicities])
ethnic_by_religion.columns = [col.replace('0', '') for col in ethnic_by_religion.columns]

marital_by_sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'marital_by_sex_by_age.csv'))
# print(convert_marital_cross_table(getdictionary(marital_by_sex_by_age, 'E02005924')))

qualification_by_sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'qualification_by_sex_by_age.csv'))
# print(convert_qualification_cross_table(getdictionary(qualification_by_sex_by_age, 'E02005924')))

HH_composition_by_sex_by_age = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'HH_composition_by_sex_by_age.csv'))
HH_composition_by_sex_by_age = HH_composition_by_sex_by_age.rename(columns = {'Household Composition: All categories: Household composition; Religion: All categories: Religion of HRP':'total'})
HH_composition_by_sex_by_age = HH_composition_by_sex_by_age.drop(columns=[col for col in HH_composition_by_sex_by_age.columns if 'All persons' in col])
HH_composition_by_sex_by_age = HH_composition_by_sex_by_age.drop(columns=[col for col in HH_composition_by_sex_by_age.columns if 'All categories:' in col])
HH_composition_by_sex_by_age = HH_composition_by_sex_by_age.drop(columns=[col for col in HH_composition_by_sex_by_age.columns if 'Total' in col])

updated_columns = ['geography code', 'total']
for column in HH_composition_by_sex_by_age.columns[2:]:
    composition = column.split(";")[2].strip()
    composition = composition.replace("Household Composition: One person household: Aged 65 and over", "1PE")
    composition = composition.replace("Household Composition: One person household: Other", "1PA")
    composition = composition.replace("Household Composition: One family only: All aged 65 and over", "1FE")
    composition = composition.replace("Household Composition: One family only: Married or same-sex civil partnership couple: No children", "1FM-0C")
    composition = composition.replace("Household Composition: One family only: Married or same-sex civil partnership couple: Dependent children","1FM-nC")
    composition = composition.replace("Household Composition: One family only: Married or same-sex civil partnership couple: All children non-dependent", "1FM-nA")
    composition = composition.replace("Household Composition: One family only: Cohabiting couple: No children","1FC-0C")
    composition = composition.replace("Household Composition: One family only: Cohabiting couple: Dependent children","1FC-nC")
    composition = composition.replace("Household Composition: One family only: Cohabiting couple: All children non-dependent","1FC-nA")
    composition = composition.replace("Household Composition: One family only: Lone parent: One dependent child","1FL-1C")
    composition = composition.replace("Household Composition: One family only: Lone parent: Dependent children","1FL-nC")
    composition = composition.replace("Household Composition: One family only: Lone parent: All children non-dependent", "1FL-nA")
    composition = composition.replace("Household Composition: Other household types: With dependent children", "1H-nC")
    composition = composition.replace("Household Composition: Other household types: All full-time students","1H-nA")
    composition = composition.replace("Household Composition: Other household types: All aged 65 and over","1H-nE")
    composition = composition.replace("Household Composition: Other household types: Other", "1H-nA")

    gender = column.split(";")[0].strip()
    gender = gender.replace("Sex: Males", "M")
    gender = gender.replace("Sex: Females", "F")

    age = column.split(";")[1].strip()
    age = age.replace("Age: Age 0 to 15", "0-15")
    age = age.replace("Age: Age 16 to 24", "16-24")
    age = age.replace("Age: Age 25 to 34", "25-34")
    age = age.replace("Age: Age 35 to 49", "35-49")
    age = age.replace("Age: Age 50 and over", "50+")
    updated_columns.append(gender + ' ' +  age +  ' ' + composition)

HH_composition_by_sex_by_age.columns = updated_columns

# processing household composition by ethnicity cross table
HH_composition_by_Ethnicity = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'HH_composition_by_ethnicity.csv'))
HH_composition_by_Ethnicity = HH_composition_by_Ethnicity.rename(columns = {'Household Composition: All categories: Household composition; Ethnic Group: All categories: Ethnic group of HRP; measures: Value':'total'})
HH_composition_by_Ethnicity = HH_composition_by_Ethnicity.drop(columns=[col for col in HH_composition_by_Ethnicity.columns if 'All persons' in col])
HH_composition_by_Ethnicity = HH_composition_by_Ethnicity.drop(columns=[col for col in HH_composition_by_Ethnicity.columns if 'All categories:' in col])

updated_columns = ['geography code', 'total']
subtract_columns = []
for column in HH_composition_by_Ethnicity.columns[2:]:
    composition = column.split(";")[0].strip()
    # check if composition has Total or All categories
    composition = composition.replace("Household Composition: One person household: Aged 65 and over", "1PE")
    composition = composition.replace("Household Composition: One person household: Other", "1PA")
    composition = composition.replace("Household Composition: One family only: All aged 65 and over", "1FE")
    composition = composition.replace("Household Composition: One family only: Married or same-sex civil partnership couple: No children", "1FM-0C")
    composition = composition.replace("Household Composition: One family only: Married or same-sex civil partnership couple: Dependent children","1FM-nC")
    composition = composition.replace("Household Composition: One family only: Married or same-sex civil partnership couple: All children non-dependent", "1FM-nA")
    composition = composition.replace("Household Composition: One family only: Cohabiting couple: No children","1FC-0C")
    composition = composition.replace("Household Composition: One family only: Cohabiting couple: Dependent children","1FC-nC")
    composition = composition.replace("Household Composition: One family only: Cohabiting couple: All children non-dependent","1FC-nA")
    composition = composition.replace("Household Composition: One family only: Lone parent: One dependent child","1FL-1C")
    composition = composition.replace("Household Composition: One family only: Lone parent: Dependent children","1FL-nC")
    composition = composition.replace("Household Composition: One family only: Lone parent: All children non-dependent", "1FL-nA")
    composition = composition.replace("Household Composition: Other household types: With dependent children", "1H-nC")
    composition = composition.replace("Household Composition: Other household types: All full-time students","1H-nA")
    composition = composition.replace("Household Composition: Other household types: All aged 65 and over","1H-nE")
    composition = composition.replace("Household Composition: Other household types: Other", "1H-nA")

    ethnic = column.split(";")[1].strip()
    ethnic = ethnic.replace('Ethnic Group: White: Total', 'W0')
    ethnic = ethnic.replace('Ethnic Group: White: English/Welsh/Scottish/Northern Irish/British', 'W1')
    ethnic = ethnic.replace('Ethnic Group: White: Irish', 'W2')
    ethnic = ethnic.replace('Ethnic Group: White: Gypsy or Irish Traveller', 'W3')
    ethnic = ethnic.replace('Ethnic Group: White: Other White', 'W4')

    ethnic = ethnic.replace('Ethnic Group: Mixed/multiple ethnic group: Total', 'M0')
    ethnic = ethnic.replace('Ethnic Group: Mixed/multiple ethnic group: White and Black Caribbean', 'M1')
    ethnic = ethnic.replace('Ethnic Group: Mixed/multiple ethnic group: White and Black African', 'M2')
    ethnic = ethnic.replace('Ethnic Group: Mixed/multiple ethnic group: White and Asian', 'M3')
    ethnic = ethnic.replace('Ethnic Group: Mixed/multiple ethnic group: Other Mixed', 'M4')

    ethnic = ethnic.replace('Ethnic Group: Asian/Asian British: Total', 'A0')
    ethnic = ethnic.replace('Ethnic Group: Asian/Asian British: Indian', 'A1')
    ethnic = ethnic.replace('Ethnic Group: Asian/Asian British: Pakistani', 'A2')
    ethnic = ethnic.replace('Ethnic Group: Asian/Asian British: Bangladeshi', 'A3')
    ethnic = ethnic.replace('Ethnic Group: Asian/Asian British: Chinese', 'A4')
    ethnic = ethnic.replace('Ethnic Group: Asian/Asian British: Other Asian', 'A5')

    ethnic = ethnic.replace('Ethnic Group: Black/African/Caribbean/Black British: Total', 'B0')
    ethnic = ethnic.replace('Ethnic Group: Black/African/Caribbean/Black British: African', 'B1')
    ethnic = ethnic.replace('Ethnic Group: Black/African/Caribbean/Black British: Caribbean', 'B2')
    ethnic = ethnic.replace('Ethnic Group: Black/African/Caribbean/Black British: Other Black', 'B3')

    ethnic = ethnic.replace('Ethnic Group: Other ethnic group: Total', 'O0')
    ethnic = ethnic.replace('Ethnic Group: Other ethnic group: Arab', 'O1')
    ethnic = ethnic.replace('Ethnic Group: Other ethnic group: Any other ethnic group', 'O2')
    # print(composition + ' ' + ethnic)
    updated_columns.append(composition + ' ' + ethnic)

HH_composition_by_Ethnicity.columns = updated_columns
HH_composition_by_Ethnicity = HH_composition_by_Ethnicity.drop(columns=[col for col in HH_composition_by_Ethnicity.columns if ':' in col])



HH_composition_by_Religion = pd.read_csv(os.path.join(path,  'Census_2011_MSOA', 'crosstables', 'HH_composition_by_religion.csv'))
HH_composition_by_Religion = HH_composition_by_Religion.rename(columns = {'Household Composition: All categories: Household composition; Ethnic Group: All categories: Ethnic group of HRP; measures: Value':'total'})
HH_composition_by_Religion = HH_composition_by_Religion.drop(columns=[col for col in HH_composition_by_Religion.columns if 'All persons' in col])
HH_composition_by_Religion = HH_composition_by_Religion.drop(columns=[col for col in HH_composition_by_Religion.columns if 'All categories:' in col])
HH_composition_by_Religion = HH_composition_by_Religion.drop(columns=[col for col in HH_composition_by_Religion.columns if 'Total' in col])
updated_columns = ['geography code', 'total']
for column in HH_composition_by_Religion.columns[2:]:
    composition = column.split(";")[0].strip()
    composition = composition.replace("Household Composition: One person household: Aged 65 and over", "1PE")
    composition = composition.replace("Household Composition: One person household: Other", "1PA")
    composition = composition.replace("Household Composition: One family only: All aged 65 and over", "1FE")
    composition = composition.replace(
        "Household Composition: One family only: Married or same-sex civil partnership couple: No children", "1FM-0C")
    composition = composition.replace(
        "Household Composition: One family only: Married or same-sex civil partnership couple: Dependent children",
        "1FM-nC")
    composition = composition.replace(
        "Household Composition: One family only: Married or same-sex civil partnership couple: All children non-dependent",
        "1FM-nA")
    composition = composition.replace("Household Composition: One family only: Cohabiting couple: No children", "1FC-0C")
    composition = composition.replace("Household Composition: One family only: Cohabiting couple: Dependent children",
                                      "1FC-nC")
    composition = composition.replace(
        "Household Composition: One family only: Cohabiting couple: All children non-dependent", "1FC-nA")
    composition = composition.replace("Household Composition: One family only: Lone parent: One dependent child", "1FL-1C")
    composition = composition.replace("Household Composition: One family only: Lone parent: Dependent children", "1FL-nC")
    composition = composition.replace("Household Composition: One family only: Lone parent: All children non-dependent",
                                      "1FL-nA")
    composition = composition.replace("Household Composition: Other household types: With dependent children", "1H-nC")
    composition = composition.replace("Household Composition: Other household types: All full-time students", "1H-nA")
    composition = composition.replace("Household Composition: Other household types: All aged 65 and over", "1H-nE")
    composition = composition.replace("Household Composition: Other household types: Other", "1H-nA")

    religion = column.split(";")[1].strip()
    religion = religion.replace('Religion: Christian', 'C')
    religion = religion.replace('Religion: Buddhist', 'B')
    religion = religion.replace('Religion: Hindu', 'H')
    religion = religion.replace('Religion: Jewish', 'J')
    religion = religion.replace('Religion: Muslim', 'M')
    religion = religion.replace('Religion: Sikh', 'S')
    religion = religion.replace('Religion: Other religion', 'O')
    religion = religion.replace('Religion: No religion', 'N')
    religion = religion.replace('Religion: Religion not stated', 'NS')
    updated_columns.append(composition + ' ' + religion)
HH_composition_by_Religion.columns = updated_columns
