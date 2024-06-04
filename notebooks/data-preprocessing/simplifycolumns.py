import random
import numpy as np
import geopandas as gpd
import pandas as pd
import os

path = os.path.join(os.path.dirname(os.getcwd()))

# Read cross tables
# sex_by_age = pd.read_csv(os.path.join(path, 'NOMIS', 'Census_2011_MSOA', 'crosstables', 'sex_by_age.csv'))
# # sex_by_age = sex_by_age[sex_by_age['geography code'] == 'E02005949']
# sex_by_age = sex_by_age.drop(columns=[col for col in sex_by_age.columns if 'A' in col])
#
# sex_by_age_5yrs = pd.DataFrame()
# sex_by_age_5yrs['geography code'] = sex_by_age['geography code']
# sex_by_age_5yrs['total'] = sex_by_age['total']
#
#
# sex_by_age_5yrs['0-4 M'] = sex_by_age[['0M', '1M', '2M', '3M', '4M']].sum(axis=1)
# sex_by_age_5yrs['5-7 M'] = sex_by_age[['5M', '6M', '7M']].sum(axis=1)
# sex_by_age_5yrs['8-9 M'] = sex_by_age[['8M', '9M']].sum(axis=1)
# sex_by_age_5yrs['10-14 M'] = sex_by_age[['10M', '11M', '12M', '13M', '14M']].sum(axis=1)
# sex_by_age_5yrs['15 M'] = sex_by_age[['15M']].sum(axis=1)
# sex_by_age_5yrs['16-17 M'] = sex_by_age[['16M', '17M']].sum(axis=1)
# sex_by_age_5yrs['18-19 M'] = sex_by_age[['18M', '19M']].sum(axis=1)
# sex_by_age_5yrs['20-24 M'] = sex_by_age[['20M', '21M', '22M', '23M', '24M']].sum(axis=1)
# sex_by_age_5yrs['25-29 M'] = sex_by_age[['25M', '26M', '27M', '28M', '29M']].sum(axis=1)
# sex_by_age_5yrs['30-34 M'] = sex_by_age[['30M', '31M', '32M', '33M', '34M']].sum(axis=1)
# sex_by_age_5yrs['35-39 M'] = sex_by_age[['35M', '36M', '37M', '38M', '39M']].sum(axis=1)
# sex_by_age_5yrs['40-44 M'] = sex_by_age[['40M', '41M', '42M', '43M', '44M']].sum(axis=1)
# sex_by_age_5yrs['45-49 M'] = sex_by_age[['45M', '46M', '47M', '48M', '49M']].sum(axis=1)
# sex_by_age_5yrs['50-54 M'] = sex_by_age[['50M', '51M', '52M', '53M', '54M']].sum(axis=1)
# sex_by_age_5yrs['55-59 M'] = sex_by_age[['55M', '56M', '57M', '58M', '59M']].sum(axis=1)
# sex_by_age_5yrs['60-64 M'] = sex_by_age[['60M', '61M', '62M', '63M', '64M']].sum(axis=1)
# sex_by_age_5yrs['65-69 M'] = sex_by_age[['65M', '66M', '67M', '68M', '69M']].sum(axis=1)
# sex_by_age_5yrs['70-74 M'] = sex_by_age[['70M', '71M', '72M', '73M', '74M']].sum(axis=1)
# sex_by_age_5yrs['75-79 M'] = sex_by_age[['75M', '76M', '77M', '78M', '79M']].sum(axis=1)
# sex_by_age_5yrs['80-84 M'] = sex_by_age[['80M', '81M', '82M', '83M', '84M']].sum(axis=1)
# sex_by_age_5yrs['85+ M'] = sex_by_age[['85M']].sum(axis=1)
#
# sex_by_age_5yrs['0-4 F'] = sex_by_age[['0F', '1F', '2F', '3F', '4F']].sum(axis=1)
# sex_by_age_5yrs['5-7 F'] = sex_by_age[['5F', '6F', '7F']].sum(axis=1)
# sex_by_age_5yrs['8-9 F'] = sex_by_age[['8F', '9F']].sum(axis=1)
# sex_by_age_5yrs['10-14 F'] = sex_by_age[['10F', '11F', '12F', '13F', '14F']].sum(axis=1)
# sex_by_age_5yrs['15 F'] = sex_by_age[['15F']].sum(axis=1)
# sex_by_age_5yrs['16-17 F'] = sex_by_age[['16F', '17F']].sum(axis=1)
# sex_by_age_5yrs['18-19 F'] = sex_by_age[['18F', '19F']].sum(axis=1)
# sex_by_age_5yrs['20-24 F'] = sex_by_age[['20F', '21F', '22F', '23F', '24F']].sum(axis=1)
# sex_by_age_5yrs['25-29 F'] = sex_by_age[['25F', '26F', '27F', '28F', '29F']].sum(axis=1)
# sex_by_age_5yrs['30-34 F'] = sex_by_age[['30F', '31F', '32F', '33F', '34F']].sum(axis=1)
# sex_by_age_5yrs['35-39 F'] = sex_by_age[['35F', '36F', '37F', '38F', '39F']].sum(axis=1)
# sex_by_age_5yrs['40-44 F'] = sex_by_age[['40F', '41F', '42F', '43F', '44F']].sum(axis=1)
# sex_by_age_5yrs['45-49 F'] = sex_by_age[['45F', '46F', '47F', '48F', '49F']].sum(axis=1)
# sex_by_age_5yrs['50-54 F'] = sex_by_age[['50F', '51F', '52F', '53F', '54F']].sum(axis=1)
# sex_by_age_5yrs['55-59 F'] = sex_by_age[['55F', '56F', '57F', '58F', '59F']].sum(axis=1)
# sex_by_age_5yrs['60-64 F'] = sex_by_age[['60F', '61F', '62F', '63F', '64F']].sum(axis=1)
# sex_by_age_5yrs['65-69 F'] = sex_by_age[['65F', '66F', '67F', '68F', '69F']].sum(axis=1)
# sex_by_age_5yrs['70-74 F'] = sex_by_age[['70F', '71F', '72F', '73F', '74F']].sum(axis=1)
# sex_by_age_5yrs['75-79 F'] = sex_by_age[['75F', '76F', '77F', '78F', '79F']].sum(axis=1)
# sex_by_age_5yrs['80-84 F'] = sex_by_age[['80F', '81F', '82F', '83F', '84F']].sum(axis=1)
# sex_by_age_5yrs['85+ F'] = sex_by_age[['85F']].sum(axis=1)
# sex_by_age_5yrs.to_csv(os.path.join(path, 'NOMIS', 'Census_2011_MSOA', 'crosstables', 'sex_by_age_5yrs.csv'), index=False)

# ethnic_by_sex_by_age = pd.read_csv(os.path.join(path, 'Diff-SynPoP', 'Census_2011_MSOA', 'crosstables', 'ethnic_by_sex_by_age_aggregated.csv'))
# ethnic_by_sex_by_age = ethnic_by_sex_by_age.rename(columns = {'Sex: All persons; Age: All categories: Age; Ethnic Group: All categories: Ethnic group':'total'})
# ethnic_by_sex_by_age = ethnic_by_sex_by_age.drop(columns=[col for col in ethnic_by_sex_by_age.columns if 'All' in col])
# ethnic_by_sex_by_age = ethnic_by_sex_by_age.drop(columns=[col for col in ethnic_by_sex_by_age.columns if 'Total' in col])
# ethnic_by_sex_by_age = ethnic_by_sex_by_age.drop(columns=['date', 'geography'])
#
# columns = list(ethnic_by_sex_by_age.columns)[2:]
# updated_columns = ['geography code', 'total']
# for column in columns:
#     genders = column.split(";")[0].replace('Sex: Females', 'F').replace('Sex: Males', 'M')
#     ages = column.split(";")[1].replace(' Age: Age ', '').replace(' and over', '+'). replace(' to ', '-')
#     ethnic = column.split(";")[2].replace(' Ethnic Group: ', '')
#     ethnic = ethnic.replace('White: English/Welsh/Scottish/Northern Irish/British', 'W0')
#     ethnic = ethnic.replace('White: Irish', 'W1')
#     ethnic = ethnic.replace('White: Gypsy or Irish Traveller', 'W2')
#     ethnic = ethnic.replace('White: Other White', 'W3')
#     ethnic = ethnic.replace('Mixed/multiple ethnic group: White and Black Caribbean', 'M0')
#     ethnic = ethnic.replace('Mixed/multiple ethnic group: White and Black African', 'M1')
#     ethnic = ethnic.replace('Mixed/multiple ethnic group: White and Asian', 'M2')
#     ethnic = ethnic.replace('Mixed/multiple ethnic group: Other Mixed', 'M3')
#     ethnic = ethnic.replace('Asian/Asian British: Indian', 'A0')
#     ethnic = ethnic.replace('Asian/Asian British: Pakistani', 'A1')
#     ethnic = ethnic.replace('Asian/Asian British: Bangladeshi', 'A2')
#     ethnic = ethnic.replace('Asian/Asian British: Chinese', 'A3')
#     ethnic = ethnic.replace('Asian/Asian British: Other Asian', 'A4')
#     ethnic = ethnic.replace('Black/African/Caribbean/Black British: African', 'B0')
#     ethnic = ethnic.replace('Black/African/Caribbean/Black British: Caribbean', 'B1')
#     ethnic = ethnic.replace('Black/African/Caribbean/Black British: Other Black', 'B2')
#     ethnic = ethnic.replace('Other ethnic group: Arab', 'O0')
#     ethnic = ethnic.replace('Other ethnic group: Any other ethnic group', 'O1')
#     updated_columns.append(genders + ' ' +  ages +  ' ' + ethnic)
#
# ethnic_by_sex_by_age.columns = updated_columns
# ethnic_by_sex_by_age.to_csv(os.path.join(path, 'Diff-SynPoP', 'Census_2011_MSOA', 'crosstables', 'ethnic_by_sex_by_age_aggregated.csv'), index=False)

# religion_by_sex_by_age = pd.read_csv(os.path.join(path, 'NOMIS', 'Census_2011_MSOA', 'crosstables', 'religion_by_sex_by_age.csv'))
# religion_by_sex_by_age = religion_by_sex_by_age.rename(columns = {'Sex: All persons; Age: All categories: Age; Religion: All categories: Religion':'total'})
# religion_by_sex_by_age = religion_by_sex_by_age.drop(columns=[col for col in religion_by_sex_by_age.columns if 'All' in col])
# religion_by_sex_by_age = religion_by_sex_by_age.drop(columns=[col for col in religion_by_sex_by_age.columns if 'Total' in col])
# religion_by_sex_by_age = religion_by_sex_by_age.drop(columns=['date', 'geography'])
# columns = list(religion_by_sex_by_age.columns)[2:]
# updated_columns = ['geography code', 'total']
# for column in columns:
#     genders = column.split(";")[0].replace('Sex: Females', 'F').replace('Sex: Males', 'M')
#     ages = column.split(";")[1].replace(' Age: Age ', '').replace(' and over', '+'). replace(' to ', '-')
#     religion = column.split(";")[2]
#     religion = religion.replace(' Religion: Christian', 'C')
#     religion = religion.replace(' Religion: Buddhist', 'B')
#     religion = religion.replace(' Religion: Hindu', 'H')
#     religion = religion.replace(' Religion: Jewish', 'J')
#     religion = religion.replace(' Religion: Muslim', 'M')
#     religion = religion.replace(' Religion: Sikh', 'S')
#     religion = religion.replace(' Religion: Other religion', 'OR')
#     religion = religion.replace(' Religion: No religion', 'NR')
#     religion = religion.replace(' Religion: Religion not stated', 'NS')
#     updated_columns.append(genders + ' ' +  ages +  ' ' + religion)
#
# religion_by_sex_by_age.columns = updated_columns
# religion_by_sex_by_age.to_csv(os.path.join(path, 'NOMIS', 'Census_2011_MSOA', 'crosstables', 'religion_by_sex_by_age2.csv'), index=False)


# marital_by_sex_by_age = pd.read_csv(os.path.join(path, 'SPONGE', 'Census_2011_MSOA', 'crosstables', 'marital_by_sex_by_age.csv'))
# marital_by_sex_by_age = marital_by_sex_by_age.rename(columns = {'Age: All categories: Age 16 and over; Sex: All persons; Marital Status: All categories: Marital and civil partnership status':'total'})
# marital_by_sex_by_age = marital_by_sex_by_age.drop(columns=[col for col in marital_by_sex_by_age.columns if 'All' in col])
# marital_by_sex_by_age = marital_by_sex_by_age.drop(columns=['date', 'geography'])
#
# columns = list(marital_by_sex_by_age.columns)[2:]
# updated_columns = ['geography code', 'total']
# for column in columns:
#     ages = column.split(";")[0].replace('Age: Age ', '').replace(' to ', '-').replace(' and over', '+')
#     genders = column.split(";")[1].replace(' Sex: Females', 'F').replace(' Sex: Males', 'M')
#     status = column.split(";")[2].replace(' Marital Status: ', '')
#     status = status.replace('Single (never married or never registered a same-sex civil partnership)','Single')
#     status = status.replace('In a registered same-sex civil partnership', 'Partner')
#     status = status.replace('Separated (but still legally married or still legally in a same-sex civil partnership)', 'Seperated')
#     status = status.replace('Divorced or formerly in a same-sex civil partnership which is now legally dissolved', 'Divorced')
#     status = status.replace('Widowed or surviving partner from a same-sex civil partnership', 'Widowed')
#     updated_columns.append(genders + ' ' +  ages +  ' ' + status)
#
# marital_by_sex_by_age.columns = updated_columns
# marital_by_sex_by_age.to_csv(os.path.join(path, 'SPONGE', 'Census_2011_MSOA', 'crosstables', 'marital_by_sex_by_age2.csv'), index=False)

# qualification_by_sex_by_age = pd.read_csv(os.path.join(path, 'NOMIS', 'Census_2011_MSOA', 'crosstables', 'qualification_by_sex_by_age.csv'))
# qualification_by_sex_by_age = qualification_by_sex_by_age.rename(columns = {'Sex: All persons; Age: All categories: Age 16 and over; Highest Level of Qualification: All categories: Highest level of qualification':'total'})
# qualification_by_sex_by_age = qualification_by_sex_by_age.drop(columns=[col for col in qualification_by_sex_by_age.columns if 'All' in col])
# qualification_by_sex_by_age = qualification_by_sex_by_age.drop(columns=['date', 'geography'])
# columns = list(qualification_by_sex_by_age.columns)[2:]
# updated_columns = ['geography code', 'total']
# for column in columns:
#     genders = column.split(";")[0].replace('Sex: Females', 'F').replace('Sex: Males', 'M')
#     ages = column.split(";")[1].replace(' Age: Age ', '').replace(' and over', '+'). replace(' to ', '-').replace(' and under', '*')
#     qualification = column.split(";")[2]
#     qualification = qualification.replace(' Highest Level of Qualification: Level 1 qualifications', 'level1')
#     qualification = qualification.replace(' Highest Level of Qualification: Level 2 qualifications', 'level2')
#     qualification = qualification.replace(' Highest Level of Qualification: Apprenticeship', 'apprent')
#     qualification = qualification.replace(' Highest Level of Qualification: Level 3 qualifications', 'level3')
#     qualification = qualification.replace(' Highest Level of Qualification: Level 4 qualifications and above', 'level4+')
#     qualification = qualification.replace(' Highest Level of Qualification: Other qualifications', 'other')
#     qualification = qualification.replace(' Highest Level of Qualification: No qualifications', 'no')
#     updated_columns.append(genders + ' ' +  ages +  ' ' + qualification)
#
# qualification_by_sex_by_age.columns = updated_columns
# qualification_by_sex_by_age.to_csv(os.path.join(path, 'NOMIS', 'Census_2011_MSOA', 'crosstables', 'qualification_by_sex_by_age2.csv'), index=False)

# ethnic_by_religion = pd.read_csv(os.path.join(path, 'Diff-SynPoP', 'Census_2011_MSOA', 'crosstables', 'ethnic_by_religion.csv'))
# ethnic_by_religion = ethnic_by_religion.rename(columns = {'Ethnic Group: All categories: Ethnic group; Religion: All categories: Religion':'total'})
# ethnic_by_religion = ethnic_by_religion.drop(columns=[col for col in ethnic_by_religion.columns if 'All categories' in col])
# ethnic_by_religion = ethnic_by_religion.drop(columns=[col for col in ethnic_by_religion.columns if 'Total' in col])
# ethnic_by_religion = ethnic_by_religion.drop(columns=['date', 'geography'])
#
# columns = list(ethnic_by_religion.columns)[2:]
# updated_columns = ['geography code', 'total']
# for column in columns:
#     ethnic = column.split(";")[0].replace('Ethnic Group: ', '')
#     ethnic = ethnic.replace('White: English/Welsh/Scottish/Northern Irish/British', 'W0')
#     ethnic = ethnic.replace('White: Irish', 'W1')
#     ethnic = ethnic.replace('White: Gypsy or Irish Traveller', 'W2')
#     ethnic = ethnic.replace('White: Other White', 'W3')
#     ethnic = ethnic.replace('Mixed/multiple ethnic group: White and Black Caribbean', 'M0')
#     ethnic = ethnic.replace('Mixed/multiple ethnic group: White and Black African', 'M1')
#     ethnic = ethnic.replace('Mixed/multiple ethnic group: White and Asian', 'M2')
#     ethnic = ethnic.replace('Mixed/multiple ethnic group: Other Mixed', 'M3')
#     ethnic = ethnic.replace('Asian/Asian British: Indian', 'A0')
#     ethnic = ethnic.replace('Asian/Asian British: Pakistani', 'A1')
#     ethnic = ethnic.replace('Asian/Asian British: Bangladeshi', 'A2')
#     ethnic = ethnic.replace('Asian/Asian British: Chinese', 'A3')
#     ethnic = ethnic.replace('Asian/Asian British: Other Asian', 'A4')
#     ethnic = ethnic.replace('Black/African/Caribbean/Black British: African', 'B0')
#     ethnic = ethnic.replace('Black/African/Caribbean/Black British: Caribbean', 'B1')
#     ethnic = ethnic.replace('Black/African/Caribbean/Black British: Other Black', 'B2')
#     ethnic = ethnic.replace('Other ethnic group: Arab', 'O0')
#     ethnic = ethnic.replace('Other ethnic group: Any other ethnic group', 'O1')
#
#     religion = column.split(";")[1]
#     religion = religion.replace(' Religion: Christian', 'C')
#     religion = religion.replace(' Religion: Buddhist', 'B')
#     religion = religion.replace(' Religion: Hindu', 'H')
#     religion = religion.replace(' Religion: Jewish', 'J')
#     religion = religion.replace(' Religion: Muslim', 'M')
#     religion = religion.replace(' Religion: Sikh', 'S')
#     religion = religion.replace(' Religion: Other religion', 'OR')
#     religion = religion.replace(' Religion: No religion', 'NR')
#     religion = religion.replace(' Religion: Religion not stated', 'NS')
#     updated_columns.append(ethnic + ' ' + religion)
#
# ethnic_by_religion.columns = updated_columns
# ethnic_by_religion.to_csv(os.path.join(path, 'Diff-SynPoP', 'Census_2011_MSOA', 'crosstables', 'ethnic_by_religion.csv'), index=False)


# HH_composition_by_ethnic = pd.read_csv(os.path.join(path, 'Diff-SynPoP', 'Census_2011_MSOA', 'crosstables', 'HH_composition_by_ethnic.csv'))
# HH_composition_by_ethnic = HH_composition_by_ethnic.rename(columns = {'Sex: All persons; Age: All categories: Age; Household Composition: All categories: Household composition':'total'})
# HH_composition_by_ethnic = HH_composition_by_ethnic.drop(columns=[col for col in HH_composition_by_ethnic.columns if 'All persons' in col])
# HH_composition_by_ethnic = HH_composition_by_ethnic.drop(columns=[col for col in HH_composition_by_ethnic.columns if 'All categories' in col])
# HH_composition_by_ethnic = HH_composition_by_ethnic.drop(columns=[col for col in HH_composition_by_ethnic.columns if 'Total' in col])
# updated_columns = ['geography code', 'total']
# for column in HH_composition_by_ethnic.columns[2:]:
#     genders = column.split(";")[0].replace('Sex: Females', 'F').replace('Sex: Males', 'M')
#     ages = column.split(";")[1].replace(' Age: Age ', '').replace(' and over', '+'). replace(' to ', '-').replace(' and under', '*')
#     composition = column.split(";")[2]
#     composition = composition.replace(' Household Composition: One person household: Aged 65 and over', 'SP-Elder')
#     composition = composition.replace(' Household Composition: One person household: Other', 'SP-Adult')
#     composition = composition.replace(' Household Composition: One family only: All aged 65 and over', 'OF-Elder')
#     composition = composition.replace(' Household Composition: One family only: Married or same-sex civil partnership couple: No children','OF-Married-0C')
#     composition = composition.replace(' Household Composition: One family only: Married or same-sex civil partnership couple: Dependent children','OF-Married-2C')
#     composition = composition.replace(' Household Composition: One family only: Married or same-sex civil partnership couple: All children non-dependent','OF-Married-ND')
#     composition = composition.replace(' Household Composition: One family only: Cohabiting couple: No children','OF-Cohabiting-0C')
#     composition = composition.replace(' Household Composition: One family only: Cohabiting couple: Dependent children','OF-Cohabiting-2C')
#     composition = composition.replace(' Household Composition: One family only: Cohabiting couple: All children non-dependent', 'OF-Cohabiting-ND')
#     composition = composition.replace(' Household Composition: One family only: Lone parent: Dependent children','OF-Lone-2C')
#     composition = composition.replace(' Household Composition: One family only: Lone parent: All children non-dependent', 'OF-Lone-ND')
#     composition = composition.replace(' Household Composition: Other household types: With dependent children', 'OH-2C')
#     composition = composition.replace(' Household Composition: Other household types: All full-time students','OH-Student')
#     composition = composition.replace(' Household Composition: Other household types: All aged 65 and over', 'OH-Elder')
#     composition = composition.replace(' Household Composition: Other household types: Other', 'OH-Adult')
#     updated_columns.append(genders + ' ' +  ages +  ' ' + composition)
# 
# HH_composition_by_ethnic.columns = updated_columns
# HH_composition_by_ethnic.to_csv(os.path.join(path, 'Diff-SynPoP', 'Census_2011_MSOA', 'crosstables', 'HH_composition_by_ethnic.csv'), index=False)


# HH_composition_by_ethnic = pd.read_csv(os.path.join(path, 'Diff-SynPoP', 'Census_2011_MSOA', 'crosstables', 'HH_composition_by_ethnicity.csv'))
# HH_composition_by_ethnic = HH_composition_by_ethnic.rename(columns = {'Household Composition: All categories: Household composition; Ethnic Group: All categories: Ethnic group of HRP':'total'})
# # HH_composition_by_ethnic = HH_composition_by_ethnic.drop(columns=[col for col in HH_composition_by_ethnic.columns if 'All persons' in col])
# HH_composition_by_ethnic = HH_composition_by_ethnic.drop(columns=[col for col in HH_composition_by_ethnic.columns if 'All categories' in col])
# # HH_composition_by_ethnic = HH_composition_by_ethnic.drop(columns=[col for col in HH_composition_by_ethnic.columns if 'Total' in col])
#
# updated_columns = ['geography code', 'total']
# for column in HH_composition_by_ethnic.columns[2:]:
#     composition = column.split(";")[0]
#
#     composition = composition.replace('Household Composition: One person household: Total', 'SP')
#     composition = composition.replace('Household Composition: One person household: Aged 65 and over', 'SP-Elder')
#     composition = composition.replace('Household Composition: One person household: Other', 'SP-Adult')
#
#     composition = composition.replace('Household Composition: One family only: Total', 'OF')
#     composition = composition.replace('Household Composition: One family only: All aged 65 and over', 'OF-Elder')
#
#     composition = composition.replace('Household Composition: One family only: Married or same-sex civil partnership couple: Total','OF-Married')
#     composition = composition.replace('Household Composition: One family only: Married or same-sex civil partnership couple: No children','OF-Married-0C')
#     composition = composition.replace('Household Composition: One family only: Married or same-sex civil partnership couple: Dependent children','OF-Married-2C')
#     composition = composition.replace('Household Composition: One family only: Married or same-sex civil partnership couple: All children non-dependent','OF-Married-ND')
#
#     composition = composition.replace('Household Composition: One family only: Cohabiting couple: Total', 'OF-Cohabiting')
#     composition = composition.replace('Household Composition: One family only: Cohabiting couple: No children','OF-Cohabiting-0C')
#     composition = composition.replace('Household Composition: One family only: Cohabiting couple: Dependent children','OF-Cohabiting-2C')
#     composition = composition.replace('Household Composition: One family only: Cohabiting couple: All children non-dependent', 'OF-Cohabiting-ND')
#
#     composition = composition.replace('Household Composition: One family only: Lone parent: Total','OF-Lone')
#     composition = composition.replace('Household Composition: One family only: Lone parent: Dependent children','OF-Lone-2C')
#     composition = composition.replace('Household Composition: One family only: Lone parent: All children non-dependent', 'OF-Lone-ND')
#
#     composition = composition.replace('Household Composition: Other household types: Total', 'OH')
#     composition = composition.replace('Household Composition: Other household types: With dependent children', 'OH-2C')
#     composition = composition.replace('Household Composition: Other household types: All full-time students','OH-Student')
#     composition = composition.replace('Household Composition: Other household types: All aged 65 and over', 'OH-Elder')
#     composition = composition.replace('Household Composition: Other household types: Other', 'OH-Adult')
#
#     ethnic = column.split(";")[1]
#     ethnic = ethnic.replace(' Ethnic Group: White: Total', 'W0')
#     ethnic = ethnic.replace(' Ethnic Group: White: English/Welsh/Scottish/Northern Irish/British', 'W1')
#     ethnic = ethnic.replace(' Ethnic Group: White: Irish', 'W2')
#     ethnic = ethnic.replace(' Ethnic Group: White: Gypsy or Irish Traveller', 'W3')
#     ethnic = ethnic.replace(' Ethnic Group: White: Other White', 'W4')
#
#     ethnic = ethnic.replace(' Ethnic Group: Mixed/multiple ethnic group: Total', 'M0')
#     ethnic = ethnic.replace(' Ethnic Group: Mixed/multiple ethnic group: White and Black Caribbean', 'M1')
#     ethnic = ethnic.replace(' Ethnic Group: Mixed/multiple ethnic group: White and Black African', 'M2')
#     ethnic = ethnic.replace(' Ethnic Group: Mixed/multiple ethnic group: White and Asian', 'M3')
#     ethnic = ethnic.replace(' Ethnic Group: Mixed/multiple ethnic group: Other Mixed', 'M4')
#
#     ethnic = ethnic.replace(' Ethnic Group: Asian/Asian British: Total', 'A0')
#     ethnic = ethnic.replace(' Ethnic Group: Asian/Asian British: Indian', 'A1')
#     ethnic = ethnic.replace(' Ethnic Group: Asian/Asian British: Pakistani', 'A2')
#     ethnic = ethnic.replace(' Ethnic Group: Asian/Asian British: Bangladeshi', 'A3')
#     ethnic = ethnic.replace(' Ethnic Group: Asian/Asian British: Chinese', 'A4')
#     ethnic = ethnic.replace(' Ethnic Group: Asian/Asian British: Other Asian', 'A5')
#
#     ethnic = ethnic.replace(' Ethnic Group: Black/African/Caribbean/Black British: Total', 'B0')
#     ethnic = ethnic.replace(' Ethnic Group: Black/African/Caribbean/Black British: African', 'B1')
#     ethnic = ethnic.replace(' Ethnic Group: Black/African/Caribbean/Black British: Caribbean', 'B2')
#     ethnic = ethnic.replace(' Ethnic Group: Black/African/Caribbean/Black British: Other Black', 'B3')
#
#     ethnic = ethnic.replace(' Ethnic Group: Other ethnic group: Total', 'O0')
#     ethnic = ethnic.replace(' Ethnic Group: Other ethnic group: Arab', 'O1')
#     ethnic = ethnic.replace(' Ethnic Group: Other ethnic group: Any other ethnic group', 'O2')
#     updated_columns.append(composition +' '+ ethnic)
#
# HH_composition_by_ethnic.columns = updated_columns
# HH_composition_by_ethnic.to_csv(os.path.join(path, 'Diff-SynPoP', 'Census_2011_MSOA', 'crosstables', 'HH_composition_by_ethnicity.csv'), index=False)
#


# HH_composition_by_sex_by_age = pd.read_csv(os.path.join(path, 'Diff-SynPoP', 'Census_2011_MSOA', 'crosstables', 'HH_composition_by_sex_by_age.csv'))
# HH_composition_by_sex_by_age = HH_composition_by_sex_by_age.rename(columns = {'Household Composition: All categories: Household composition; Religion: All categories: Religion of HRP':'total'})
# HH_composition_by_sex_by_age = HH_composition_by_sex_by_age.drop(columns=[col for col in HH_composition_by_sex_by_age.columns if 'All categories' in col])
# HH_composition_by_sex_by_age = HH_composition_by_sex_by_age.drop(columns=[col for col in HH_composition_by_sex_by_age.columns if 'Total' in col])
#
# updated_columns = ['geography code', 'total']
# for column in HH_composition_by_sex_by_age.columns[2:]:
#     composition = column.split(";")[0]
#     composition = composition.replace('Household Composition: One person household: Aged 65 and over', 'SP-Elder')
#     composition = composition.replace('Household Composition: One person household: Other', 'SP-Adult')
#     composition = composition.replace('Household Composition: One family only: All aged 65 and over', 'OF-Elder')
#     composition = composition.replace('Household Composition: One family only: Married or same-sex civil partnership couple: No children','OF-Married-0C')
#     composition = composition.replace('Household Composition: One family only: Married or same-sex civil partnership couple: Dependent children','OF-Married-2C')
#     composition = composition.replace('Household Composition: One family only: Married or same-sex civil partnership couple: All children non-dependent','OF-Married-ND')
#     composition = composition.replace('Household Composition: One family only: Cohabiting couple: No children','OF-Cohabiting-0C')
#     composition = composition.replace('Household Composition: One family only: Cohabiting couple: Dependent children','OF-Cohabiting-2C')
#     composition = composition.replace('Household Composition: One family only: Cohabiting couple: All children non-dependent', 'OF-Cohabiting-ND')
#     composition = composition.replace('Household Composition: One family only: Lone parent: Dependent children','OF-Lone-2C')
#     composition = composition.replace('Household Composition: One family only: Lone parent: All children non-dependent', 'OF-Lone-ND')
#     composition = composition.replace('Household Composition: Other household types: With dependent children', 'OH-2C')
#     composition = composition.replace('Household Composition: Other household types: All full-time students','OH-Student')
#     composition = composition.replace('Household Composition: Other household types: All aged 65 and over', 'OH-Elder')
#     composition = composition.replace('Household Composition: Other household types: Other', 'OH-Adult')
#
#     religion = column.split(";")[1]
#     religion = religion.replace(' Religion: Christian', 'C')
#     religion = religion.replace(' Religion: Buddhist', 'B')
#     religion = religion.replace(' Religion: Hindu', 'H')
#     religion = religion.replace(' Religion: Jewish', 'J')
#     religion = religion.replace(' Religion: Muslim', 'M')
#     religion = religion.replace(' Religion: Sikh', 'S')
#     religion = religion.replace(' Religion: Other religion', 'OR')
#     religion = religion.replace(' Religion: No religion', 'NR')
#     religion = religion.replace(' Religion: Religion not stated', 'NS')
#     updated_columns.append(composition +' '+ religion)
#
# HH_composition_by_sex_by_age.columns = updated_columns
# HH_composition_by_sex_by_age.to_csv(os.path.join(path, 'Diff-SynPoP', 'Census_2011_MSOA', 'crosstables', 'HH_composition_by_sex_by_age.csv'), index=False)


HH_composition_by_sex_by_age = pd.read_csv(os.path.join(path, 'Diff-SynPoP', 'Census_2011_MSOA', 'crosstables', 'HH_composition_by_sex_by_age.csv'))
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
# HH_composition_by_sex_by_age.to_csv(os.path.join(path, 'Diff-SynPoP', 'Census_2011_MSOA', 'crosstables', 'HH_composition_by_sex_by_age.csv'), index=False)

