import pandas as pd
import numpy as np


def import_glass():
    df = pd.read_csv("data/glass.data", header=None)
    df.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Class']
    return df


def import_abalone():
    df = pd.read_csv("data/short_abalone.data", header=None)
    df.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight',
                  'Shell_weight', 'Class']
    df['Id'] = range(1, len(df) + 1)
    df = pd.get_dummies(data=df, columns=['Sex'])
    df = df[['Id', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight',
             'Shell_weight', 'Sex_F', 'Sex_I', 'Sex_M', 'Class']]
    return df


def import_segmentation():
    df = pd.read_csv("data/segmentation.test", header=None)
    df.columns = ['Class', 'Region_centroid_col', 'Region_centroid_row', 'Region_pixel_count', 'Short_line_density_5',
                  'Short_line_density_2', 'Vedge_mean', 'Vedge_sd', 'Hedge_mean', 'Hede_sd', 'Intensity_mean',
                  'Rawred_mean', 'Rawblue_mean', 'Rawgreen_mean', 'Exred_mean', 'Exblue_mean', 'Exgreen_mean',
                  'Value_mean', 'Saturation_mean', 'Hue_mean']
    df['Id'] = range(1, len(df) + 1)
    df = df[['Id', 'Region_centroid_col', 'Region_centroid_row', 'Region_pixel_count', 'Short_line_density_5',
             'Short_line_density_2', 'Vedge_mean', 'Vedge_sd', 'Hedge_mean', 'Hede_sd', 'Intensity_mean',
             'Rawred_mean', 'Rawblue_mean', 'Rawgreen_mean', 'Exred_mean', 'Exblue_mean', 'Exgreen_mean',
             'Value_mean', 'Saturation_mean', 'Hue_mean', 'Class']]
    return df


def import_voter():
    df = pd.read_csv('data/house-votes-84.data', header=None, na_values=['?'])
    df = df.fillna(np.random.randint(0, 1))
    df.columns = ['Class', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-budget',
                  'physician-fee-freeze',
                  'el-salvador-aid', 'religious-groups-in-school', 'anti-satellite-test-ban', 'aid-to-contras',
                  'mx-missile', 'immigration', 'synfuels-corp-cutback', 'ed-spending', 'superfunds-right-to-sue',
                  'crime', 'duty-free-exports', 'export-admin-act-sa']
    df['handicapped-infants'] = df['handicapped-infants'].apply(lambda x: 1 if x == 'y' else 0)
    df['water-project-cost-sharing'] = df['water-project-cost-sharing'].apply(lambda x: 1 if x == 'y' else 0)
    df['adoption-of-budget'] = df['adoption-of-budget'].apply(lambda x: 1 if x == 'y' else 0)
    df['physician-fee-freeze'] = df['physician-fee-freeze'].apply(lambda x: 1 if x == 'y' else 0)
    df['el-salvador-aid'] = df['el-salvador-aid'].apply(lambda x: 1 if x == 'y' else 0)
    df['religious-groups-in-school'] = df['religious-groups-in-school'].apply(lambda x: 1 if x == 'y' else 0)
    df['anti-satellite-test-ban'] = df['anti-satellite-test-ban'].apply(lambda x: 1 if x == 'y' else 0)
    df['aid-to-contras'] = df['aid-to-contras'].apply(lambda x: 1 if x == 'y' else 0)
    df['mx-missile'] = df['mx-missile'].apply(lambda x: 1 if x == 'y' else 0)
    df['immigration'] = df['immigration'].apply(lambda x: 1 if x == 'y' else 0)
    df['synfuels-corp-cutback'] = df['synfuels-corp-cutback'].apply(lambda x: 1 if x == 'y' else 0)
    df['ed-spending'] = df['ed-spending'].apply(lambda x: 1 if x == 'y' else 0)
    df['superfunds-right-to-sue'] = df['superfunds-right-to-sue'].apply(lambda x: 1 if x == 'y' else 0)
    df['crime'] = df['crime'].apply(lambda x: 1 if x == 'y' else 0)
    df['duty-free-exports'] = df['duty-free-exports'].apply(lambda x: 1 if x == 'y' else 0)
    df['export-admin-act-sa'] = df['export-admin-act-sa'].apply(lambda x: 1 if x == 'y' else 0)
    df['Id'] = range(1, len(df) + 1)
    df = df[['Id', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-budget',
                  'physician-fee-freeze',
                  'el-salvador-aid', 'religious-groups-in-school', 'anti-satellite-test-ban', 'aid-to-contras',
                  'mx-missile', 'immigration', 'synfuels-corp-cutback', 'ed-spending', 'superfunds-right-to-sue',
                  'crime', 'duty-free-exports', 'export-admin-act-sa', 'Class']]
    return df


def import_machine():
    df = pd.read_csv("data/machine.data", header=None)
    df.columns = ['Vendor', 'Id', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN',
                  'CHMAX', 'Class', 'ERP']
    df = df.drop(columns=['ERP', 'Vendor'])
    # df['Id'] = range(1, len(df) + 1)
    df = df[['Id', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN',
                  'CHMAX', 'Class']]
    return df


def import_forest_fires():
    df = pd.read_csv("data/forestfires.csv", header=None)
    df.columns = ['X', 'Y', 'Month', 'Day', 'FFMC', 'DMC', 'DC', 'ISI', 'Temp',
                  'RH', 'Wind', 'Rain', 'Class']
    df['Id'] = range(1, len(df) + 1)
    df = pd.get_dummies(data=df, columns=['Month', 'Day'])
    df = df[['Id', 'X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'Temp','RH', 'Wind', 'Rain', 'Month_apr', 'Month_aug',
             'Month_dec', 'Month_feb', 'Month_jan', 'Month_jul', 'Month_jun', 'Month_mar', 'Month_nov', 'Month_oct',
             'Month_sep', 'Day_fri', 'Day_mon', 'Day_sat', 'Day_sun', 'Day_thu', 'Day_tue', 'Day_wed', 'Class']]
    return df

import_forest_fires()