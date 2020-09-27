import pandas as pd

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

import_abalone()
