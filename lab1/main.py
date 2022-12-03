# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def countWeight(point: float, vareps: float, real_data: float):
    eps = vareps
    if point - eps > real_data:
        if point - eps == 0:
            eps -= 10 ** (-6)
        return real_data/(point - eps)
    elif point + eps < real_data:
        if point + eps == 0:
            eps -= 10 ** (-6)
        return real_data/(point + eps)
    else:
        return 1


def Jakkar(v1, v2):
    numerator = min(v1[-1], v2[-1]) - max(v1[0], v2[0])
    denominator = max(v1[-1], v2[-1]) - min(v1[0], v2[0])
    if denominator == 0:
        print("Hm. denominator is zero, it is ERROR")
        print(f"max(v1[-1], v2[-1]) = {max(v1[-1], v2[-1])}, min(v1[0], v2[0]) = {min(v1[0], v2[0])}")
        return None
    return numerator/denominator


def findIntervals(folder_path: str, file1, file2):
    eps = 10**(-4) * 0.5
    if not os.path.isdir(folder_path):
        print(f'Error with file {folder_path}')
        return None
    filepath1 = os.path.join(folder_path, file1)
    filepath2 = os.path.join(folder_path, file2)
    if not os.path.isfile(filepath1) or not os.path.isfile(filepath2):
        print(f'Error with file {file1} or file {file2}')
        return None
    df = pd.DataFrame([])

    try:
        df1 = pd.read_csv(filepath1, sep=';', encoding='ANSI')
    except Exception:
        print(f'Some errors with reading dataframe from file {file1}')
        return None
    else:
        cols = df1.columns
        df1.rename(columns={cols[0]: 'sensor'}, inplace=True)
        df['sensor'] = df1['sensor']

    try:
        df2 = pd.read_csv(filepath2, sep=';', encoding='ANSI')
    except Exception:
        print(f'Some errors with reading dataframe from file {file2}')
        return None
    else:
        cols = df2.columns
        df2.rename(columns={cols[0]: 'standard'}, inplace=True)
        df['standard'] = df2['standard']

    print(df.head(2))
    # count R
    df['div'] = df['sensor'] / df['standard']
    df = df.sort_values(by='div', ignore_index=True)
    print(df.head(2))

    # plt.plot(df.index, df['div'])
    # plt.show()
    # методом наименьших квадратов вычисляем коэффициенты линейной регрессии
    # y = a0*x + a1 * 1
    t = np.array(range(len(df)))

    r = df['div'].to_numpy()
    # print(type(x), type(y))
    A = np.vstack([t, np.ones(len(t))]).T
    a0, a1 = np.linalg.lstsq(A, r, rcond=None)[0]
    plt.plot(t, r, 'o', label='Original data')
    plt.plot(t, a0 * t + a1, 'r', label='Fitted line')
    plt.legend()
    plt.show()

    print('r = a0*t + a1')
    print(f"a0 = {a0}, a1 = {a1}")


    # now we need to crate vector weights for intervals
    df['weight'] = np.array([(countWeight(r[i], eps, a0 * i + a1)) for i in range(len(df))])
    print(df.head(2))
    return df, a0, a1
    # now we need count coefficient Jakkar


'''
    for en, _file in enumerate(files):
        # read data from csv to dataframe
        print(f'_file = {_file}')
        file_path = os.path.join(folder_path, _file)
        try:
            df = pd.read_csv(file_path, sep=';', encoding='ANSI')
        except Exception:
            print('Some errors with reading dataframe from file')
            break
        else:
            print('otlichno')
            print(df.head(2))
            cols = df.columns
            df.rename(columns={cols[0]: 'sensor' + str(en), cols[1]: 'standard'}, inplace=True)
            # count R
            df['div'] = df['sensor'] / df['standard']
            df = df.sort_values(by='div', ignore_index=True)
            print(df.head(2))

            # plt.plot(df.index, df['div'])
            # plt.show()
            # методом наименьших квадратов вычисляем коэффициенты линейной регрессии
            # y = a0*x + a1 * 1
            x = np.array(range(len(df)))

            y = df['div'].to_numpy()
            # print(type(x), type(y))
            A = np.vstack([x, np.ones(len(x))]).T
            a0, a1 = np.linalg.lstsq(A, y, rcond=None)[0]
            plt.plot(x, y, 'o', label='Original data')
            plt.plot(x, a0 * x + a1, 'r', label='Fitted line')
            plt.legend()
            plt.show()

            print(a0, a1)

            # now we need to crate vector weights for intervals
            df['weight'] = np.array([(countWeight(y[i], eps, a0*i+a1)) for i in range(len(df))])
            print(df.head(2))
'''

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('main')
    dir_path = r'E:\11сем\интервальный анализ\лаб1\Статистика измерений\400_2'
    if not os.path.isdir(dir_path):
        dir_path = dir_path.replace("\\", "/")
        if not os.path.isdir(dir_path):
            print('ai ai ai')
        else:
            print('o1 o1 o1')
    else:
        print(dir_path)

    files = os.listdir(dir_path)
    print(files)

    # file_name = 'Канал 1_400nm_0.23mm'
    # file_extension = '.xlsx'
    # file_path = os.path.join(dir_path, file_name + file_extension)
    df, a0, a1 = findIntervals(dir_path, files[0], files[1])
    print(a0, a1)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
