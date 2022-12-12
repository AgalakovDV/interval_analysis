# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl


def myDraw(t, a0: float, a1: float, ar, weight, eps, title: str, l1: str, l2: str):
    fig, ax = plt.subplots()
    plt.title(title)
    for en, (d, w) in enumerate(zip(ar, weight)):
        if en > 0:
            ax.vlines(en, d - eps*w, d + eps)
        else:
            ax.vlines(en, d - eps*w, d + eps, label=l1)

    ax.plot(t, a0 * t + a1, 'r', label=l2)
    ax.legend()
    plt.show()


def countWeight(real_data: float, vareps: float, point: float):
    eps = vareps
    if -eps <= point - real_data <= eps:
        return 1
    elif real_data - point < -eps:
        return -(real_data - point) / eps
    elif real_data - point > eps:
        return (real_data - point) / eps
    return 0


def data_without_lin(np_ar, a0: float):
    try:
        ar = []
        for en, d in enumerate(np_ar):
            ar = ar + [d - en*a0]
    except Exception:
        print('ERROR:: I can not, data_without_lin, line 45')
        return []
    return ar


def Jakkar(v1, v2):
    v1.sort()
    v2.sort()
    numerator = min(v1[-1], v2[-1]) - max(v1[0], v2[0])
    denominator = max(v1[-1], v2[-1]) - min(v1[0], v2[0])
    if denominator == 0:
        print("Hm. denominator is zero, it is ERROR")
        print(f"max(v1[-1], v2[-1]) = {max(v1[-1], v2[-1])}, min(v1[0], v2[0]) = {min(v1[0], v2[0])}")
        return None
    return numerator/denominator


def findIntervals(folder_path: str, file1, file2):
    eps = 10**(-4)
    # read data from files to df
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
    # df = df.sort_values(by='div', ignore_index=True)
    # print('sorted')
    # print(df.head(2))

    # методом наименьших квадратов вычисляем коэффициенты линейной регрессии
    # y = a0*x + a1 * 1
    t = np.array(range(len(df)))

    r = df['div'].to_numpy()
    st = df['standard'].to_numpy()
    sen = df['sensor'].to_numpy()
    # print(type(x), type(y))
    A = np.vstack([t, np.ones(len(t))]).T
    st0, st1 = np.linalg.lstsq(A, st, rcond=None)[0]
    sen0, sen1 = np.linalg.lstsq(A, sen, rcond=None)[0]

    # count weights
    # now we need to crate vector weights for intervals
    df['weight sensor'] = np.array([(countWeight(sen[i], eps, sen0 * i + sen1)) for i in range(len(df))])
    df['weight standard'] = np.array([(countWeight(st[i], eps, st0 * i + st1)) for i in range(len(df))])
    for i in range(5):
        print(f'{i}) {sen[i] - sen0 * i - sen1}, {st[i] - st0 * i - st1}')
    print(df.head(2))

    # data without linear component
    df['standard1'] = data_without_lin(df['standard'].to_numpy(), st0)
    df['sensor2'] = data_without_lin(df['sensor'].to_numpy(), sen0)

    # draw data
    myDraw(t, st0, st1, st, [1]*len(df),
           eps, "Обинтерваленые данные эталонные", 'Standard data', 'Fitted line')
    myDraw(t, sen0, sen1, sen, [1]*len(df),
           eps, "Обинтерваленые данные с датчика", 'Sensor data', 'Fitted line')
    myDraw(t, st0, st1, st, df['weight standard'],
           eps, "Обинтерваленые данные эталонные с добавкой", 'Standard data', 'Fitted line')
    myDraw(t, sen0, sen1, sen, df['weight sensor'],
           eps, "Обинтерваленые данные с датчика с добавкой", 'Sensor data', 'Fitted line')
    myDraw(t, 0, st1, df['standard1'], df['weight standard'],
           eps, "Спрямлённые эталонные данные", 'Standard data', 'Fitted line')
    myDraw(t, 0, sen1, df['sensor2'], df['weight sensor'],
           eps, "Спрямлённые данные с датчика", 'Sensor data', 'Fitted line')

    print('sensor = a0*t + a1')
    print(f"a0 = {sen0}, a1 = {sen1}")
    print('standard = a0*t + a1')
    print(f"a0 = {st0}, a1 = {st1}")

    print('do you want to save excel with statistics?')
    ch = input()
    if 'y' in ch or 'Y' in ch:
        df.to_excel(os.path.join(folder_path, 'answ.xlsx'))
    elif 'n' in ch or 'N' in ch:
        print('data not saved, answer is ', ch)
    else:
        print('answer is ', ch)

    # now we need count coefficient Jakkar
    # jak = Jakkar(df['sensor'].to_numpy(), df['standard'].to_numpy())
    # print('Jakkar = ', jak)
    return df, sen0, sen1, st0, st1


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
    dir_path = r'E:\11сем\интервальный_анализ\lab1_\статистика_измерений\400_2'
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
    df, sen0, sen1, st0, st1 = findIntervals(dir_path, files[0], files[1])
    print(sen0, sen1)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
