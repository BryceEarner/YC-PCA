import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import quandl

quandl.ApiConfig.api_key = "YOUR KEY HERE"


def get_quandl_data(source='USTREASURY', table='YIELD', start_date='2020-06-1',
                     end_date=datetime.today().strftime("%Y-%m-%d")):
    """
    Import data from Quandl as a dataframe.
    """
    name_to_pull = source + '/' + table
    return quandl.get(name_to_pull, start_date=start_date, end_date=end_date)


def prep_data(yields, tenors, col_name):
    """
    Calculate the one day changes for each selected tenor
    """
    df = yields.loc[:, [str(i) + col_name for i in tenors]]
    for i in tenors:
        df[str(i) + col_name + ' Change'] = df.loc[:, str(i) + col_name].values - df.loc[:, str(i) + col_name].shift(
            1).values
    return df


def calc_Eig_Val_Vec(change_matrix):
    """
    Find the covariance matrix, solve for eigenvalues and eigenvectors and then return them sorted by eigenvalue (high to low)
    """
    covariance_matrix = np.cov(change_matrix.T)

    eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    return eig_pairs


def plot_PCA_Factors(eig_pairs, ax, plot_cum=1, title=''):
    """
    Plot each factor and the cumulative amount of variance explained. Return the cumulative variance for the third PC
    """
    eig_vals = [i[0] for i in eig_pairs]
    n = len(eig_vals)
    tot = sum(eig_vals)
    var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    if plot_cum:
        with plt.style.context('seaborn-whitegrid'):
            ax.bar(range(n), var_exp, alpha=0.5, align='center',
                   label='individual')
            ax.step(range(n), cum_var_exp, where='mid',
                    label='cumulative')
            ax.set_ylabel('% Explained')
            ax.set_xlabel('Principal components')
            ax.set_title(title)
            ax.legend(loc='best')

    return cum_var_exp[2]


def plot_PCA(matrix_w, ax, country, tenors, var_exp, start, end):
    """
    Plot the three chosen principal components against the tenors
    """
    n = len(tenors)
    ax.plot(tenors, matrix_w[:, 0])
    ax.plot(tenors, matrix_w[:, 1])
    ax.plot(tenors, matrix_w[:, 2])
    ax.legend(['PCA1', 'PCA2', 'PCA3'])
    ax.set_xticks(tenors)
    ax.set_xticklabels(tenors)  # Set text labels.
    ax.set_title(country + ' from ' + start.strftime("%Y-%m-%d") + ' to ' + end.strftime("%Y-%m-%d") + ' (' + str(
        round(var_exp, 1)) + '%)')
    ax.set_xlabel('Tenor')


def plot_YC(ax, x,df, country, end):
    """
    Plot the yield curve
    """
    ax.plot(x,df.loc[end])
    ax.set_title(country + ' Yield Curve on ' + end.strftime("%Y-%m-%d"))



plot_cum = 1

# No standard tenor for yield curves in Quandl. Went through and saw what was returned.
all_tenor = [[1 / 12, 2 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30],
             [5, 7, 10, 20, 30],
             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40],
             [1, 2, 3, 4, 5, 6, 7, 10],
             [5, 10, 20],
[6 / 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
             [1 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 30],
             [1 / 12, 3 / 12, 6 / 12, 2, 3, 5, 10],
             [3 / 12, 6 / 12, 1, 2, 3, 5, 10, 20]]

# [source, table, tenors, all_tenor, col_name, title, country]
curve_list = [('USTREASURY', 'YIELD', [2, 3, 5, 7, 10, 20, 30],all_tenor[0], ' YR', 'USA PCA', 'USA'),
              ('USTREASURY', 'REALYIELD', [5, 7, 10, 20, 30],all_tenor[1], ' YR', 'USA Real Yield PCA', 'USA Real'),
              ('MOFJ', 'INTEREST_RATE_JAPAN', [2, 3, 5, 7, 10, 20, 30],all_tenor[2], 'Y', 'Japan PCA', 'Japan'),
              ('YC', 'CHN', [1, 2, 3, 5, 7, 10],all_tenor[3], '-Year', 'Chinese PCA', 'China'),
              ('YC', 'GBR', [5, 10, 20],all_tenor[4], '-Year', 'UK PCA', 'United Kingdom'),
              ('YC', 'DEU', [2, 3, 5, 7, 10, 20, 30],all_tenor[5], '-Year', 'German PCA', 'Germany'),
              ('YC', 'CAN', [2, 3, 5, 7, 10, 30],all_tenor[6], '-Year', 'Canadian PCA', 'Canada'),
              ('YC', 'AUS', [2, 3, 5, 10],all_tenor[7], '-Year', 'Australian PCA', 'Australia'),
              ('YC', 'KOR', [2, 3, 5, 10, 20],all_tenor[8], '-Year', 'Korean PCA', 'Korea')]

# ('YC', 'ESP', [2, 3, 5, 10, 15], '-Year', 'Spanish Yields'),
# ('YC', 'MEX', [1, 5, 10, 20, 30], '-Year', 'Mexican Yields'),
for source, table, tenors, all_tenor, col_name, title, country in curve_list:
    yields_raw = get_quandl_data(source=source, table=table)

    change_yields = prep_data(yields_raw, tenors, col_name)

    fig = plt.figure(figsize=(9, 6))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 1, 2)

    plot_YC(ax=ax2,x=all_tenor,df=yields_raw, country=country, end=change_yields.index[-1])
    eig_pairs = calc_Eig_Val_Vec(change_yields.iloc[1:, len(tenors):])
    var_exp = plot_PCA_Factors(eig_pairs, ax1, plot_cum, title)
    # Only want 3
    matrix_w = np.hstack((eig_pairs[0][1].reshape(len(tenors), 1),
                          eig_pairs[1][1].reshape(len(tenors), 1),
                          eig_pairs[2][1].reshape(len(tenors), 1)))
    plot_PCA(matrix_w, ax3, title, tenors, var_exp, start=change_yields.index[0],
             end=change_yields.index[-1])
    fig.tight_layout()
    plt.show()
