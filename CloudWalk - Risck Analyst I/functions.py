import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score, RocCurveDisplay



def count_previous_day_occurrences(group, id_column):
    '''
    Counts occurrences from the previous day for each row in the group.

    Parameters:
        group (pd.DataFrame): Grouped DataFrame containing transaction data.
        id_column (str): Column name representing the identifier for grouping.

    Returns:
        list: List of counts corresponding to occurrences from the previous day for each row.
    '''
    counts = []
    
    # Iterate over each row in the grouped DataFrame
    for idx, row in group.iterrows():
        # Calculate the previous day
        prev_day = row['transaction_day'] - pd.Timedelta(days=1)
        
        # Count occurrences matching criteria from the previous day
        count = group[
            (group['transaction_day'] == prev_day) & 
            (group['transaction_hour'] <= row['transaction_hour']) & 
            (group[id_column] == row[id_column])
        ].shape[0]
        
        counts.append(count)  # Append the count to the list
    
    return counts


def classify_period(hour):
    '''
    Classifies the time of day based on the given hour.

    Parameters:
        hour (int): Hour of the day (24-hour format)

    Returns:
        str: Period of the day ('morning', 'evening', 'night', 'early_morning')
    '''
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'evening'
    elif 18 <= hour < 24:
        return 'night'
    else:
        return 'early_morning'

    

def plot_kde_plots(df, var):
    '''
    Plot KDE plots for numerical variables colored by 'has_cbk' categories.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data
        var_num (list): List of numerical variable names to plot
        var (list): List of corresponding variable names for plotting

    Returns:
        None
    '''
    plt.figure(figsize=(24, 28))

    for i in range(len(var)):
        plt.subplot(8, 2, i + 1)
        sns.kdeplot(x=df[var[i]], 
                    palette=['green', 'red'], 
                    shade=True, 
                    hue=df['has_cbk'],
                    warn_singular=False)

        plt.title(var[i], fontsize=16)
        plt.xlabel(' ')
        plt.tight_layout()

    plt.show()


def plot_daily_transaction_stats(df):
    '''
    Plot daily transaction statistics (mean, max, min, std).

    Parameters:
        df (pd.DataFrame): DataFrame containing transaction data with a 'transaction_datetime' column.

    Returns:
        None
    '''
    daily_stats = df.groupby(df['transaction_datetime'].dt.date)['transaction_amount'].agg(['mean', 'max', 'min', 'std']).reset_index()

    plt.figure(figsize=(14, 7))
    plt.plot(daily_stats['transaction_datetime'], daily_stats['mean'], label='Mean')
    plt.plot(daily_stats['transaction_datetime'], daily_stats['max'], label='Max')
    plt.plot(daily_stats['transaction_datetime'], daily_stats['min'], label='Min')
    plt.plot(daily_stats['transaction_datetime'], daily_stats['std'], label='Std')

    plt.xlabel('Date')
    plt.ylabel('Transaction Value')
    plt.title('Daily Transaction Value Statistics')
    plt.legend()
    plt.grid(True)
    plt.show()


def train_lightgbm_model(X_train, y_train, X_test, y_test):
    """
    Train a LightGBM model and visualize feature importance.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series or np.array): Training target.
    - X_test (pd.DataFrame): Test features.
    - y_test (pd.Series or np.array): Test target.

    Returns:
    - gbm (lgb.Booster): Trained LightGBM model.
    - importance_df (pd.DataFrame): DataFrame with feature names and importances.
    """
    # Define LightGBM datasets
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    # Define parameters for LightGBM
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    
    # Train the model
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=100,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=10)
    
    # Get feature importance
    feature_importance = gbm.feature_importance()
    feature_names = gbm.feature_name()
    
    # Create a DataFrame for feature importance
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance using LightGBM')
    plt.show()
    
    # Return the trained model and feature importance DataFrame
    return gbm, importance_df


def validation_table(model, dataframe, percentiles):
    '''
    Table with probability and share information

    Parameters:
        model (model): executed model
        dataframe (pd.DataFrame): dataframe to be analyzed
        percentiles (int): number of percentiles

    Returns:
        dataframe (pd.DataFrame): DataFrame with calculated statistics
    '''    
    
    df = dataframe.copy()
    df['prob'] = model.predict_proba(df.drop('has_cbk', axis=1))[:, 1]
    
    df_val_sorted = df.sort_values('prob')
    num_percentiles = percentiles
    percentile_size = len(df_val_sorted) // num_percentiles

    statistics = []

    total_fraud = df_val_sorted['has_cbk'].sum()

    for i in range(num_percentiles):
        start_index = i * percentile_size
        end_index = (i + 1) * percentile_size
        percentile_rows = df_val_sorted.iloc[start_index:end_index]
        
        prob_min = round(percentile_rows['prob'].min(), 6)
        prob_max = round(percentile_rows['prob'].max(), 6)
        fraud_count = percentile_rows['has_cbk'].sum()
        non_fraud_count = len(percentile_rows) - fraud_count
        fraud_rate = fraud_count / percentile_size
        fraud_share = fraud_count / total_fraud
        share_total = len(percentile_rows) / len(df_val_sorted)
        fraud_value_total = percentile_rows.loc[percentile_rows['has_cbk'] == 1, 'transaction_amount'].sum()
        non_fraud_value_total = percentile_rows.loc[percentile_rows['has_cbk'] == 0, 'transaction_amount'].sum()
        
        statistics.append([share_total, prob_min, prob_max, fraud_count, non_fraud_count, fraud_rate, fraud_share, fraud_value_total, non_fraud_value_total])

    statistics_df = pd.DataFrame(statistics, columns=[
        'share', 'min_pro', 'max_pro', 'has_cbk', 'hasnt_cbk', 'cbk_rate', 'cbk_share', 'has_cbk_amount', 'hasnt_cbk_amount'
    ])

    return statistics_df


def train_and_evaluate_model(mod, X_train, X_test, y_train, y_test, balance_ratio):
    '''
    Train and evaluate a machine learning model with scaling and SMOTE balancing.

    Parameters:
        mod (model): The machine learning model to be trained and evaluated
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        y_train (pd.Series): Training labels
        y_test (pd.Series): Testing labels
        balance_ratio (float): The desired balance ratio for SMOTE

    Returns:
        mod_pipe (Pipeline): The trained model pipeline
    '''
    
    scaler = StandardScaler()
    mod_pipe = make_pipeline(scaler, mod)
    
    smote = SMOTE(sampling_strategy=balance_ratio, random_state=42, n_jobs=-1)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train.values, y_train)
   
    mod_pipe.fit(X_train_balanced, y_train_balanced)
    
    # Calculate predictions and probabilities once
    y_pred = mod_pipe.predict(X_test)
    y_pred_proba = mod_pipe.predict_proba(X_test)[:, 1]
    
    accuracy = mod_pipe.score(X_test, y_test)
    print('Accuracy', mod, ': ', accuracy,'\n')
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    
    ConfusionMatrixDisplay.from_estimator(mod_pipe, X_test, y_test, ax=axs[0]).ax_.set_title('Confusion Matrix')
    ConfusionMatrixDisplay.from_estimator(mod_pipe, X_test, y_test, ax=axs[1], normalize='true').ax_.set_title('Normalized Confusion Matrix')
    
    # Plot ROC curve with AUC
    roc_display = RocCurveDisplay.from_estimator(mod_pipe, X_test, y_test, ax=axs[2])
    axs[2].plot([0, 1], [0, 1], ls='--', label='Baseline (AUC = 0.5)')
    axs[2].legend()
    
    print('Classification Report for model', mod, '\n')
    print(classification_report(y_test, y_pred))
    
    print('Model:', mod)
    print(f'Training AUC: {roc_auc_score(y_train_balanced, mod_pipe.predict_proba(X_train_balanced)[:,1])}')
    print(f'Testing AUC: {roc_auc_score(y_test, y_pred_proba)}','\n')
    
    plt.tight_layout()
    plt.show()
    
    return mod_pipe

