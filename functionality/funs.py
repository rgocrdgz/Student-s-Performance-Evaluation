
import os
import joblib
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_auc_score, confusion_matrix, roc_curve

# Set a custom color palette:
colors = ['red','darksalmon','lightseagreen','dodgerblue','navy']

color = ['firebrick','red','darksalmon','tomato',
         'seagreen','lightseagreen','olive','green',
         'dodgerblue','navy','blue','royalblue']

working = os.getcwd()
dirname = os.path.dirname(working)



def eda_plotter(df, variable, classes, facet_col=None):
    '''
    Visualizes data using bar plots for exploratory data analysis (EDA).

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        variable (str or list): The variable(s) to plot.
        classes (list of lists): The classes/categories for each variable.
        facet_col (str, optional): The column for creating facets. Default is None.

    Returns:
        None
    '''

    # Check if variable is a list or a single variable
    if not isinstance(variable, list):
        grades = np.sort(df['Grade'].unique())

        # Check if facet_col is specified
        if facet_col is not None:
            cols = ['Grade', variable, facet_col]
        else:
            cols = ['Grade', variable]

        # Create a bar plot
        fig = px.bar(
            (df
             .groupby(cols)['Grade']
             .count()
             .to_frame('Count')
             .reset_index()),
            x='Grade',
            y='Count',
            color=variable,
            facet_col=facet_col,
            category_orders={'Grade': grades, variable: classes},
            labels={'Count': ''},
            color_discrete_map=dict(zip(classes, np.flip(colors)[:len(classes)]))
        )

        # Update layout settings
        fig.update_layout(
            showlegend=True,
            height=500,
            width=1650,
            template='plotly_white',
            title='Grades by ' + variable,
            yaxis_range=[0, 28],
            yaxis_title='# Students'
        )

        fig.show()
    
    else:
        # Create subplots for multiple variables
        fig = make_subplots(
            rows=1,
            cols=len(classes),
            shared_yaxes=True,
            subplot_titles=variable,
            horizontal_spacing=0.025,
            vertical_spacing=0.01,
        )

        for i, var in enumerate(variable):
            # Group data by 'Grade' and current variable
            data = (df
                    .groupby(['Grade', var])['Grade']
                    .count()
                    .to_frame('Count')
                    .reset_index())

            for k, category in enumerate(classes[i]):
                # Filter data for the current category
                sub = data[data[var] == category]

                # Add bar trace to the subplot
                fig.add_trace(
                    go.Bar(
                        x=sub['Grade'],
                        y=sub['Count'],
                        name=category,
                        offsetgroup=0,
                        marker={'color': np.flip(colors)[k]},
                        showlegend=True if i == 2 else False,
                    ),
                    row=1,
                    col=i + 1
                )

        # Update layout settings for subplots
        fig.update_layout(
            showlegend=True,
            height=500,
            width=1650,
            template='plotly_white',
            barmode='group',
            yaxis_range=[0, 28],
            yaxis_title='# Students'
        )

        fig.show()


def grades_distribution(sets, names):
    '''
    Calculate and visualize the distribution of grades for multiple sets.

    Args:
        sets (list): A list of sets, where each set contains grades.
        names (list): A list of names corresponding to the sets.

    Returns:
        dict: A dictionary representing the weights of each grade in the 'Train Set'.

    '''

    # Create subplots with a domain plot for each name
    fig = make_subplots(
        rows=1,
        cols=len(names),
        subplot_titles=names,
        specs=[[{'type':'domain'}, {'type':'domain'}]])

    # Iterate over the names and corresponding sets
    for i, name in enumerate(names):

        # Create a DataFrame to calculate the grade distribution
        df = (pd.DataFrame({'Grade':sets[i]})
              .groupby('Grade')['Grade'].count()
              .to_frame('Count')
              .reset_index()
              .sort_values('Grade'))

        grades = df['Grade'].sort_values().unique()

        # Add a pie trace to the subplot
        fig.add_trace(
            go.Pie(
                labels=df['Grade'],
                values=df['Count'], 
                name=name,
                marker={'colors':color[0:len(grades)+1]},  # Assuming 'grades' is defined elsewhere
                sort=False),
            1, i+1)
        
        # If the name is 'Train Set', store the weights for each grade
        if name == 'Train Set':
            # Get the weights for each grade - the number of occurrences in the train set
            weights = dict(zip(df['Grade'].tolist(), df['Count'].tolist()))

    # Update trace settings
    fig.update_traces(textposition='inside', textinfo='percent+label')

    # Update layout settings for the figure
    fig.update_layout(
        title={'text':'Grades % Distribution Train & Test Sets','font_size':20},
        showlegend=False,
        height=650,
        width=1650,
        template='plotly_white')
    
    # Show the figure
    fig.show()

    # Return the weights dictionary
    return weights


def cv_models_performance(estimators, Transformer, X, y, metrics, cv, variance_threshold=None, best=False):
    '''
    Perform cross-validation and evaluate the performance of multiple models.
    Iterate over the given estimators and create a pipeline with each estimator.
    Perform cross-validation on each pipeline using the given feature matrix 'X' and response variable 'y'.
    Calculate various performance metrics for each model.
    Return DataFrames containing the performance metrics for the training and validation sets.

    Parameters:
        estimators (list): A list of machine learning estimators to evaluate.
        Transformer: The transformer used to preprocess the data.
        X: The input features.
        y: The target variable.
        metrics (str or list): The scoring metric(s) used for evaluation.
        cv: The cross-validation strategy.
        variance_threshold (float, optional): The variance threshold for feature selection.
        best (bool, optional): Flag indicating if the best estimators are provided with whole preprocessing pipeline.

    Returns:
        train (DataFrame): DataFrame containing performance metrics on the training set.
        validate (DataFrame): DataFrame containing performance metrics on the validation set.
    '''
    
    # List to store the model names.
    names = []  

    # Lists to store performance metrics for training and validation sets.
    accuracy_train = []
    recall_train = []
    weighted_train = []
    auc_train = []
    accuracy_test = []
    recall_test = []
    weighted_test = []
    auc_test = []
    
    # Preprocessor for data transformation.
    preprocessor = Transformer  

    # Loop over the estimators.
    for estimator in estimators:
        if not best:
            # Create the pipeline without variance threshold.
            if variance_threshold is None:
                model = make_pipeline(preprocessor, estimator)
                i = 1
            else:
                # Create the pipeline with variance threshold.
                model = make_pipeline(
                    preprocessor,
                    VarianceThreshold(threshold=variance_threshold),
                    estimator)
                i = 2
        else:
            # Use the best estimator pipeline.
            model = estimator
            i = 1

        # Perform cross-validation and compute performance metrics.
        cross = cross_validate(model, X, y, scoring=metrics, cv=cv, return_train_score=True)

        # Extract the name of the model.
        name = model.steps[i][0].replace('classifier', '')

        # Append penalty information if available.
        if 'penalty' in model.steps[i][1].get_params().keys():
            penalty = model.steps[i][1].get_params().get('penalty')
            if penalty is not None:
                name = name + '_' + penalty
                names.append(name)
            else:
                names.append(name)
        else:
            names.append(name)

        # Store performance metrics for training set.
        accuracy_train.append(cross['train_accuracy'].mean())
        recall_train.append(cross['train_recall_macro'].mean())
        weighted_train.append(cross['train_precision_macro'].mean())
        auc_train.append(cross['train_auc'].mean())

        # Store performance metrics for validation set
        accuracy_test.append(cross['test_accuracy'].mean())
        recall_test.append(cross['test_recall_macro'].mean())
        weighted_test.append(cross['test_precision_macro'].mean())
        auc_test.append(cross['test_auc'].mean())

    # Create DataFrames for training and validation performance metrics.
    train = pd.DataFrame({
        'Model': names,
        'Accuracy': accuracy_train,
        'Recall weighted': recall_train,
        'Precision weighted': weighted_train,
        'AUC': auc_train}).round(4)

    validate = pd.DataFrame({
        'Model': names,
        'Accuracy': accuracy_test,
        'Recall weighted': recall_test,
        'Precision weighted': weighted_test,
        'AUC': auc_test}).round(4)

    return train, validate


def models_performance_train_test(estimators, Transformer, X_train, y_train, X_test, y_test, classes, variance_threshold=None, best=False):
    '''
    Evaluate the performance of multiple models on training and test sets.
    Iterate over the given estimators and create a pipeline with each estimator.
    Fit each model on the training data and calculate performance metrics on both training and test sets.
    Return DataFrames containing the performance metrics for the training and test sets.

    Args:
        estimators (list): List of estimators to evaluate.
        Transformer: Preprocessing transformer.
        X_train (array-like): The feature matrix of the training set.
        y_train (array-like): The response variable of the training set.
        X_test (array-like): The feature matrix of the test set.
        y_test (array-like): The response variable of the test set.
        classes (list): The list of class labels.
        variance_threshold (float or None): Threshold below which features are removed.
        best (bool, optional): Flag indicating if the best estimators are provided with whole preprocessing pipeline.

    Returns:
        train (DataFrame): DataFrame containing performance metrics on the training set.
        validate (DataFrame): DataFrame containing performance metrics on the validation set.
    '''

    # List to store the models' names.
    names =[]

    # Lists to store performance metrics for training and validation sets.
    accuracy_train = []
    recall_train = []
    precision_train = []
    auc_train = []
    accuracy_test = []
    recall_test = []
    precision_test = []
    auc_test = []

    preprocessor = Transformer

    # Loop over the estimators.
    for estimator in estimators:
        if not best:
            # Create the pipeline without variance threshold.
            if variance_threshold is None:
                model = make_pipeline(preprocessor, estimator)
                i = 1
            else:
                # Create the pipeline with variance threshold.
                model = make_pipeline(
                    preprocessor,
                    VarianceThreshold(threshold=variance_threshold),
                    estimator)
                i = 2
        else:
            # Use the best estimator pipeline.
            model = estimator
            i = 1

        # Extract the name of the model.
        name = model.steps[i][0].replace('classifier','')

        if 'penalty' in model.steps[i][1].get_params().keys():
            penalty = model.steps[i][1].get_params().get('penalty')
            if not penalty == None:
                name = name + '_' + penalty
                names.append(name)
            else:
                names.append(name)
        else:
            names.append(name)

        # Fit the model on the training data.
        model.fit(X_train, y_train)

        # Make predictions on the training and test sets.
        p_train = model.predict(X_train)
        p_test = model.predict(X_test)

        # Calculate performance metrics for training set.
        accuracy_train.append(accuracy_score(y_train, p_train))
        recall_train.append(recall_score(y_train, p_train, average='weighted', zero_division=0))
        precision_train.append(precision_score(y_train, p_train, average='weighted', zero_division=0))
        auc_train.append(roc_auc_score(y_train, model.predict_proba(X_train), average='weighted', multi_class='ovo', labels=classes))

        # Calculate performance metrics for test set.
        accuracy_test.append(accuracy_score(y_test, p_test))
        recall_test.append(recall_score(y_test, p_test, average='weighted', zero_division=0))
        precision_test.append(precision_score(y_test, p_test, average='weighted', zero_division=0))
        auc_test.append(roc_auc_score(y_test, model.predict_proba(X_test), average='weighted', multi_class='ovo', labels=classes))

    # Create DataFrames for training and validation performance metrics.
    train = pd.DataFrame({
        'Model':names,
        'Accuracy':accuracy_train,
        'Recall weighted':recall_train,
        'Precision weighted':precision_train,
        'AUC':auc_train}).round(4)

    test = pd.DataFrame({
        'Model':names,
        'Accuracy':accuracy_test,
        'Recall weighted':recall_test,
        'Precision weighted':precision_test,
        'AUC':auc_test}).round(4)

    return train, test


def performance_plotter(train, validate, name, color):
    '''
    Plot the performance metrics for training and validation sets.
    Reshape the DataFrames for plotting using the 'melt' function.
    Create a subplot with two subplots: one for the training set and one for the validation set.
    Add bar plots for each performance metric and each model to the respective subplots.
    Customize the layout and display the plot.

    Args:
        train (pd.DataFrame): DataFrame with performance metrics for the training set.
        validate (pd.DataFrame): DataFrame with performance metrics for the validation set.
        name (str): Name of the dataset (e.g., 'Test Set').
        color (list): List of colors for the plot.

    Returns:
        None
    '''
    
    trains = train.melt(id_vars='Model', var_name='Metric')
    valids = validate.melt(id_vars='Model', var_name='Metric')

    fig = make_subplots(rows=1, cols=2, 
                        horizontal_spacing=0.025, 
                        vertical_spacing=0.01, 
                        subplot_titles=['Train Set', name + ' Set'], 
                        shared_yaxes=True)
    
    for i, model in enumerate(trains.Model.unique()):

        fig.add_trace(go.Bar(x=trains.loc[trains['Model'] == model]['Metric'], 
                                y=trains.loc[trains['Model'] == model]['value'],
                                name=model, 
                                hovertemplate='<br>'.join(['<b>%{x}:</b> \t %{y}']),
                                marker={'color':color[i]}),
                            row=1, col=1)
        
        fig.add_trace(go.Bar(x=valids.loc[valids['Model'] == model]['Metric'], 
                                y=valids.loc[valids['Model'] == model]['value'],
                                name=model, 
                                hovertemplate='<br>'.join(['<b>%{x}:</b> \t %{y}']),
                                marker={'color':color[i]},
                                showlegend=False),
                            row=1, col=2)

    fig.update_layout(showlegend=True, 
                    height=600, width=1650, 
                    template='plotly_white',
                    yaxis_range = [0,1.1])

    fig.show()


def comparison_plotter(metric, data, names, color):
    '''
    Create a performance comparison plot using Plotly.

    Args:
        validation (list of pd.DataFrame): List of DataFrames containing the validation data for each model.
        metric (str): The metric to plot on the y-axis.
        color (list of str): List of colors for each model.

    Returns:
        None (displays the plot)

    '''
    
    # Create a subplots figure with 1 row and 3 columns.
    fig = make_subplots(
        rows=1, 
        cols=len(names),
        horizontal_spacing=0.025,
        vertical_spacing=0.01,
        subplot_titles=names,
        shared_yaxes=True)

    # Iterate over the validation data.
    for i, df in enumerate(data):

        # Melt the DataFrame to have a 'Model' and 'Metric' column.
        df = df[['Model', metric]].melt(id_vars='Model', var_name='Metric')

        # Iterate over the unique models in the DataFrame.
        for j, model in enumerate(df.Model.unique()):

            # Add a bar trace to the figure for each model.
            fig.add_trace(
                go.Bar(
                    x=df.loc[df['Model'] == model]['Metric'], 
                    y=df.loc[df['Model'] == model]['value'],
                    name=model, 
                    hovertemplate='<br>'.join(['<b>%{x}:</b> \t %{y}']),
                    marker={'color':color[j]},
                    showlegend=True if i==0 else False),
                row=1,
                col=i+1)

    # Update x-axis to hide tick labels.
    fig.update_xaxes(showticklabels=False)

    # Update layout settings for the figure.
    fig.update_layout(
        title={'text':metric + ' Perfromance','font_size':20},
        showlegend=True, 
        height=500,
        width=1650,
        template='plotly_white',
        yaxis=dict(range=[0, 1.1]))

    # Display the figure.
    fig.show()


def save_best_estimators(estimators):
    """
    Saves the best estimators/models to disk.

    Args:
        estimators (list): List of trained estimator/model objects.

    Returns:
        None
    """

    for model in estimators:
        # Extract the name of the model
        name = model.steps[1][0].replace('classifier', '')

        if 'penalty' in model.steps[1][1].get_params().keys():
            penalty = model.steps[1][1].get_params().get('penalty')

            if penalty is not None:
                name = name + '_' + penalty

        # Define the filename
        filename = os.path.join(dirname, 'working/best_estimators/' + name + '.joblib')

        # Save the model to disk
        joblib.dump(model, filename)


def load_best_estimators():
    '''
    Loads the best estimators/models from disk.

    Returns:
        list: List of loaded estimator/model objects.
    '''

    file_names = []

    # Read the file names from the text file.
    with open(os.path.join(dirname, 'working/best_file_names.txt'), 'r') as file:
        file_names = [line.strip('\n') for line in file]

    best_estimators = []

    # Check if all the required files exist in the directory.
    directory = os.path.join(dirname, 'working/best_estimators/')
    for file_name in file_names:
        # Load the models from the directory.
        path = os.path.join(directory, file_name)
        estimator = joblib.load(path)
        best_estimators.append(estimator)

    return best_estimators


def classification_report(model, y_train, X_train, y_test, X_test, classes, roc_plot=True, binarizer=None):
    '''
    Generate a metrics report from a predictive classification model.

    Args:
        model (sklearn.pipeline.Pipeline): The predictive trained classification model.
        y_train (array-like): The response variable for the training set.
        X_train (array-like): The feature matrix for the training set.
        y_test (array-like): The response variable for the test set.
        X_test (array-like): The feature matrix for the test set.
        classes (list): The list of class labels.
    '''

    p_train = model.predict(X_train)
    p_test = model.predict(X_test)

    accuracy_train = accuracy_score(y_train, p_train)
    recall_train = recall_score(y_train, p_train, average='weighted', zero_division=0)
    precision_train = precision_score(y_train, p_train, average='weighted', zero_division=0)
    auc_train = roc_auc_score(y_train, model.predict_proba(X_train), average='weighted', multi_class='ovo', labels=classes)

    accuracy_test = accuracy_score(y_test, p_test)
    recall_test = recall_score(y_test, p_test, average='weighted', zero_division=0)
    precision_test = precision_score(y_test, p_test, average='weighted', zero_division=0)
    auc_test = roc_auc_score(y_test, model.predict_proba(X_test), average='weighted', multi_class='ovo', labels=classes)

    print('\t\t TRAIN \t TEST\n')
    print('Accuracy: \t {:.3f} \t {:.3f}'.format(accuracy_train, accuracy_test))
    print('Recall: \t {:.3f} \t {:.3f}'.format(recall_train, recall_test))
    print('Precision: \t {:.3f} \t {:.3f}'.format(precision_train, precision_test))
    print('\nAUC: \t\t {:.3f} \t {:.3f}'.format(auc_train, auc_test))

    if roc_plot:
        roc_auc_plot(model, X_train, y_train, X_test, y_test, binarizer)


def roc_auc_plot(model, X, y, binarizer):

    # Create a subplots figure with 1 row and 3 columns.
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=['Train Set','Test Set'],
        horizontal_spacing=0.025,
        vertical_spacing=0.01,
        shared_yaxes=True)
    
    for i in range(0, len(X)):

        y_score = model.predict_proba(X[i])
        y_true = binarizer.transform(y[i])

        for j, x in enumerate(binarizer.classes_):

            fpr, tpr, _ = roc_curve(y_true[:,j], y_score[:,j])
            auc = (fpr, tpr)

            # Add a bar trace to the figure for each model.
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=x + ' (AUC=' + str(np.round(np.mean(auc),3)) + ')',
                    hovertemplate='<br>'.join([
                        '<b>FPR:</b> \t %{x}',
                        '<b>TPR:</b> \t %{y}']),
                    marker={'color':color[j]},
                    showlegend=True if i == 0 else False),
                row=1,
                col=i+1)          

        fig.add_trace(
            go.Scatter(
                x=[0.0,1.0],
                y=[0.0,1.0],
                mode='lines',
                name='Chance Level (AUC=0.5)',
                line = dict(color='grey', width=1, dash='dash'),
                showlegend=True),
            row=1,
            col=i+1)

    # Update layout settings for the figure.
    fig.update_layout(
        title={'text':'ROC-AUC Plot','font_size':20},
        showlegend=True, 
        height=650,
        width=1650,
        template='plotly_white',
        yaxis=dict(range=[0,1]),
        xaxis=dict(range=[0,1]))

    # Display the figure.
    fig.show()


def confusion_matrix_plot(model, X, y):

    names = ['Train Set','Test Set']

    # Create a subplots figure with 1 row and 3 columns.
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=names,
        horizontal_spacing=0.025,
        vertical_spacing=0.01,
        shared_yaxes=True)
    
    for i in range(0, len(X)):
    
        cm = (pd.DataFrame(
            confusion_matrix(
                y[i], 
                model.predict(X[i]), 
                labels=model.classes_),
            index=model.classes_, 
            columns=model.classes_))

        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=model.classes_,
                y=model.classes_,
                name=names[i],
                texttemplate='%{z}',
                hovertemplate='<br>'.join([
                    'Real: %{y}',
                    'Predicted: %{x}',
                    'Count: %{z}']),
                xgap = 5,
                ygap = 5,
                colorscale='Mint'),
            row=1, col=i+1)

    fig.update_yaxes(
        autorange='reversed',
        title_text='Real Grade', 
        row=1, 
        col=1)
    
    fig.update_xaxes(
        title_text='Predicted Grade', 
        row=1,
        col=1)
    
    fig.update_xaxes(
        title_text='Predicted Grade', 
        row=1,
        col=2)

    fig.update_traces(showscale=False)

    fig.update_layout(
            title={'text':'Confusion Matrix','font_size':20},
            showlegend=False, 
            height=650,
            width=1650,
            template='plotly_white')

    fig.show()


def logit2prob(logodds):
    odds = np.exp(logodds)
    return odds / (1 + odds)


def linear_coefficients(model, model_name, plot=True, logodds=True, proba=False):
    '''
    Retrieves and visualizes the linear coefficients of a model for each variable.

    Args:
        model (object): The trained model object.
        model_name (str): The name of the model.
        plot (bool, optional): Whether to plot the coefficients. Defaults to True.
        logodds (bool, optional): Whether to display the coefficients as log(odds). Defaults to True.
        proba (bool, optional): Whether to display the coefficients as probabilities. Defaults to False.

    Returns:
        pandas.DataFrame: DataFrame containing the coefficients for each variable.
    '''

    # Extract the indices of the selected variables from the feature selection step.
    idx = np.squeeze(np.where(model.named_steps['pipeline']['variancethreshold'].get_support() == True))
    variables = model.named_steps['pipeline']['columntransformer'].get_feature_names_out()[idx]
    variables = [v.replace('onehotencoder__', '').replace('_', ': ') for v in variables]
    coefs = pd.DataFrame({'Variable': variables})

    grades = model.named_steps[model_name].classes_
    
    # Retrieve the coefficients for each grade and merge them into the DataFrame.
    for i, c in enumerate(grades):
        coefs = coefs.merge(pd.DataFrame({'Variable': variables, c: model.named_steps[model_name].coef_[i]}), on='Variable')

    # Identify the indices of variables with all zero coefficients.
    idxs = (coefs.select_dtypes(np.number) == 0).all(axis=1)
    idxs = idxs.loc[idxs == True].index

    if proba:
        logodds = False
        df = (coefs
            .loc[~coefs.index.isin(idxs)]
            .replace(0, np.nan)
            .select_dtypes(np.number)
            .applymap(lambda x: logit2prob(x))
            .assign(Variable=coefs['Variable']))

        data = df.copy()

        labels = {'Coefficient': 'Probability', 'Variable': ''}
        title = {'text': 'Probability by Attribute for each Grade', 'font_size': 20}

    if logodds:
        df = (coefs
            .loc[~coefs.index.isin(idxs)]
            .replace(0, np.nan))

        data = df.copy()

        labels = {'Coefficient': 'log(odds)', 'Variable': ''}
        title = {'text': 'log(odds) by Attribute for each Grade', 'font_size': 20}

    if plot:
        # Prepare the DataFrame for plotting.
        df = df.melt(id_vars='Variable', var_name='Grades', value_name='Coefficient')

        fig = px.scatter(
            df,
            x='Coefficient',
            y='Variable',
            color='Grades',
            labels=labels,
            color_discrete_map=dict(zip(grades, color)))

        fig.update_traces(marker=dict(size=10))

        fig.update_yaxes(zeroline=False)

        # Update layout settings for the figure.
        fig.update_layout(
            title=title,
            showlegend=True,
            height=650,
            width=1650,
            template='plotly_white',
            margin_pad=20,
            xaxis_range=[-15, 15] if logodds else [-0.01, 1.01],
            yaxis=dict(tickmode='linear', tickfont=dict(size=10)))

        fig.show()

    return data


def probabilities_by_grade(data, grade, grades):

    '''
    Retrieves and visualizes the linear coefficients of a model for each variable by grade.

    Args:
        data (pandas.DataFrame): The data frame contaning all probabilities by variable and grade..
        grade (str): The name the grade to plot.
        grades (list or array type): List containing all category grades.

    Returns:
        pandas.DataFrame: DataFrame containing the coefficients for each variable.
    '''

    # Prepare the DataFrame for plotting.
    df = data.melt(id_vars='Variable', var_name='Grades', value_name='Coefficient')

    fig = px.scatter(
        df.loc[df['Grades'] == grade],
        x='Coefficient',
        y='Variable',
        color_discrete_map=dict(zip(grades, color)))
    
    color_map = dict(zip(grades, color))

    fig.update_traces(
        marker=dict(
            size=10, 
            color=color_map[grade]))

    # Update layout settings for the figure.
    fig.update_layout(
        title=grade,
        showlegend=True,
        height=650,
        width=1650,
        template='plotly_white',
        margin_pad=20,
        xaxis_range=[-0.05, 1.05])

def probabilities_by_grade(data, grades):
    '''
    Generate a Plotly scatter plot to visualize data based on different grades.

    Parameters:
        data (pandas DataFrame): The input data containing the variables and their coefficients.
                                 It should have at least two columns: 'Variable' and one column for each grade.
        grades (list): A list of grade labels corresponding to the columns in the 'data' DataFrame.

    Returns:
        None (displays the plot)

    Example:
        # Sample data with grade columns (AA, BA, BB, CB, CC, DC, DD, Fail)
        data = pd.DataFrame({
            'Variable': ['Var1', 'Var2', 'Var3'],
            'AA': [0.8, 0.6, 0.7],
            'BA': [0.5, 0.4, 0.3],
            # Rest of the grade columns...
        })

        # List of grades in the same order as the columns in 'data'
        grades = ['AA', 'BA', 'BB', 'CB', 'CC', 'DC', 'DD', 'Fail']

        probabilities_by_grade(data, grades)
    '''
    
    # Import Plotly and any required libraries if not already imported.
    import plotly.graph_objects as go
    
    # Create a new figure.
    fig = go.Figure()

    # Prepare the DataFrame for plotting.
    df = data.melt(id_vars='Variable', var_name='Grades', value_name='Coefficient')

    # Define colors for each grade based on the provided 'color' variable.
    # Note: 'color' variable should be defined before calling this function.
    color_map = dict(zip(grades, color))

    # Map colors to each grade and add a new column 'color' in the DataFrame.
    df['color'] = df['Grades'].map(color_map)

    # Iterate through each grade to plot data points for the corresponding variables.
    for grade in df['Grades'].unique().tolist():
        fig.add_trace(
            go.Scatter(
                x=df.loc[df['Grades'] == grade]['Coefficient'],
                y=df.loc[df['Grades'] == grade]['Variable'],
                name=grade,
                marker=dict(color=df.loc[df['Grades'] == grade]['color']),
                mode='markers',
            )
        )

        # Update the marker size for better visibility.
        fig.update_traces(marker=dict(size=10))
        
    # Update the layout with an update menu for toggling between different grades.
    fig.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active=0,
            x=-0.1,
            y=1.155,
            buttons=list(
                [dict(label = 'AA',
                    method = 'update',
                    args = [{'visible':[True,False,False,False,False,False,False,False]},
                            {'title': 'AA',
                            'showlegend':True}]),
                dict(label = 'BA',
                    method = 'update',
                    args = [{'visible':[False,True,False,False,False,False,False,False]},
                            {'title': 'BA',
                            'showlegend':True}]),
                dict(label = 'BB',
                    method = 'update',
                    args = [{'visible':[False,False,True,False,False,False,False,False]},
                            {'title': 'BB',
                            'showlegend':True}]),
                dict(label = 'CB',
                    method = 'update',
                    args = [{'visible':[False,False,False,True,False,False,False,False]},
                            {'title': 'CB',
                            'showlegend':True}]),
                dict(label = 'CC',
                    method = 'update',
                    args = [{'visible':[False,False,False,False,True,False,False,False]},
                            {'title': 'CC',
                            'showlegend':True}]),
                dict(label = 'DC',
                    method = 'update',
                    args = [{'visible':[False,False,False,False,False,True,False,False]},
                            {'title': 'DC',
                            'showlegend':True}]),
                dict(label = 'DD',
                    method = 'update',
                    args = [{'visible':[False,False,False,False,False,False,True,False]},
                            {'title': 'DD',
                            'showlegend':True}]),
                dict(label = 'Fail',
                    method = 'update',
                    args = [{'visible':[False,False,False,False,False,False,False,True]},
                            {'title': 'Fail',
                            'showlegend':True}]),
                ])
            )
        ])

    # Remove the zero baseline for y-axis.
    fig.update_yaxes(zeroline=False)

    # Update layout settings for the figure.
    fig.update_layout(
        showlegend=True,
        height=650,
        width=1650,
        template='plotly_white',
        margin_pad=20,
        xaxis=dict(range=[-0.01, 1.01]),
        yaxis=dict(tickmode='linear', tickfont=dict(size=10)))

    # Display the plot.
    fig.show()



def tree_importance(model, model_name, plot=True):
    '''
    Retrieves and visualizes the feature importances of a tree-based model.

    Args:
        model (object): The trained tree-based model object.
        model_name (str): The name of the model.
        plot (bool, optional): Whether to plot the feature importances. Defaults to True.

    Returns:
        pandas.DataFrame: DataFrame containing the feature importances.
    '''

    # Extract the indices of the selected variables from the feature selection step.
    idx = np.squeeze(np.where(model.named_steps['pipeline']['variancethreshold'].get_support() == True))
    variables = model.named_steps['pipeline']['columntransformer'].get_feature_names_out()[idx]
    variables = [v.replace('onehotencoder__', '').replace('_', ': ') for v in variables]
    importance = pd.DataFrame({'Variable': variables, 'Importance': model.named_steps[model_name].feature_importances_})

    # Sort the importance values in descending order and remove zero importances.
    importance = (importance
                  .sort_values('Importance', ascending=False)
                  .replace(0, np.nan)
                  .dropna())

    if plot:
        fig = px.bar(
            importance,
            x='Importance',
            y='Variable')

        fig.update_traces(marker_color='royalblue')

        # Update layout settings for the figure.
        fig.update_layout(
            title={'text': 'Feature Importance', 'font_size': 20},
            showlegend=True,
            height=650,
            width=1650,
            template='plotly_white',
            margin_pad=20,
            yaxis=dict(tickmode='linear', tickfont=dict(size=10)))

        # Display the figure.
        fig.show()

    return importance



# Unused functions:

def params_writer(models):
    '''
    Writes the best parameters of the models to text files.

    Args:
        models (list): List of trained model objects.

    Returns:
        None
    '''

    names = []

    for model in models:
        # Extract the name of the model.
        name = (model
                .best_estimator_
                .steps[1][0]
                .replace('classifier', ''))

        if 'penalty' in model.best_estimator_.steps[1][1].get_params().keys():
            penalty = (model
                       .best_estimator_
                       .steps[1][1]
                       .get_params()
                       .get('penalty'))

            if penalty is not None:
                name = name + '_' + penalty

        with open('best_params/' + name + '.txt', 'w') as file:
            for key, value in model.best_params_.items():
                if '__estimator' in key:
                    file.write('{}, {}\n'.format(key, type(model.best_params_[key])))
                    
                    # Write the parameters of the nested estimator to a separate file.
                    with open('best_params/' + key.replace('classifier__', '_') + '.txt', 'w') as estimator:
                        for k, v in model.best_params_[key].get_params().items():
                            estimator.write('{}, {}\n'.format(k, v))
                    estimator.close()
                    continue
                file.write('{}, {}\n'.format(key, value))
        file.close()

        names.append(name)

    with open('model_names.txt', 'w') as file:
        [file.write('{} \n'.format(name)) for name in names]