import pickle
import pandas as pd
import seaborn as sns
from os.path import exists
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
import math
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from sklearn.utils.class_weight import compute_class_weight

def add_features_to(df):
    vader_analyzer = SentimentIntensityAnalyzer()  
    # Helper functions for sentiment analysis
    def vader_sentiment(text):
        return vader_analyzer.polarity_scores(text)['compound']
    def textblob_sentiment(text):
        return TextBlob(text).sentiment.polarity
    def count_uppercase(text):
        if isinstance(text, str):
            return sum(1 for char in text if char.isupper())
        return 0  # Return 0 if the text is not a string
    def extract_score(text):
        if isinstance(text, str):
        # Search for whole or half-star ratings
            match = re.search(r'\b([1-5](?:\.5)?)\s?(?:stars|\/5)\b', text.lower())
            if match:
                return float(match.group(1)), True  # Return score and True flag
        return None, False  # No score mentioned
    def avg_sentence_length(text):
        if isinstance(text, str): 
            sentences = re.split(r'[.!?]', text)  # Split on sentence-ending punctuation
            sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty sentences
            total_words = sum(len(s.split()) for s in sentences)  # Count words in all sentences
            return total_words / len(sentences) if sentences else 0
        return 0

    # Feature 1: Helpfulness ratio
    df['Helpfulness'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']
    df['Helpfulness'] = df['Helpfulness'].replace([np.inf, -np.inf], 0).fillna(0)
    print("Helpfulness Complete")

    # Feature 2: VADER and TextBlob sentiment scores for title and textx
    df['Vader_Sentiment_Title'] = df['Summary'].apply(lambda x:vader_sentiment(str(x)))
    df['Vader_Sentiment_Text'] = df['Text'].apply(lambda x:vader_sentiment(str(x)))
    print("Vader Complete")

    df['TextBlob_Sentiment_Title'] = df['Summary'].apply(lambda x:textblob_sentiment(str(x)))
    df['TextBlob_Sentiment_Text'] = df['Text'].apply(lambda x:textblob_sentiment(str(x)))
    print("TextBlob Complete")

    # Feature 3: Average sentiment (between title and text for both Vader and TextBlob)
    df['Average_Title_Sentiment'] = (df['Vader_Sentiment_Title'] + df['TextBlob_Sentiment_Title']) / 2
    df['Average_Text_Sentiment'] = (df['TextBlob_Sentiment_Text'] + df['Vader_Sentiment_Text']) / 2
    print("Average Sentiment Complete")
    
    # Feature 4: Review length (number of words)
    df['Review_Length'] = df['Text'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
    print("Length Complete")

    # Feature 5: Non-alphanumeric token count
    df['Non_Alpha_Count'] = df['Text'].apply(lambda x: len(re.findall(r'\W+', x)) if isinstance(x, str) else 0)
    df['Uppercase_Count'] = df['Text'].apply(count_uppercase)
 
    
    # Feature 6: Review contains a score
    df[['Extracted_Score', 'Score_Mentioned']] = (
        df['Text'].apply(lambda x: pd.Series(extract_score(x))))
    
    # Feature 7: Uppercase count 
    df['Uppercase_Count'] = df['Text'].apply(lambda x: sum(1 for char in x if char.isupper()) if isinstance(x, str) else 0)
    
    # Feature 8: Sentence Length
    df['Sentence_Length'] = df['Text'].apply(avg_sentence_length)
    
    df['Helpfulness_ReviewLength'] = df['Helpfulness'] * df['Review_Length']
    df['NonAlpha_ReviewLength'] = df['Non_Alpha_Count'] * df['Review_Length']
    df['Upper_ReviewLength'] = df['Uppercase_Count'] * df['Review_Length']
    df['Text_Sentiment_Length'] = df['Average_Text_Sentiment'] * df['Review_Length']
    
    print("Completed")
    return df

def scale(df, features):
    # Initialize the scaler
    scaler = StandardScaler()

    # Scale the features and assign them back properly using .loc
    df.loc[:, features] = scaler.fit_transform(df[features])

def log_params_to_file(n_estimators, learning_rate, max_depth, boost, accuracy, filename="model_training_log.txt"):
    with open(filename, 'a') as f:
        f.write(
            f"Params: n_estimators={n_estimators}, "
            f"learning_rate={learning_rate}, max_depth={max_depth}, boost={boost}, class_weights=false -> "
            f"Accuracy: {accuracy:.2f}%\n"
        )

def create_Model(X_train_select, X_test_select, Y_train, Y_test):
    param_grid = {
        'boosting_type': ['gbdt'],
        'n_estimators': [200, 250, 300, 350, 400, 450],
        'learning_rate': [0.05, 0.1, .15, 0.2],
        'max_depth': [5]}
    
    classes = np.unique(Y_train)

    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',  
        classes=classes,
        y=Y_train
    )
    class_weight_dict = {cls: 1/weight for cls, weight in zip(classes, class_weights)}
    print(f'the class weight dict: {class_weight_dict}')
    
    # Initialize variables to track the best model and accuracy
    best_model = None
    best_accuracy = 0
    best_params = {}

    # Custom grid search loop
    for boost in param_grid['boosting_type']:
        for n_estimators in param_grid['n_estimators']:
            for learning_rate in param_grid['learning_rate']:
                for max_depth in param_grid['max_depth']:
                    # Initialize the model with the current parameters
                    model = lgb.LGBMClassifier(
                        boosting_type=boost,
                        n_estimators=n_estimators,
                        #class_weight=class_weight_dict,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        random_state = 100,
                        bagging_freq=5,        
                        bagging_fraction=0.8,
                        verbosity=-1
                    )   
                
                # Train the model
                model.fit(X_train_select, Y_train)
                
                # Make predictions
                Y_test_predictions = model.predict(X_test_select)
                
                # Calculate accuracy
                accuracy = accuracy_score(Y_test, Y_test_predictions) * 100
                print(f"Params: n_estimators={n_estimators}, "
                        f"learning_rate={learning_rate}, max_depth={max_depth}, boost={boost} -> "
                        f"Accuracy: {accuracy:.2f}%")
                log_params_to_file(n_estimators, learning_rate, max_depth, boost, accuracy)
                
                # Track the best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_params = {
                        'n_estimators': n_estimators,
                        'learning_rate': learning_rate,
                        'max_depth': max_depth
                    }

    print(f"\nBest Parameters: {best_params}")
    print(f"Best Accuracy: {best_accuracy:.2f}%")
    Y_test_predictions = best_model.predict(X_test_select)
    return best_model, Y_test_predictions
    
def visualize_Model(Y_test, Y_test_predictions):
    # Evaluate your model on the testing set
    print("Accuracy on testing set = ", accuracy_score(Y_test, Y_test_predictions))

    # Plot a confusion matrix
    cm = confusion_matrix(Y_test, Y_test_predictions, normalize='true')
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.title('Confusion matrix of the classifier')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    return

def predictions(model, X_submission_select):
    final_predictions = []
    submission_predictions = model.predict(X_submission_select)

    # Iterate through each row in the test DataFrame
    for i, (row, model_pred) in enumerate(zip(X_submission_select.itertuples(index=False), submission_predictions)):
        extracted_score = getattr(row, 'Extracted_Score', None)
        score_mentioned = getattr(row, 'Score_Mentioned', None)

        # Apply logic for using the mentioned score if it has >50% accuracy
        if score_mentioned != 1 and extracted_score:
            final_predictions.append(math.floor(extracted_score))
        else:
            final_predictions.append(model_pred)

    # Convert final predictions to a NumPy array
    final_predictions = np.array(final_predictions)
    return final_predictions

def create_submission(X_submission, X_submission_select, model):
    X_submission['Score'] = predictions(model, X_submission_select)
    submission = X_submission[['Id', 'Score']]
    submission.to_csv("./data/submission.csv", index=False)
    return

def main():
    # CREATING FEATURES
    
    if exists('./data/X_train.csv'):
        X_train = pd.read_csv("./data/X_train.csv")
    if exists('./data/X_submission.csv'):
        X_submission = pd.read_csv("./data/X_submission.csv")
    else:
        trainingSet = pd.read_csv("./data/train.csv")
        testingSet = pd.read_csv("./data/test.csv")
        # Process the DataFrame
        train = add_features_to(trainingSet)

        # Merge on Id so that the submission set can have feature columns as well
        X_submission = pd.merge(train, testingSet, left_on='Id', right_on='Id')
        X_submission = X_submission.drop(columns=['Score_x'])
        X_submission = X_submission.rename(columns={'Score_y': 'Score'})

        # The training set is where the score is not null
        X_train =  train[train['Score'].notnull()]

        X_submission.to_csv("./data/X_submission.csv", index=False)
        X_train.to_csv("./data/X_train.csv", index=False)
    print("FEATURES DONE")

    # SPLIT INTO TEST AND TRAIN
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_train.drop(columns=['Score']),
        X_train['Score'],
        test_size=1/4.0,
        random_state=0)
    print("SPLIT DONE")
        
    # FEATURE SELECTION
    
    features = [ 'Helpfulness', 
                'HelpfulnessDenominator', 
                'Average_Title_Sentiment' , 
                'Average_Text_Sentiment' , 
                'Review_Length', 
                'Time', 
                'Vader_Sentiment_Title',
                'Vader_Sentiment_Text',
                'TextBlob_Sentiment_Title',
                'TextBlob_Sentiment_Text', 
                'Non_Alpha_Count',  
                'Uppercase_Count', 
                'Sentence_Length',
                'Helpfulness_ReviewLength',
                'NonAlpha_ReviewLength',
                'Upper_ReviewLength',
                'Text_Sentiment_Length' 
    ]
    X_train_select = X_train[features]
    X_test_select = X_test[features]
    X_submission_select = X_submission[features]
    scale(X_train_select, features)
    scale(X_test_select, features)  
    scale(X_submission_select, features)
    print("SCALING DONE")
    
    # CREATE THE MODEL
    
    model, Y_test_predictions = create_Model(X_train_select, X_test_select, Y_train, Y_test)
    print("MODEL DONE")
    
    # GRAPH THE MODEL'S ACCURACY
    
    visualize_Model(Y_test, Y_test_predictions)
    
    # CREATE SUBMISSION
    
    create_submission(X_submission, X_submission_select, model)
    
    return
main()