#Study Design Classifier
from catboost import Pool, CatBoostClassifier

def get_study_type(test_pool):

    """Predict study design based on title and abstract of an article
        Params:
        test_pool: title and abstract organized in catboost compliant format
        EX:
        test_pool = Pool(df[['abstract', 'title']],
        feature_names=['abstract', 'title'],
        text_features=['abstract', 'title'])
    """
    CATBOOST_MODEL_NAME="study_design_catboost_classifier_7_June_2020.cbm"
    # Instantiating the model
    model = CatBoostClassifier()
    # Loading pretrained classifier
    model.load_model(os.path.join(local_dir,'study-type-classifier',CATBOOST_MODEL_NAME))
    # Predicting on input data
    predictions = model.predict(test_pool)
    return predictions
