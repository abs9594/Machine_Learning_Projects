from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.linear_model import SGDClassifier

from ml_modeling import preprocessors as pp
from ml_modeling import config 

review_pipe = Pipeline(
    [
    ('text_cleaner',
                pp.TextCleaner(variables=config.TEXT_FEATURES)),
         
    ('text_lematizer',
                pp.TextLematizer(variables=config.TEXT_FEATURES)),

    ("f1",FeatureUnion([
        ("p1",Pipeline(
            [
            ('create_length_feature',
                pp.CreateLengthFeature(variables=config.TEXT_FEATURES)),
             
            ('standrd_scaling_numeric',
                pp.StandardScalarNumeric(variables=config.NUMERIC_FEATURES)),

            ]
            )),
        ("p2",Pipeline(
            [          

            ('tfidf_converter_text_2',
                pp.TfidfConverterText()),
            
            ]
            )),

        ("p3",Pipeline(
            [
     
            ('increase_summary_weightage_3'
                ,pp.IncreaseSummaryWeightage()),
             
            ('tfidf_converter_summary_3',
                pp.TfidfConverterSummary()),

            ]
            ))

        ])),

        ('Linear_model', SGDClassifier(penalty='l1', loss='log',alpha=0.000001, random_state=42,class_weight='balanced'))
    ]
)


# review_pipe = Pipeline(
#     [
#         ('text_cleaner',
#             pp.TextCleaner(variables=config.TEXT_FEATURES)),
         
#         ('text_lematizer',
#             pp.TextLematizer(variables=config.TEXT_FEATURES)),
         
#         ('create_length_feature',
#             pp.CreateLengthFeature(variables=config.TEXT_FEATURES)),
         
#         ('standrd_scaling_numeric',
#             pp.StandardScalarNumeric(variables=config.NUMERIC_FEATURES)),

#         ('increase_summary_weightage'
#             ,pp.IncreaseSummaryWeightage()),
         
#         ('tfidf_converter_summary',
#             pp.TfidfConverterSummary()),

#         ('tfidf_converter_text',
#             pp.TfidfConverterText()),
        
#         ('drop_features',
#             pp.DropUnecessaryFeatures()),

#         ('Linear_model', SGDClassifier(penalty='l1', loss='log',alpha=0.000001, random_state=42,class_weight='balanced'))
#     ]
# )
