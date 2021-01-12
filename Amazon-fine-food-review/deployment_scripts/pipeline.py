from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.linear_model import SGDClassifier

import preprocessors as pp
import config

review_pipe = Pipeline(
    [("f1",FeatureUnion([
        ("p1",Pipeline(
            [
            ('text_cleaner_1',
                pp.TextCleaner(variables=config.TEXT_FEATURES)),
         
            ('text_lematizer_1',
                pp.TextLematizer(variables=config.TEXT_FEATURES)),
             
            ('create_length_feature_1',
                pp.CreateLengthFeature(variables=config.TEXT_FEATURES)),
             
            ('standrd_scaling_numeric_1',
                pp.StandardScalarNumeric(variables=config.NUMERIC_FEATURES)),

            ('drop_features_1',
                pp.DropUnecessaryFeatures()),
            ]
            )),
        ("p2",Pipeline(
            [
            ('text_cleaner_2',
            pp.TextCleaner(variables=config.TEXT_FEATURES)),
         
            ('text_lematizer_2',
                pp.TextLematizer(variables=config.TEXT_FEATURES)),

            ('tfidf_converter_text_2',
                pp.TfidfConverterText()),
            
            ]
            )),

        ("p3",Pipeline(
            [
            ('text_cleaner_3',
            pp.TextCleaner(variables=config.TEXT_FEATURES)),
         
            ('text_lematizer_3',
                pp.TextLematizer(variables=config.TEXT_FEATURES)),
             
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
