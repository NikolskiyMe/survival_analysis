class Info:
    """
    Класс предоставляет методы для вывода информации
    """
    def __init__(self):
        self.ensemble_models = ['ComponentwiseGradientBoostingSurvivalAnalysis',
                                'GradientBoostingSurvivalAnalysis',
                                'RandomSurvivalForest',
                                'ExtraSurvivalTrees']

        self.linear_models = ['CoxnetSurvivalAnalysis',
                              'CoxPHSurvivalAnalysis',
                              'IPCRidge']

        self.ssvm_models = ['HingeLossSurvivalSVM',
                            'FastKernelSurvivalSVM',
                            'FastSurvivalSVM',
                            'MinlipSurvivalAnalysis',
                            'NaiveSurvivalSVM']

        self.trees_models = ['SurvivalTree']

        self.metrics = ['BrierScore',
                        'ConcordanceIndexCensored',
                        'ConcordanceIndexIpcw',
                        'CumulativeDynamicAuc',
                        'IntegratedBrierScore',
                        'AsConcordanceIndexIpcwScorer',
                        'AsCumulativeDynamicAucScorer',
                        'AsIntegratedBrierScoreScorer']

    def models(self):
        info = f'{self.ensemble_models} {self.linear_models} ' \
               f'{self.ssvm_models} {self.trees_models}'

        print(info)

    def metrics(self):
        info = f'{self.metrics}'
        print(info)
