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

        self.metrics = ['brier_score',
                        'c_index_ipcw',
                        'cumulative_dynamic_auc',
                        'integrated_brier_score',
                        'c_index_censored',
                        'AsConcordanceIndexIpcwScorer',
                        'AsCumulativeDynamicAucScorer',
                        'AsIntegratedBrierScoreScorer']

    @property
    def models(self):
        info = f'{self.ensemble_models} {self.linear_models} ' \
               f'{self.ssvm_models} {self.trees_models}'

        return info

    def metrics(self):
        return self.metrics
