from sklearn.model_selection import cross_validate


def cv_results(reg_type, x, y, cv=3, scoring=None, return_train_score=True):
    result = cross_validate(reg_type, x, y, cv=cv, scoring=scoring,
                            return_train_score=return_train_score)
    sorted(result.keys())

    test_result_key = 'test'

    test_result_key += str(scoring[1]) if scoring is None else 'score'

    if return_train_score:
        train_result_key = 'train'
        train_result_key += str(scoring[0])
        return result[test_result_key], result[train_result_key]

    return result[test_result_key]
