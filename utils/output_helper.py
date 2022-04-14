def print_report(report_dict):
    print('\n----REPORT BEGIN---------------------------------\n')

    for key, value in report_dict.items():

        print(f'Model: {key[0]}')
        print(f'Fit time: {key[2]} sec.')
        for metric_name, metric_res in value.items():
            print(f'{metric_name}: {metric_res}')

        print(f'\n--------PARAMETERS--------')

        param = key[1]
        param = param.replace("'", '')
        param = param.replace(":", ' =')
        param = param.replace("{", '')
        param = param.replace("}", '')
        param = param.replace("()", '')
        param = param.replace("(", 'BEGIN PARAMS [')
        param = param.replace(")", '] END PARAMS')
        param = param.replace(", ", '\n')
        print(f'{param}')
        print(f'-------------------------\n')

    print('\n----REPORT END-----------------------------------\n')
