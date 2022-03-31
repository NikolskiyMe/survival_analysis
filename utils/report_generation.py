"""
Функционал формирования PDF отчета сессии пользователя
"""

from fpdf import FPDF


# ToDo:
#  1. Отображать графики -- найти как вставлять в pdf
#  2. Выделять цветом лучшие значения
#  3. Если неудача - raise CreateReportError
#  4. Расширить столбец имени модели/использовать алиасы

def get_report(report_name: str, result: dict, param: tuple = None) -> None:
    """
    :param report_name: имя файла с отчетом
    :param result: словарь вида {(<имя модели: str>, <список параметров: str>): {<имя метрики>: <результат>, ...}, ...}
    :return: None
    :param: Кортеж с параметрами модели и временем обучения
    """
    pdf = FPDF(format='letter', unit='in')

    pdf.add_page()
    pdf.set_font('Times', '', 10.0)
    epw = pdf.w - 2 * pdf.l_margin
    col_width = epw / 5

    header = ['model', 'parameteres', 'time']
    res_lst = list(result.values())
    header.extend(res_lst[0].keys())

    models = list(result.keys())

    data = [header]

    for model_name, models_res in result.items():
        line = [model_name, 'params', 'time']
        for metric in header[3:]:
            line.append(result[model_name][metric][0])
        data.append(line)

    pdf.set_font('Times', 'B', 14.0)
    pdf.cell(epw, 0.0, str(report_name).replace('_', ' '), align='C')
    pdf.set_font('Times', '', 10.0)
    pdf.ln(0.5)

    th = pdf.font_size

    for row in data:
        for datum in row:
            pdf.cell(col_width, 2 * th, str(datum), border=1)

        pdf.ln(2 * th)

    report_name += '.pdf'
    pdf.output(f'reports/{report_name}', 'F')
