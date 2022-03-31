"""
Функционал формирования PDF отчета сессии пользователя
"""

from fpdf import FPDF


# ToDo:
#  1. Занесение в таблицу result
#  2. Отображать графики -- найти как вставлять в pdf
#  3. Выделять цветом лучшие значения
#  4. Если неудача - raise CreateReportError

def get_report(report_name: str, result: dict) -> None:
    """
    :param report_name: имя файла с отчетом
    :param result: словарь вида {<имя модели>: {<имя метрики>: <результат>, ...}, ...}
    :return: None
    """
    pdf = FPDF(format='letter', unit='in')

    pdf.add_page()
    pdf.set_font('Times', '', 10.0)
    epw = pdf.w - 2 * pdf.l_margin
    col_width = epw / 5

    # ToDo: обработка result
    data = [['model', 'c index', 'brier score', 'time', 'params?'],
            ['xxx', 0.00, 0.00, 1.00, '{"a": 1, "b": 2, "c": 3}']]

    pdf.set_font('Times', 'B', 14.0)
    pdf.cell(epw, 0.0, 'Report', align='C')
    pdf.set_font('Times', '', 10.0)
    pdf.ln(0.5)

    th = pdf.font_size

    for row in data:
        for datum in row:
            pdf.cell(col_width, 2 * th, str(datum), border=1)

        pdf.ln(2 * th)

    report_name += '.pdf'
    pdf.output(report_name, 'F')
