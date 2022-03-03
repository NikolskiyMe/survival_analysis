
"""
Функционал формирования PDF отчета сессии пользователя

ToDo List
1. Сортировать по качеству
2. Отображать графики -- найти как вставлять в pdf
3. Выделять цветом
"""

from fpdf import FPDF


# ToDo: после получения результатов -> запись в таблицу
#  // индексировать одинаковые модели
def get_report(report_name: str):
    pdf = FPDF(format='letter', unit='in')

    pdf.add_page()
    pdf.set_font('Times', '', 10.0)
    epw = pdf.w - 2 * pdf.l_margin
    col_width = epw / 5

    # ToDo: рефактор
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
