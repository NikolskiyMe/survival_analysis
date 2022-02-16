
"""
Функционал формирования PDF отчета сессии пользователя
ToDo: документировать по необходмости
"""

from fpdf import FPDF

pdf = FPDF(format='letter', unit='in')

pdf.add_page()
pdf.set_font('Times', '', 10.0)
epw = pdf.w - 2 * pdf.l_margin
col_width = epw / 5

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

pdf.output('survival_report.pdf', 'F')