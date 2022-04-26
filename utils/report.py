from fpdf import FPDF


def make_pdf(report_name: str, result) -> None:
    pdf = FPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font('Times', '', 12)
    line_no = 1
    for model, score in result.items():
        _model = str(model[0])
        pdf.cell(0, 10, txt=_model, ln=1)
        param = model[1]
        param = param.replace("'", '')
        param = param.replace(":", ' =')
        param = param.replace("{", '')
        param = param.replace("}", '')

        for _param in param.split(','):
            pdf.cell(0, 10, txt=_param, ln=1)
        _time = f'Fit time: {str(model[2])} sec.'
        pdf.cell(0, 10, txt=_time, ln=1)
        line_no += 1
        for score_name, score_value in score.items():
            _score = f'{score_name}: {score_value}'
            pdf.cell(0, 10, txt=_score, ln=1)

    pdf.output(f'reports/{report_name}.pdf', 'F')


def print_report(report):
    for k, v in report.items():
        print(f'---{k}---\n{v}\n')

