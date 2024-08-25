from xlsxwriter import Workbook


class ExcelUtils:

    def __init__(self, wb: Workbook, name="save_to_excel"):
        self.name = name
        self.wb = wb
        self.bold = wb.add_format({"bold": True})
        self.bold.set_align("center")
        self.bold.set_valign("vcenter")
        self.center = wb.add_format({"align": "center", "valign": "vcenter"})
        self.ws_data = dict()

        self.wb.add_format({"border": 1, "border_color": "black"})

    # --------------------------------------------------------------------

    def add_worksheet(self, ws_name):
        self.wb.add_worksheet(str(ws_name))
        self.ws_data[str(ws_name)] = dict()
        self.ws_data[str(ws_name)]["row"] = 0
        self.ws_data[str(ws_name)]["col"] = 0

    # ---------------------------------------------------------------------

    def get_worksheets(self):
        return self.wb.worksheets()

    # ---------------------------------------------------------------------

    def find_ws_by_name(self, ws_name):
        worksheets = self.get_worksheets()
        for ws in worksheets:
            if ws.name == ws_name:
                return ws
        return

    # ----------------------------------------------------------------------

    def close_wb(self):
        self.wb.close()

    # ---------------------------------------------------------------------

    def write_to_cell(self, ws_name, value, style=None, merge=False, num_col=None, num_row=None):
        ws = self.find_ws_by_name(ws_name=ws_name)
        if ws is None:
            return
        row = self.get_row(ws_name=ws_name)
        col = self.get_col(ws_name=ws_name)
        style = self.get_style(style_name=style)
        if merge:
            ws.merge_range(row, col, row + num_row - 1, col + num_col - 1, value, style)
            self.go_to_xy(ws_name=ws_name, row=row, col=col + num_col)
            return True
        # print("write {} in row {} and col {} in ws {}".format(value, row, col, ws_name))
        # print(row, col, value, style)
        ws.write(row, col, value, style)
        self.next_col(ws_name=ws_name)
        return True

    def get_style(self, style_name):
        if style_name == "bold":
            return self.bold
        elif style_name == "center":
            return self.center
        else:
            return

    def get_row(self, ws_name):
        ws_data = self.ws_data.get(ws_name, None)
        if ws_data is None:
            return
        return ws_data["row"]

    def get_col(self, ws_name):
        ws_data = self.ws_data.get(ws_name, None)
        if ws_data is None:
            return
        return ws_data["col"]

    def next_row(self, ws_name):
        ws_data = self.ws_data.get(ws_name, None)
        if ws_data is None:
            return
        ws_data["row"] += 1
        ws_data["col"] = 0
        self.ws_data[ws_name] = ws_data

    def next_col(self, ws_name):
        ws_data = self.ws_data.get(ws_name, None)
        if ws_data is None:
            return
        ws_data["col"] += 1
        self.ws_data[ws_name] = ws_data

    def go_to_xy(self, ws_name, row, col):
        ws_data = self.ws_data.get(ws_name, None)
        if ws_data is None:
            return
        ws_data["row"] = row
        ws_data["col"] = col
        self.ws_data[ws_name] = ws_data

    def initialize_ws(self, ws_name, data):
        for item in data:
            name = item["name"]
            style = item.get("style", None)
            num_col = item.get("num_col", None)
            num_row = item.get("num_row", None)
            merge = False if num_col is None or num_col == 1 else True
            self.write_to_cell(ws_name=ws_name, value=name, style=style, merge=merge, num_col=num_col, num_row=num_row)
        self.next_row(ws_name=ws_name)


"""

data = [{"name": "test1", "style": "bold"},
        {"name": "test2", "style": "center"},
        {"name": "test1", "style": "bold", "num_col": 4},
        {"name": "test1", "style": "bold"},
        {"name": "test1", "style": "bold"},
        {"name": "test1", "style": "bold"}]

wb = excel_writer.Workbook("test.xlsx")
excel = SaveToExcel(wb=wb)
excel.add_worksheet(ws_name="test1")
print(excel.get_worksheets())
excel.initialize_ws(ws_name="test1", data=data)
#excel.write_to_cell("test1", 0, 0, "test")
#excel.write_to_cell("test1", 0, 1, "test2", excel.bold)
#excel.write_to_cell("test1", 0, 2, "test2", excel.center)
#excel.write_to_cell("test1", 0, 3, "test5", excel.bold, True, 3)
excel.close_wb()
"""
