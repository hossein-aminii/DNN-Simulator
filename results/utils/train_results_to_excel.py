from general.utils import ExcelUtils
import xlsxwriter as excel_writer
import json


class TrainResultsToExcel:

    def __init__(
        self,
        json_filepath: str,
        excel_filepath: str,
        worksheet_name: str = "test",
        name: str = "A module to covnvert json train results file into excel one!",
    ) -> None:
        self.name = name

        self.json_filepath = json_filepath
        self.excel_filepath = excel_filepath
        wb = excel_writer.Workbook(self.excel_filepath)
        self.ws = worksheet_name
        self.excel_utils = ExcelUtils(wb=wb)
        self.excel_utils.add_worksheet(ws_name=worksheet_name)
        self.add_excel_headers()
        self.num_epochs = 5

    def add_excel_headers(self):
        first_row_data = [
            {"name": "", "style": "bold", "num_col": 1, "num_row": 1},
            {"name": "Embedding", "style": "bold", "num_col": 3, "num_row": 1},
            {"name": "LSTM", "style": "bold", "num_col": 3, "num_row": 1},
            {"name": "Dense", "style": "bold", "num_col": 3, "num_row": 1},
            {"name": "Train Information", "style": "bold", "num_col": 11, "num_row": 1},
        ]
        self.excel_utils.initialize_ws(ws_name=self.ws, data=first_row_data)
        second_row_data = [
            {"name": "", "style": "bold", "num_col": 1, "num_row": 1},
            {"name": "input dim", "style": "bold", "num_col": 1, "num_row": 1},
            {"name": "output dim", "style": "bold", "num_col": 1, "num_row": 1},
            {"name": "input length", "style": "bold", "num_col": 1, "num_row": 1},
            {"name": "units", "style": "bold", "num_col": 1, "num_row": 1},
            {"name": "use bias", "style": "bold", "num_col": 1, "num_row": 1},
            {"name": "input shape", "style": "bold", "num_col": 1, "num_row": 1},
            {"name": "units", "style": "bold", "num_col": 1, "num_row": 1},
            {"name": "use bias", "style": "bold", "num_col": 1, "num_row": 1},
            {"name": "activation", "style": "bold", "num_col": 1, "num_row": 1},
            {"name": "epochs", "style": "bold", "num_col": 1, "num_row": 1},
            {"name": "batch size", "style": "bold", "num_col": 1, "num_row": 1},
            {"name": "loss function", "style": "bold", "num_col": 1, "num_row": 1},
            {"name": "accuracy function", "style": "bold", "num_col": 1, "num_row": 1},
            {"name": "#", "style": "bold", "num_col": 1, "num_row": 1},
            {"name": "train loss", "style": "bold", "num_col": 1, "num_row": 1},
            {"name": "val loss", "style": "bold", "num_col": 1, "num_row": 1},
            {"name": "train accuracy", "style": "bold", "num_col": 1, "num_row": 1},
            {"name": "val accuracy", "style": "bold", "num_col": 1, "num_row": 1},
            {"name": "loss diff", "style": "bold", "num_col": 1, "num_row": 1},
            {"name": "accuracy diff", "style": "bold", "num_col": 1, "num_row": 1},
        ]
        self.excel_utils.initialize_ws(ws_name=self.ws, data=second_row_data)

    def write_to_cell(self, value, type="general"):
        if type == "general":
            self.excel_utils.write_to_cell(
                ws_name=self.ws,
                value=value,
                style="center",
                merge=True,
                num_col=1,
                num_row=self.num_epochs,
            )
        else:
            self.excel_utils.write_to_cell(
                ws_name=self.ws,
                value=value,
                style="center",
                merge=False,
                num_col=1,
                num_row=1,
            )

    def write_exp_data_to_excel(self, exp_data: dict):
        for idx in range(self.num_epochs):
            if idx == 0:  # write general info
                # embedding
                self.write_to_cell(value=exp_data["embedding"]["input_dim"])
                self.write_to_cell(value=exp_data["embedding"]["output_dim"])
                self.write_to_cell(value=exp_data["embedding"]["input_length"])
                # -------------------------------------------------------------------------
                # LSTM
                self.write_to_cell(value=exp_data["LSTM"]["units"])
                self.write_to_cell(value="YES" if exp_data["LSTM"]["use_bias"] else "NO")
                self.write_to_cell(
                    value=f"({exp_data['LSTM']['input_shape'][0]}, {exp_data['LSTM']['input_shape'][1]})"
                )
                # ---------------------------------------------------------------------------
                # Dense
                self.write_to_cell(value=exp_data["Dense"]["units"])
                self.write_to_cell(value="YES" if exp_data["Dense"]["use_bias"] else "NO")
                self.write_to_cell(value=exp_data["Dense"]["activation"])
                # ---------------------------------------------------------------------------
                # some general Train Information
                self.write_to_cell(value=exp_data["epochs"])
                self.write_to_cell(value=exp_data["batch_size"])
                self.write_to_cell(value=exp_data["loss"])
                self.write_to_cell(value=exp_data["accuracy"])
            else:
                self.excel_utils.next_row(ws_name=self.ws)
                for _ in range(14):
                    self.excel_utils.next_col(ws_name=self.ws)
            # --------------------------------------------------------------------------------------------
            key = f"train_results_epoch#{idx+1}"
            data = exp_data[key]
            self.write_to_cell(value=idx + 1, type="specific")
            self.write_to_cell(value=data["train_loss"], type="specific")
            self.write_to_cell(value=data["val_loss"], type="specific")
            self.write_to_cell(value=data["train_accuracy"], type="specific")
            self.write_to_cell(value=data["val_accuracy"], type="specific")
            self.write_to_cell(value=abs(data["train_loss"] - data["val_loss"]), type="specific")
            self.write_to_cell(value=abs(data["train_accuracy"] - data["val_accuracy"]), type="specific")

    def write_data_to_excel(self, data: dict):
        for exp, exp_data in data.items():
            self.excel_utils.write_to_cell(
                ws_name=self.ws, value=exp, style="bold", merge=True, num_col=1, num_row=self.num_epochs
            )
            self.write_exp_data_to_excel(exp_data=exp_data)
            self.excel_utils.next_row(ws_name=self.ws)

    def read_json(self):
        with open(self.json_filepath, "r") as jf:
            return json.load(jf)

    def convert(self):
        data = self.read_json()
        self.write_data_to_excel(data=data)
        self.excel_utils.close_wb()
