from .IMDB import IMDBDatasetLoader


class DataLoaderDispatcher:

    def __init__(self, dataset: str, name="A module to select the right dataset loader") -> None:
        self.name = name
        self.dataset = dataset

    def dispatch(self, filepath: str, params: dict = {}):
        if self.dataset == "IMDB":
            return IMDBDatasetLoader(filepath=filepath, load=True, **params)
        else:
            raise Exception(f"ERROR: Invalid or not supported dataset: {self.dataset}")
