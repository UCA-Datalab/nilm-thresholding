from nilm_thresholding.utils.format_list import to_list
from nilm_thresholding.utils.threshold import get_threshold_params


class DataloaderWrapper:
    build_idx_train: list = list()
    build_idx_valid: list = list()
    build_idx_test: list = list()

    def __init__(self, config: dict):
        # Read parameters from config files
        self.build_id_train = config["data"]["building_train"]
        self.build_id_valid = config["data"]["building_valid"]
        self.build_id_test = config["data"]["building_test"]
        self.dates = config["data"]["dates"]
        self.period = config["data"]["period"]
        self.train_size = config["train"]["train_size"]
        self.valid_size = config["train"]["valid_size"]
        self.batch_size = config["train"]["batch_size"]
        self.output_len = config["train"]["model"]["output_len"]
        self.border = config["train"]["model"]["border"]
        self.power_scale = config["data"]["power_scale"]
        self.max_power = config["data"]["max_power"]
        self.return_means = config["train"]["return_means"]
        self.threshold_method = config["data"]["threshold"]["method"]
        self.threshold_std = config["data"]["threshold"]["std"]
        self.thresholds = config["data"]["threshold"]["list"]
        self.min_off = config["data"]["threshold"]["min_off"]
        self.min_on = config["data"]["threshold"]["min_on"]
        self.buildings = to_list(config["data"]["buildings"])
        self.appliances = config["data"]["appliances"]

        self.num_buildings = len(self.buildings)
        self._buildings_to_idx()

        # Set the parameters according to given threshold method
        if self.threshold_method != "custom":
            (
                self.thresholds,
                self.min_off,
                self.min_on,
                self.threshold_std,
            ) = get_threshold_params(self.threshold_method, self.appliances)

    def _buildings_to_idx(self):
        """
        Takes the list of buildings ID and changes them to their corresponding
        index.
        """

        # Train, valid and tests buildings must contain the index, not the ID of
        # the building. Change that
        if self.build_id_train is None:
            self.build_idx_train = [i for i in range(self.num_buildings)]
        else:
            self.build_idx_train = []

        if self.build_id_valid is None:
            self.build_idx_valid = [i for i in range(self.num_buildings)]
        else:
            self.build_idx_valid = []

        if self.build_id_test is None:
            self.build_idx_test = [i for i in range(self.num_buildings)]
        else:
            self.build_idx_test = []

        for idx, building in enumerate(self.buildings):
            if (self.build_id_train is not None) and (building in self.build_id_train):
                self.build_idx_train += [idx]
            if (self.build_id_valid is not None) and (building in self.build_id_valid):
                self.build_idx_valid += [idx]
            if (self.build_id_test is not None) and (building in self.build_id_test):
                self.build_idx_test += [idx]

        assert (
            len(self.build_idx_train) > 0
        ), f"No ID in build_id_train matches the ones of buildings."
        assert (
            len(self.build_idx_valid) > 0
        ), f"No ID in build_id_valid matches the ones of buildings."
        assert (
            len(self.build_idx_test) > 0
        ), f"No ID in build_id_test matches the ones of buildings."
