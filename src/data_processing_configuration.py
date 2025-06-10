from search_configuration import DataSearcherConfiguration
from plotter_base_configuration import BasePlotterConfiguration
from source_hyg_data_configuration import DataConfigurationStellarCatalogsHYG
from source_stellar_parameters_configuration import DataConfigurationStellarParametersTable
import numpy as np
# from numpy.dtypes import StringDType ## requires numpy 2.0+; incompatible with scipy + astropy


class BasestDataProcessingConfiguration(BasePlotterConfiguration):

	def __init__(self):
		super().__init__()
		self._sources = None
		self._data = None
		self._sun = None
		self._classification_parameters = None

	@property
	def sources(self):
		return self._sources
	
	@property
	def data(self):
		return self._data

	@property
	def sun(self):
		return self._sun
	
	@property
	def classification_parameterization(self):
		return self._classification_parameterization

	@staticmethod
	def convert_roman_numeral_to_integer(roman_numeral):
		if not isinstance(roman_numeral, str):
			raise ValueError("invalid type(roman_numeral): {}".format(type(roman_numeral)))
		mapping = {
			"I" : 1,
			"V" : 5,
			"X" : 10,
			"L" : 50,
			"C" : 100,
			"D" : 500,
			"M" : 1000}
		true_value, previous_value = 0, 0
		for index_at_character in range(len(roman_numeral) - 1, -1, -1):
			character = roman_numeral[index_at_character]
			if character in mapping.keys():
				current_value = mapping[character]
				if current_value < previous_value:
					true_value -= current_value
				else:
					true_value += current_value
				previous_value = current_value
		return true_value

	def pre_initialize_classification_parameterization(self):
		classification_parameterization = dict()
		self._classification_parameterization = classification_parameterization

	def initialize_classification_parameterization_by_spectral_type(self):
		universal_spectral_types = (
			("W", "Wolf-Rayet Stars", "Wolf-Rayet Stars"),
			("O", "Main-Sequence Stars", "Blue Stars\nwith neutral and ionized Helium lines, weak Hydrogen lines"),
			("B", "Main-Sequence Stars", "Blue-White Stars\nwith neutral Helium lines, strong Hydrogen lines"),
			("A", "Main-Sequence Stars", "White Stars\nwith strongest Hydrogen lines, weak ionized Calcium and metal lines"),
			("F", "Main-Sequence Stars", "Yellow-White Stars\nwith strong Hydrogen lines, strong ionized Calcium lines\nwith weak Sodium lines, many ionized metal lines"),
			("G", "Main-Sequence Stars", "Yellow Stars\nwith strong Sodium and ionized Calcium lines, many metal lines"),
			("K", "Main-Sequence Stars", "Orange Dwarfs\nwith strong molecular bands of Titanium Oxide\nwith strong Sodium and Calcium lines"),
			("M", "Main-Sequence Stars", "Red Dwarfs\nwith strong molecular bands of Titanium Oxide\nwith strong Sodium and Calcium lines"),
			("L", "Brown Dwarfs", "Brown Dwarfs\nwith strong alkali metals"),
			("T", "Brown Dwarfs", "Brown Dwarfs\nwith strong Methane absorption"),
			("Y", "Brown Dwarfs", "Brown Dwarfs\nwith strong Ammonia absorption"))
		particular_spectral_types = (
			("DA", "White Dwarfs", "White Dwarfs\nwith Hydrogen lines"),
			("DB", "White Dwarfs", "White Dwarfs\nwith Helium lines"),
			("DC", "White Dwarfs", "White Dwarfs\nwith Continuous Spectrum\n(no lines)"),
			("DO", "White Dwarfs", "White Dwarfs\nwith ionized Helium lines"),
			("DQ", "White Dwarfs", "White Dwarfs\nwith Carbon lines"),
			("DX", "White Dwarfs", "White Dwarfs\nUncertain Composition"),
			("DZ", "White Dwarfs", "White Dwarfs\nwith metal lines"),
			("DZA", "White Dwarfs", "White Dwarfs\nwith metal lines"))
		generic_spectral_types = (
			("D", "White Dwarfs", "Uncategorized\nWhite Dwarfs"),)
		spectral_sub_types = tuple([
			str(number)
				for number in range(
					0,
					10)])
		self._classification_parameterization["spectral type"] = {
			"universal" : universal_spectral_types,
			"particular" : particular_spectral_types,
			"generic" : generic_spectral_types,
			"sub-type" : spectral_sub_types}

	def initialize_classification_parameterization_by_luminosity_class(self):
		universal_luminosity_classes = (
			("VII", "White Dwarfs"),
			("VI", "Sub-Dwarfs"),
			("IV", "Sub-Giants"),
			("V", "Main-Sequence Stars"),
			("0", "Hyper-Giants"),
			("III", "Normal Giants"),
			("II", "Bright Giants"))
		particular_luminosity_classes = (
			("Ia+", "Extremely Luminous\nSuper-Giants"),
			("Ia", "More Luminous\nSuper-Giants"),
			("Iab", "Intermediate Luminous\nSuper-Giants"),
			("Ib", "Less Luminous\nSuper-Giants"))
		generic_luminosity_classes = (
			("I", "Super-Giants"),)
		self._classification_parameterization["luminosity class"] = {
			"universal" : universal_luminosity_classes,
			"particular" : particular_luminosity_classes,
			"generic" : generic_luminosity_classes}

class BaseDataProcessingConfiguration(BasestDataProcessingConfiguration):

	def __init__(self):
		super().__init__()

	@staticmethod
	def get_temperatures(bv_color_index):

		def get_temperatures(bv_color_index, interval):
			if interval == "ballasteros":
				first_denominator = 0.92 * bv_color_index + 1.7
				second_denominator = 0.92 * bv_color_index + 0.62
				temperature = 4600 * ((1 / first_denominator) + (1 / second_denominator))
			else:
				raise ValueError("invalid interval: {}".format(interval))
			return temperature

		def get_boolean_indices_at_interval_of_bv_color_index(bv_color_index, interval):
			if interval == "ballasteros":
				lower_bound = 0
				upper_bound = 1.5
				if isinstance(bv_color_index, np.ndarray):
					boolean_indices = (
						(lower_bound <= bv_color_index) & (bv_color_index <= upper_bound))
				elif isinstance(bv_color_index, (int, float)):
					boolean_indices = (
						(lower_bound <= bv_color_index) and (bv_color_index <= upper_bound))					
				else:
					raise ValueError("invalid type(bv_color_index): {}".format(type(bv_color_index)))
			else:
				raise ValueError("invalid interval: {}".format(interval))
			return boolean_indices

		## see flowers 1996 ApJ
		## see Allan Astrophysical Quantities
		## see cox 2000
		## interpolate using values from table
		temperatures = np.full(
			fill_value=np.nan,
			shape=bv_color_index.shape,
			dtype=float)
		for interval in ("ballasteros", ):
			boolean_indices = get_boolean_indices_at_interval_of_bv_color_index(
				bv_color_index=np.array(
					bv_color_index),
				interval=interval)
			if np.any(boolean_indices):
				temperatures_at_interval = get_temperatures(
					bv_color_index=bv_color_index[boolean_indices],
					interval=interval)
				temperatures[boolean_indices] = temperatures_at_interval
		temperatures = np.array(
			temperatures)
		return temperatures

	@staticmethod
	def get_peak_wavelengths(temperatures):
		wiens_constant = 2.897771955e-3 ## m K
		conversion_factor_at_m_to_nm = 1e9
		peak_wavelengths = conversion_factor_at_m_to_nm * wiens_constant / temperatures
		return peak_wavelengths

	def get_classification_args_by_spectral_type(self, full_spectral_classification):
		args = None
		for parameter_specifier in ("universal", "particular", "generic"):
			if args is not None:
				break
			for candidate in self.classification_parameterization["spectral type"][parameter_specifier]:
				if candidate[0] in full_spectral_classification:
					args = tuple(
						candidate)
					break
		return args

	def get_classification_args_by_spectral_sub_type(self, full_spectral_classification):
		args = None
		for candidate in self.classification_parameterization["spectral type"]["sub-type"]:
			if candidate in full_spectral_classification:
				args = tuple(
					candidate)
				break
		return args

	def get_classification_args_by_luminosity_class(self, full_spectral_classification):
		args = None
		for parameter_specifier in ("universal", "particular", "generic"):
			if args is not None:
				break
			for candidate in self.classification_parameterization["luminosity class"][parameter_specifier]:
				if candidate[0] in full_spectral_classification:
					args = tuple(
						candidate)
					break
		return args

	def get_spectral_types_and_luminosity_classes(self, full_spectral_classifications):
		spectral_types, spectral_sub_types, luminosity_classes = list(), list(), list()
		for full_spectral_classification in full_spectral_classifications:
			spectral_type_args = self.get_classification_args_by_spectral_type(
				full_spectral_classification=str(
					full_spectral_classification))
			spectral_sub_type_args = self.get_classification_args_by_spectral_sub_type(
				full_spectral_classification=str(
					full_spectral_classification))
			luminosity_class_args = self.get_classification_args_by_luminosity_class(
				full_spectral_classification=str(
					full_spectral_classification))
			if spectral_type_args is None:
				spectral_types.append(
					"None")
			else:
				spectral_types.append(
					spectral_type_args[0])
			if spectral_sub_type_args is None:
				spectral_sub_types.append(
					"None")
			else:
				spectral_sub_types.append(
					spectral_sub_type_args[0])
			if luminosity_class_args is None:
				luminosity_classes.append(
					"None")
			else:
				luminosity_classes.append(
					luminosity_class_args[0])
		spectral_types = np.array(
			spectral_types,
			dtype="<U5") # dtype=StringDType
		spectral_sub_types = np.array(
			spectral_sub_types,
			dtype="<U5") # dtype=StringDType
		luminosity_classes = np.array(
			luminosity_classes,
			dtype="<U5") # dtype=StringDType
		return spectral_types, spectral_sub_types, luminosity_classes

class DataProcessingConfiguration(BaseDataProcessingConfiguration):

	def __init__(self):
		super().__init__()

	def initialize_sources(self, path_to_hyg_data_file, path_to_classification_file, path_to_data_directory):
		source_at_hyg_data = DataConfigurationStellarCatalogsHYG()
		source_at_hyg_data.initialize(
			path_to_file=path_to_hyg_data_file,
			path_to_directory=path_to_data_directory)
		source_at_parameters_table_data = DataConfigurationStellarParametersTable()
		source_at_parameters_table_data.initialize(
			path_to_file=path_to_classification_file,
			path_to_directory=path_to_data_directory)
		sources = {
			"data" : source_at_hyg_data,
			"stellar parameters" : source_at_parameters_table_data}
		self._sources = sources

	def initialize_sun(self):
		sun = {
			"Mass" : 1.989e30, ## kg
			"Temperature" : 5778, ## K
			"Radius" : 6.957e8, ## m
			"Raw Luminosity" : 3.828e26, ## W
			"Luminosity" : 1, ## solar luminosity
			"Distance" : 1 / 206_265, ## pc
			"Visual Magnitude" : -26.74,
			"Absolute Magnitude" : 4.83,
			"Color Index (B-V)" : 0.653,
			# "E(B-V)" : 0.005,
			"V-I" : 0.702}
		self._sun = sun

	def initialize_classification_parameterization(self):
		self.pre_initialize_classification_parameterization()
		self.initialize_classification_parameterization_by_spectral_type()
		self.initialize_classification_parameterization_by_luminosity_class()

	def initialize_data(self):
		source_at_hyg_data = self.sources["data"]
		data = {
			column_header : source_at_hyg_data.df[column_header].to_numpy()
				for column_header in source_at_hyg_data.df.columns}
		temperatures = self.get_temperatures(
			bv_color_index=source_at_hyg_data.df["Color Index (B-V)"])
		peak_wavelengths = self.get_peak_wavelengths(
			temperatures=temperatures)
		spectral_types, spectral_sub_types, luminosity_classes = self.get_spectral_types_and_luminosity_classes(
			full_spectral_classifications=source_at_hyg_data.df["Full Spectral Class"])
		data["temperature"] = temperatures
		data["spectral type"] = spectral_types
		data["spectral sub-type"] = spectral_sub_types
		data["luminosity class"] = luminosity_classes
		self._data = data

##