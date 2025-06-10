from source_base_configuration import SourceConfiguration
import numpy as np
import pandas as pd


class DataConfigurationStellarCatalogsHYG(SourceConfiguration):

	def __init__(self):
		super().__init__(
			name="hygdata_v41",
			title="HYG DataBase (version 4.1)",
			author="David Nash",
			general_url="https://www.astronexus.com/projects/hyg",
			personal_url="https://codeberg.org/astronexus",
			data_url="https://www.astronexus.com/downloads/catalogs/hygdata_v41.csv.gz",
			license="CC BY-SA-4.0 license",
			original_source="AstroNexus")
		self._column_mapping = None

	@property
	def column_mapping(self):
		return self._column_mapping

	def initialize_column_mapping(self):
		column_mapping = {
			"original data" : (
				(0, "id", "ID", None),
				(1, "hip", "Hipparcos ID", None),
				(2, "hd", "Henry Draper ID", None),
				(3, "hr", "Harvard Revised ID", None),
				(4, "gl", "Gliese ID", None),
				(5, "bf", "Bayer-Flamsteed ID", None),
				(6, "proper", "Proper IAU Name", None),
				(7, "ra", "RA (hour)", r"$hour$"),
				(8, "dec", "Dec (degree)", r"$\degree$"),
				(9, "dist", "Distance", r"$pc$"),
				(10, "pmra", "RA (proper motion, arcsec)", r"$\frac{\arcsec}{year}$"),
				(11, "pmdec", "Dec (proper motion, arcsec)", r"$\frac{\arcsec}{year}$"),
				(12, "rv", "Radial Velocity", r"$\frac{km}{s}$"),
				(13, "mag", "Visual Magnitude", None),
				(14, "absmag", "Absolute Magnitude", None),
				(15, "spect", "Full Spectral Class", None),
				(16, "ci", "Color Index (B-V)", None),
				(17, "x", "x", r"$pc$"),
				(18, "y", "y", r"$pc$"),
				(19, "z", "z", r"$pc$"),
				(20, "vx", "vx", r"$\frac{pc}{year}$"),
				(21, "vy", "vy", r"$\frac{pc}{year}$"),
				(22, "vz", "vz", r"$\frac{pc}{year}$"),
				(23, "rarad", "RA (radians)", r"$rad$"),
				(24, "decrad", "Dec (radians)", r"$rad$"),
				(25, "pmrarad", "RA (proper motion, radians)", r"$\frac{rad}{year}$"),
				(26, "pmdecrad", "Dec (proper motion, radians)", r"$\frac{rad}{year}$"),
				(27, "bayer", "Bayer ID", None),
				(28, "flam", "Flamsteed ID", None),
				(29, "con", "Constellation", None),
				(30, "comp", "Companion Star ID", None),
				(31, "comp_primary", "Primary Star ID", None),
				(32, "base", "Catalog ID", None),
				(33, "lum", "Luminosity", r"$\frac{L}{L_{{⊙}}}$"),
				(34, "var", "Variable Star Designation", None),
				(35, "var_min", "min(Magnitude)", None),
				(36, "var_max", "max(Magnitude)", None)),
			"calculated data" : (
				(37, "Spectral Type", None),
				(38, "Spectral Sub-Type", None),
				(39, "Luminosity Class", None),
				(40, "Temperature", r"$K$"),
				(41, "Peak Wavelength", r"$\AA$"),
				)
			}
		self._column_mapping = column_mapping

	def get_unit_label(self, parameter):
		is_found = False
		for args in self.column_mapping["original data"]:
			(index_at_column, original_header, new_header, unit_label) = args
			if new_header == parameter:
				is_found = True
				break
		if not is_found:
			for other_args in self.column_mapping["calculated data"]:
				(index_at_column, new_header, unit_label) = other_args
				if new_header == parameter:
					is_found = True
					break
		if not is_found:
			raise ValueError("invalid parameter: {}".format(parameter))
		return unit_label

	def initialize_df(self, path_to_file):
		self.initialize_column_mapping()
		kwargs = dict()
		if path_to_file is None:
			is_web_scraped = True
			target_path = self.data_url[:]
			kwargs["compression"] = "gzip"
		elif isinstance(path_to_file, str):
			is_web_scraped = False
			target_path = path_to_file[:]
		else:
			raise ValueError("invalid type(path_to_file): {}".format(type(path_to_file)))
		df = pd.read_csv(
			target_path,
			**kwargs)
		# dec = np.array(
		# 	df["dec"])
		# if len(dec.shape) == 2:
		# 	dec_by_epoch = np.copy(
		# 		dec[:, 0])
		# 	dec_by_equinox_j2000 = np.copy(
		# 		dec[:, 1])
		# 	# df["dec"] = dec_by_epoch
		# 	df["dec"] = dec_by_equinox_j2000
		# 	# df.drop(
		# 	# 	"dec",
		# 	# 	axis=1,
		# 	# 	inplace=True)
		name_mapping = dict()
		for args in self.column_mapping["original data"]:
			(index_at_column, original_header, new_header, unit_label) = args
			name_mapping[original_header] = new_header
		df.rename(
			columns=name_mapping,
			inplace=True)
		df.dropna()
		self._is_web_scraped = is_web_scraped
		self._df = df

##