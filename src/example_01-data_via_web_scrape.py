from stellar_object_parameterization_configuration import StellarObjectParameterizationConfiguration


path_to_hyg_data_file = None
path_to_classification_file = None
path_to_data_directory = None

# is_save, path_to_save_directory = False, None
is_save, path_to_save_directory = True, "/Users/owner/Desktop/programming/hr_diagram/output/"


if __name__ == "__main__":

	## initialize
	stellar_object_parameterization = StellarObjectParameterizationConfiguration()
	stellar_object_parameterization.initialize(
		path_to_hyg_data_file=path_to_hyg_data_file,
		path_to_classification_file=path_to_classification_file,
		path_to_data_directory=path_to_data_directory)
	stellar_object_parameterization.update_save_directory(
		path_to_save_directory=path_to_save_directory)

	## view
	stellar_object_parameterization.view_hr_diagram(
		x_parameter="Color Index (B-V)",
		y_parameter="Absolute Magnitude",
		color_by=None,
		is_show_sun=True,
		# is_show_isoradius=True,
		is_show_unclassified=True,
		is_show_spectral_types=True,
		luminosity_class_at_spectral_type="V",
		search_parameters="Color Index (B-V)",
		search_conditions=(
			"greater than or equal",
			"less than or equal"),
		search_values=(
			-0.5,
			2.5),
		is_any=False,
		figsize=(12, 8),
		is_save=is_save)
	stellar_object_parameterization.view_hr_diagram(
		x_parameter="Color Index (B-V)",
		y_parameter="Absolute Magnitude",
		color_by="luminosity class",
		is_show_sun=True,
		is_show_isoradius=True,
		is_show_unclassified=False,
		is_show_spectral_types=True,
		luminosity_class_at_spectral_type="V",
		search_parameters="Color Index (B-V)",
		search_conditions=(
			"greater than or equal",
			"less than or equal"),
		search_values=(
			-0.5,
			2.5),
		is_any=False,
		figsize=(12, 8),
		is_save=is_save)
	stellar_object_parameterization.view_hr_diagram(
		x_parameter="Color Index (B-V)",
		y_parameter="Absolute Magnitude",
		color_by="spectral type",
		is_show_sun=True,
		is_show_isoradius=True,
		is_show_unclassified=False,
		is_show_spectral_types=True,
		luminosity_class_at_spectral_type="V",
		search_parameters="Color Index (B-V)",
		search_conditions=(
			"greater than or equal",
			"less than or equal"),
		search_values=(
			-0.5,
			2.5),
		is_any=False,
		figsize=(12, 8),
		is_save=is_save)
	stellar_object_parameterization.view_hr_diagram(
		x_parameter="Color Index (B-V)",
		y_parameter="Luminosity",
		color_by=None,
		is_show_sun=True,
		# is_show_isoradius=True,
		is_show_unclassified=True,
		is_show_spectral_types=True,
		luminosity_class_at_spectral_type="V",
		search_parameters="Color Index (B-V)",
		search_conditions=(
			"greater than or equal",
			"less than or equal"),
		search_values=(
			-0.5,
			2.5),
		is_any=False,
		figsize=(12, 8),
		is_save=is_save)
	stellar_object_parameterization.view_hr_diagram(
		x_parameter="Color Index (B-V)",
		y_parameter="Luminosity",
		color_by="luminosity class",
		is_show_sun=True,
		is_show_isoradius=True,
		is_show_unclassified=False,
		is_show_spectral_types=True,
		luminosity_class_at_spectral_type="V",
		search_parameters="Color Index (B-V)",
		search_conditions=(
			"greater than or equal",
			"less than or equal"),
		search_values=(
			-0.5,
			2.5),
		is_any=False,
		figsize=(12, 8),
		is_save=is_save)
	stellar_object_parameterization.view_hr_diagram(
		x_parameter="Color Index (B-V)",
		y_parameter="Luminosity",
		color_by="spectral type",
		is_show_sun=True,
		is_show_isoradius=True,
		is_show_unclassified=False,
		is_show_spectral_types=True,
		luminosity_class_at_spectral_type="V",
		search_parameters="Color Index (B-V)",
		search_conditions=(
			"greater than or equal",
			"less than or equal"),
		search_values=(
			-0.5,
			2.5),
		is_any=False,
		figsize=(12, 8),
		is_save=is_save)
	stellar_object_parameterization.view_two_dimensional_histogram(
		x_parameter="Color Index (B-V)",
		y_parameter="Absolute Magnitude",
		x_bin_selection_method="number bins",
		y_bin_selection_method="number bins",
		x_kwargs={
			"number_bins" : 31,
			"leftmost_edge" : -0.5,
			"rightmost_edge" : 2.5,
			# "round_to_base" : 0.5,
			},
		y_kwargs={
			"number_bins" : 31,
			"leftmost_edge" : 20,
			"rightmost_edge" : -20,
			"round_to_base" : 5},
		is_color_scale_log=True,
		is_show_sun=True,
		is_show_isoradius=True,
		is_show_spectral_types=True,
		luminosity_class_at_spectral_type="V",					
		search_parameters="Color Index (B-V)",
		search_conditions=(
			"greater than or equal",
			"less than or equal"),
		search_values=(
			-0.5,
			2.5),
		is_any=False,
		figsize=(12, 8),
		is_save=is_save)

##