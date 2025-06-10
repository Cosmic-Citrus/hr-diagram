from search_configuration import DataSearcherConfiguration
from plotter_base_configuration import BasePlotterConfiguration
from histogram_configuration import (
	HistogramConfiguration,
	TwoDimensionalHistogramConfiguration)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as path_effects


class BasestHRDiagramViewer(BasePlotterConfiguration):

	def __init__(self):
		super().__init__()

	@staticmethod
	def verify_xy_parameters(x_parameter, y_parameter):
		x_parameters = (
			"Color Index (B-V)",
			"Temperature")
		y_parameters = (
			"Luminosity",
			"Absolute Magnitude")
		if x_parameter not in x_parameters:
			raise ValueError("invalid x_parameter: {}".format(x_parameter))
		if y_parameter not in y_parameters:
			raise ValueError("invalid y_parameter: {}".format(y_parameter))

	@staticmethod
	def get_save_name(plot_name, x_parameter, y_parameter, is_show_isoradius, is_show_sun, is_show_spectral_types, is_mark_spectral_sub_type_at_zero, is_save, is_show_unclassified=False, is_show_particular=False, luminosity_class_at_spectral_type=None):
		
		def get_condensed_string(s):
			s_prime = s.title().replace(
				"-",
				" ")
			s_prime = s_prime.replace(
				"_",
				" ")
			s_prime = s_prime.replace(
				" ",
				"")
			return s_prime

		if is_save:
			save_name = "HR_Diagram-{}-{}_VS_{}".format(
				get_condensed_string(
					plot_name),
				get_condensed_string(
					y_parameter),
				get_condensed_string(
					x_parameter))
			if is_show_sun:
				save_name += "-wSun"
			if is_show_isoradius:
				save_name += "-wIsoRadius"
			if is_show_spectral_types:
				save_name += "-wSpectralTypes_MarkAt{}".format(
					"0" if is_mark_spectral_sub_type_at_zero else "9")
				if luminosity_class_at_spectral_type is not None:
					save_name += "{}".format(
						luminosity_class_at_spectral_type)
			if is_show_unclassified:
				save_name += "-wUnclassified"
			if is_show_particular:
				save_name += "-wParticular"
		else:
			save_name = None
		return save_name

	def autoformat_plot(self, ax, cls, x_parameter, y_parameter, is_dim2_histogram):

		def get_title(cls):
			source_at_hyg_data = cls.sources["data"]
			title = "HR Diagram via {}".format(
				source_at_hyg_data.title)
			return title

		def get_axis_label(cls, parameter):
			source_at_hyg_data = cls.sources["data"]
			unit_label = source_at_hyg_data.get_unit_label(
				parameter=parameter)
			if unit_label is None:
				axis_label = parameter.title()
			else:
				axis_label = "{} [{}]".format(
					parameter.title(),
					unit_label)
			return axis_label

		def get_axis_limits(parameter):
			mapping = {
				"Color Index (B-V)" : (
					-0.45,
					2.3),
				"Absolute Magnitude" : (
					20,
					-20),
				"Luminosity" : (
					1e-7,
					7.5e9),
				"Temperature" : (
					10_500,
					3750)}
			axis_limits = mapping[parameter]
			return axis_limits

		if x_parameter == "Temperature":
			x_log_base = 10
		else:
			x_log_base = None
		if y_parameter == "Luminosity":
			y_log_base = 10
		else:
			y_log_base = None
		xlabel = get_axis_label(
			cls=cls,
			parameter=x_parameter)
		ylabel = get_axis_label(
			cls=cls,
			parameter=y_parameter)
		title = get_title(
			cls=cls)
		if is_dim2_histogram:
			title = "2-D Histogram of {}".format(
				title)
		xlim = get_axis_limits(
			parameter=x_parameter)
		ylim = get_axis_limits(
			parameter=y_parameter)
		ax = self.visual_settings.autoformat_axis_labels(
			ax=ax,
			xlabel=xlabel,
			ylabel=ylabel,
			title=title)
		ax = self.visual_settings.autoformat_axis_limits(
			ax=ax,
			xlim=xlim,
			ylim=ylim)
		ax = self.visual_settings.autoformat_axis_ticks_and_ticklabels(
			ax=ax,
			x_major_ticks=True,
			y_major_ticks=True,
			x_minor_ticks=True,
			y_minor_ticks=True,
			x_major_ticklabels=True,
			y_major_ticklabels=True,
			x_minor_ticklabels=False,
			y_minor_ticklabels=False,
			x_major_fmt=r"{:,.2f}",
			x_minor_fmt=None,
			y_major_fmt=r"{:,.2f}",
			y_minor_fmt=None,
			x_log_base=x_log_base,
			y_log_base=y_log_base)
		ax = self.visual_settings.autoformat_grid(
			ax=ax,
			grid_color="gray")
		return ax

	def plot_legend(self, fig, ax, cls, search_args):		
		handles, labels = ax.get_legend_handles_labels()
		if search_args is None:
			leg_title = None
			number_columns = len(
				labels)
		else:
			searcher = DataSearcherConfiguration()
			number_columns = 5
			source_at_hyg_data = cls.sources["data"]
			(search_parameters, search_conditions, search_values, is_any) = search_args
			partial_titles = list()
			for (search_parameter, search_condition, search_value) in zip(search_parameters, search_conditions, search_values):
				if search_condition not in searcher.mapping.keys():
					raise ValueError("invalid search_condition: {}".format(search_condition))
				operator_symbol = searcher.mapping[search_condition][1]
				unit_label = source_at_hyg_data.get_unit_label(
					parameter=search_parameter)
				partial_title = r"{} {} {}".format(
					search_parameter,
					operator_symbol,
					search_value)
				if unit_label is not None:
					partial_title += " {}".format(
						unit_label)
				partial_titles.append(
					partial_title)
			if is_any:
				prefix = "Any"
			else:
				prefix = "All"
			primary_title = ", ".join(
				partial_titles)
			leg_title = "{} {{{}}}".format(
				prefix,
				primary_title)
		leg = self.visual_settings.get_legend(
			fig=fig,
			ax=ax,
			handles=handles,
			labels=labels,
			title=leg_title,
			number_columns=number_columns)
		return fig, ax, leg

	def plot_color_bar(self, fig, ax, handle, true_cmap, is_color_scale_log, invalid_color, title=None, orientation="vertical", ticks_position="left", label_position="left"):
		cbar = self.visual_settings.get_color_bar(
			fig=fig,
			ax=ax,
			handle=handle,
			title=title,
			orientation=orientation,
			ticks_position=ticks_position,
			label_position=label_position,
			shrink=0.5,
			pad=0.1,
			extend="max")
		cbar.ax.set_ylabel(
			"Number of Stellar Objects",
			fontsize=self.visual_settings.label_size)
		if is_color_scale_log:
			scalar_cmap = ListedColormap([
				true_cmap.get_bad()])
			scalar_mappable = ScalarMappable(
				cmap=scalar_cmap)
			divider = make_axes_locatable(
				cbar.ax)
			scalar_ax = divider.append_axes(
				"bottom",
				size="5%",
				pad="3%",
				aspect=1,
				anchor=cbar.ax.get_anchor())
			scalar_ax.grid(
				visible=False,
				which="both",
				axis="both")
			invalid_cbar = Colorbar(
				ax=scalar_ax,
				mappable=scalar_mappable,
				orientation="vertical")
			scalar_ticks = [
				0.5]
			scalar_ticklabels = [
				"Zero Stellar Objects"]
			invalid_cbar.set_ticks(
				scalar_ticks,
				labels=scalar_ticklabels)
			invalid_cbar.ax.tick_params(
				length=0,
				labelsize=self.visual_settings.tick_size)
		else:
			invalid_cbar = None
		return fig, ax, cbar, invalid_cbar

class BaseHRDiagramViewer(BasestHRDiagramViewer):

	def __init__(self):
		super().__init__()

	@staticmethod
	def plot_two_dimensional_histogram(ax, histogram, cmap, is_color_scale_log, invalid_color):
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()
		if is_color_scale_log:
			norm = LogNorm(
				vmin=1,
				vmax=np.max(
					histogram.bin_counts))
		else:
			norm = Normalize(
				vmin=0,
				vmax=np.max(
					histogram.bin_counts))
		true_cmap = plt.get_cmap(
			cmap)
		true_cmap.set_bad(
			invalid_color)
		handle = ax.imshow(
			histogram.bin_counts.T,
			interpolation="nearest",
			origin="lower",
			extent=[
				*xlim,
				*ylim],
			cmap=true_cmap,
			norm=norm,
			aspect="auto")
		return ax, handle, true_cmap, norm

	@staticmethod
	def plot_sun(ax, cls, x_parameter, y_parameter, sun_color, sun_marker="*", alpha=1):
		x = cls.sun[x_parameter]
		y = cls.sun[y_parameter]
		label = r"Sun ☉"
		ax.scatter(
			[x],
			[y],
			label=label,
			color=sun_color,
			marker=sun_marker,
			s=5,
			alpha=alpha)
		return ax

	@staticmethod
	def plot_stellar_objects_by_undifferentiated_classifications(ax, data, x_parameter, y_parameter, undifferentiated_color, undifferentiated_marker=".", alpha=0.8):
		x = data[x_parameter]
		y = data[y_parameter]
		label = r"${:,}$ Stellar Objects".format(
			data[x_parameter].size)
		ax.scatter(
			x,
			y,
			label=label,
			color=undifferentiated_color,
			marker=undifferentiated_marker,
			s=2,
			alpha=alpha)
		return ax

	@staticmethod
	def plot_stellar_objects_by_differentiated_classifications(ax, cls, data, x_parameter, y_parameter, color_by, is_show_unclassified, is_show_particular, unclassified_color, classified_marker=".", unclassified_marker="x", alpha=0.8):
		if not isinstance(is_show_particular, bool):
			raise ValueError("invalid type(is_show_particular): {}".format(type(is_show_particular)))
		if color_by not in cls.classification_parameterization.keys():
			raise ValueError("invalid color_by: {}".format(color_by))
		if "sub-type" in color_by:
			raise ValueError("invalid color_by: {}".format(color_by))
		facecolors = tuple([
			"steelblue",
			"silver",
			"limegreen",
			"goldenrod",
			"darkorange",
			"crimson",
			"plum", # "mediumorchid",
			"tan",
			"brown",
			# "blue",
			])
		searcher = DataSearcherConfiguration()
		search_parameters = color_by[:]
		other_key = "particular" if is_show_particular else "generic"
		multiple_search_values = list()
		for classification_args in cls.classification_parameterization[color_by]["universal"]:
			multiple_search_values.append(
				classification_args[0])
		for classification_args in cls.classification_parameterization[color_by][other_key]:
			multiple_search_values.append(
				classification_args[0])
		number_matched_groups = 0
		for search_values in multiple_search_values:
			searched_data, is_matches, search_args = searcher.get_data(
				original_data=data,
				search_parameters=search_parameters,
				search_conditions="equal",
				search_values=search_values,
				is_any=True,
				is_suppress_error=True)
			if np.any(is_matches):
				x = searched_data[x_parameter]
				y = searched_data[y_parameter]
				if color_by == "luminosity class":
					integer_by_roman_numeral = cls.convert_roman_numeral_to_integer(
						roman_numeral=search_values)
					index_at_facecolor = integer_by_roman_numeral - 1
					facecolor = facecolors[index_at_facecolor]
					classification_args = cls.get_classification_args_by_luminosity_class(
						full_spectral_classification=search_values)
					label = r"${:,}$ Class {} {}".format(
						x.size,
						classification_args[0],
						classification_args[1])
				else: # elif color_by == "spectral type":
					index_at_facecolor = int(
						number_matched_groups)
					facecolor = facecolors[index_at_facecolor]
					classification_args = cls.get_classification_args_by_spectral_type(
						full_spectral_classification=search_values)
					prefix = r"${:,}$ Type-${}$".format(
						x.size,
						classification_args[0])
					suffix = r"{}".format(
						classification_args[1])
					label = "{}\n{}".format(
						prefix,
						suffix)
				ax.scatter(
					x,
					y,
					label=label,
					color=facecolor,
					marker=classified_marker,
					s=2,
					alpha=alpha)
				number_matched_groups += 1
		if is_show_unclassified:
			other_searched_data, is_other_matches, other_search_args = searcher.get_data(
				original_data=data,
				search_parameters=search_parameters,
				search_conditions="not equal",
				search_values=multiple_search_values,
				is_any=False,
				is_suppress_error=True)
			if np.any(is_other_matches):
				other_x = other_searched_data[x_parameter]
				other_y = other_searched_data[y_parameter]
				other_prefix = r"${:,}$ Other".format(
					other_x.size)
				other_suffix = r"Stellar Objects"
				other_label = "{}\n{}".format(
					other_prefix,
					other_suffix)
				ax.scatter(
					other_x,
					other_y,
					label=other_label,
					color=unclassified_color,
					marker=unclassified_marker,
					s=2,
					alpha=alpha)
		return ax

	def plot_text_by_spectral_types(self, ax, cls, x_parameter, spectral_type_color, luminosity_class_at_spectral_type, is_mark_spectral_sub_type_at_zero):
		
		def get_x(ax, cls, df_key_at_x, spectral_type, spectral_sub_type, luminosity_class_at_spectral_type):
			if luminosity_class_at_spectral_type is None:
				combined_spectral_type = "{}{}".format(
					spectral_type,
					spectral_sub_type)
			else:
				combined_spectral_type = "{}{}{}".format(
					spectral_type,
					spectral_sub_type,
					luminosity_class_at_spectral_type)
			source = cls.sources["stellar parameters"]
			x = None
			for index_at_value, value in enumerate(source.df["StellarType"]):
				if combined_spectral_type in str(value):
					x = float(
						source.df[df_key_at_x][index_at_value])
					break
			if x is None:
				raise ValueError("match not found for {}".format(combined_spectral_type))
			display_coordinates = ax.transData.transform((
				x,
				1)) ## 1, not 0 in-case log-scale; np.nan is no-go
			axis_coordinates = ax.transAxes.inverted().transform(
				display_coordinates)
			x_prime = axis_coordinates[0]
			return x_prime

		if not isinstance(is_mark_spectral_sub_type_at_zero, bool):
			raise ValueError("invalid type(is_mark_spectral_sub_type_at_zero): {}".format(type(is_mark_spectral_sub_type_at_zero)))
		if luminosity_class_at_spectral_type is None:
			luminosity_class_at_spectral_type = "Ia0" # "I"
		elif not isinstance(luminosity_class_at_spectral_type, str):
			raise ValueError("invalid type(luminosity_class_at_spectral_type): {}".format(type(luminosity_class_at_spectral_type)))
		mapping_at_x = {
			"Color Index (B-V)" : "Color IndexB-V",
			"Temperature" : "TempK"}
		if x_parameter not in mapping_at_x.keys():
			raise ValueError("invalid x_parameter: {}".format(x_parameter))
		df_key_at_x = mapping_at_x[x_parameter]
		y = 1.0
		spectral_sub_type_at_mark = "0" if is_mark_spectral_sub_type_at_zero else "9"
		for classification_args in cls.classification_parameterization["spectral type"]["universal"]:
			if classification_args[1] == "Main-Sequence Stars":
				spectral_type = classification_args[0]
				for spectral_sub_type in cls.classification_parameterization["spectral type"]["sub-type"]:
					if spectral_sub_type == spectral_sub_type_at_mark:
						fontsize = float(
							self.visual_settings.tick_size)
						if luminosity_class_at_spectral_type is None:
							label = "|\n{}{}".format(
								spectral_type,
								spectral_sub_type)
						else:
							label = "|\n{}{}\n{}".format(
								spectral_type,
								spectral_sub_type,
								luminosity_class_at_spectral_type)
					else:
						label = "|"
						fontsize = 0.5 * float(
							self.visual_settings.tick_size)
					x = get_x(
						ax=ax,
						cls=cls,
						df_key_at_x=df_key_at_x,
						spectral_type=spectral_type,
						spectral_sub_type=spectral_sub_type,
						luminosity_class_at_spectral_type=luminosity_class_at_spectral_type)
					ax.text(
						x,
						y,
						label,
						color=spectral_type_color,
						horizontalalignment="center",
						verticalalignment="top",
						transform=ax.transAxes,
						fontsize=fontsize)
		return ax

	def plot_isoradius(self, ax, cls, x_parameter, y_parameter, isoradius_color, isoradius_linestyle="--"):
		
		def get_radii(exponent_at_lower_bound=-2, exponent_at_upper_bound=4):
			lower_bound = float(
				"1e{}".format(
					exponent_at_lower_bound))
			upper_bound = float(
				"1e{}".format(
					exponent_at_upper_bound))
			number_radii = exponent_at_upper_bound - exponent_at_lower_bound + 1
			radii = np.geomspace(
				lower_bound,
				upper_bound,
				number_radii)
			radius_labels = list()
			for radius in radii:
				if int(radius) == float(radius):
					label = r"${:,}$ $R_☉$".format(
						int(
							radius))
				else:
					label = r"${:.2f}$ $R_☉$".format(
						radius)
				label += "\t"
				radius_labels.append(
					label)
			return radii, radius_labels

		def plot_lines_and_labels(ax, cls, x_parameter, y_parameter, radii, radius_labels, isoradius_color, isoradius_linestyle):
			for relative_radius, radius_label in zip(radii, radius_labels):
				if x_parameter == "Color Index (B-V)":
					bv_color_index = np.array([
						0.2,
						1.4])
					temperatures = cls.get_temperatures(
						bv_color_index=bv_color_index)
					x = bv_color_index
				elif x_parameter == "Temperature":
					temperatures = array([
						3e4,
						3e2])
					x = temperatures
				else:
					raise ValueError("invalid x_parameter: {}".format(x_parameter))
				# relative_solar_luminosity = np.square(radius / cls.sun["Radius"]) * np.square(np.square(temperatures / cls.sun["Temperature"]))
				relative_solar_luminosity = np.square(relative_radius) * np.square(np.square(temperatures / cls.sun["Temperature"]))
				absolute_magnitude = cls.sun["Absolute Magnitude"] - 2.5 * np.log10(relative_solar_luminosity)
				if y_parameter == "Luminosity":
					y = relative_solar_luminosity
				elif y_parameter == "Absolute Magnitude":
					y = absolute_magnitude
				else:
					raise ValueError("invalid y_parameter: {}".format(y_parameter))
				ax.plot(
					x,
					y,
					color=isoradius_color,
					linestyle=isoradius_linestyle)
				text_handle = ax.text(
					x[0],
					y[0],
					radius_label,
					color=isoradius_color,
					fontsize=self.visual_settings.label_size,
					verticalalignment="center",
					horizontalalignment="right")
					# bbox={
					# 	"facecolor": "yellow",
					# 	"edgecolor": "red",
					# 	"linewidth": 2,
					# 	# "pad": 4,
					# 	"boxstyle" : "round",
					# 	"pad" : 0.5},
				text_handle.set_path_effects([
					path_effects.Stroke(
						linewidth=1,
						foreground=isoradius_color)])
			return ax

		radii, radius_labels = get_radii()
		ax = plot_lines_and_labels(
			ax=ax,
			cls=cls,
			x_parameter=x_parameter,
			y_parameter=y_parameter,
			radii=radii,
			radius_labels=radius_labels,
			isoradius_color=isoradius_color,
			isoradius_linestyle=isoradius_linestyle)
		return ax

class HRDiagramViewer(BaseHRDiagramViewer):

	def __init__(self):
		super().__init__()

	def view_blank_hr_diagram(self, cls, x_parameter="Color Index (B-V)", y_parameter="Absolute Magnitude", is_show_isoradius=False, is_show_sun=False, is_show_spectral_types=False, is_mark_spectral_sub_type_at_zero=False, luminosity_class_at_spectral_type=None, isoradius_color="black", sun_color="gold", spectral_type_color="black", figsize=None, is_save=False):
		self.verify_xy_parameters(
			x_parameter=x_parameter,
			y_parameter=y_parameter)
		is_show_legend = False
		fig, ax = plt.subplots(
			figsize=figsize)
		ax = self.autoformat_plot(
			ax=ax,
			cls=cls,
			x_parameter=x_parameter,
			y_parameter=y_parameter,
			is_dim2_histogram=False)
		if is_show_isoradius:
			ax = self.plot_isoradius(
				ax=ax,
				cls=cls,
				x_parameter=x_parameter,
				y_parameter=y_parameter,
				isoradius_color=isoradius_color)
		if is_show_sun:
			is_show_legend = True
			ax = self.plot_sun(
				ax=ax,
				cls=cls,
				x_parameter=x_parameter,
				y_parameter=y_parameter,
				sun_color=sun_color)
		if is_show_spectral_types:
			ax = self.plot_text_by_spectral_types(
				ax=ax,
				cls=cls,
				x_parameter=x_parameter,
				spectral_type_color=spectral_type_color,
				luminosity_class_at_spectral_type=luminosity_class_at_spectral_type,
				is_mark_spectral_sub_type_at_zero=is_mark_spectral_sub_type_at_zero)
		# handles, labels = ax.get_legend_handles_labels()
		# if len(labels) > 0:
		if is_show_legend:
			fig, ax, leg = self.plot_legend(
				fig=fig,
				ax=ax,
				cls=cls,
				search_args=None)
		plot_name = "Blank"
		save_name = self.get_save_name(
			plot_name=plot_name,
			x_parameter=x_parameter,
			y_parameter=y_parameter,
			is_show_isoradius=is_show_isoradius,
			is_show_sun=is_show_sun,
			is_show_spectral_types=is_show_spectral_types,
			is_mark_spectral_sub_type_at_zero=is_mark_spectral_sub_type_at_zero,
			is_save=is_save,
			is_show_unclassified=False,
			is_show_particular=False,
			luminosity_class_at_spectral_type=luminosity_class_at_spectral_type)
		self.visual_settings.display_image(
			fig=fig,
			save_name=save_name,
			space_replacement="_")

	def view_hr_diagram(self, cls, search_parameters=None, search_conditions=None, search_values=None, is_any=False, x_parameter="Color Index (B-V)", y_parameter="Absolute Magnitude", color_by=None, is_show_isoradius=False, is_show_sun=False, is_show_unclassified=False, is_show_particular=False, is_show_spectral_types=False, is_mark_spectral_sub_type_at_zero=False, luminosity_class_at_spectral_type=None, isoradius_color="black", sun_color="gold", undifferentiated_color="steelblue", unclassified_color="black", spectral_type_color="black", figsize=None, is_save=False):
		if not isinstance(is_show_unclassified, bool):
			raise ValueError("invalid type(is_show_unclassified): {}".format(type(is_show_unclassified)))
		self.verify_xy_parameters(
			x_parameter=x_parameter,
			y_parameter=y_parameter)
		if search_parameters is None:
			data = dict(
				cls.data)
			search_args = None
		else:
			searcher = DataSearcherConfiguration()
			data, is_matches, search_args = searcher.get_data(
				original_data=cls.data,
				search_parameters=search_parameters,
				search_conditions=search_conditions,
				search_values=search_values,
				is_any=is_any)
		fig, ax = plt.subplots(
			figsize=figsize)
		ax = self.autoformat_plot(
			ax=ax,
			cls=cls,
			x_parameter=x_parameter,
			y_parameter=y_parameter,
			is_dim2_histogram=False)
		if is_show_isoradius:
			ax = self.plot_isoradius(
				ax=ax,
				cls=cls,
				x_parameter=x_parameter,
				y_parameter=y_parameter,
				isoradius_color=isoradius_color)
		if color_by is None:
			if not is_show_unclassified:
				raise ValueError("not yet implemented")
			ax = self.plot_stellar_objects_by_undifferentiated_classifications(
				ax=ax,
				data=data,
				x_parameter=x_parameter,
				y_parameter=y_parameter,
				undifferentiated_color=undifferentiated_color)
		else:
			ax = self.plot_stellar_objects_by_differentiated_classifications(
				ax=ax,
				cls=cls,
				data=data,
				x_parameter=x_parameter,
				y_parameter=y_parameter,
				color_by=color_by,
				unclassified_color=unclassified_color,
				is_show_unclassified=is_show_unclassified,
				is_show_particular=is_show_particular)
		if is_show_sun:
			ax = self.plot_sun(
				ax=ax,
				cls=cls,
				x_parameter=x_parameter,
				y_parameter=y_parameter,
				sun_color=sun_color)
		if is_show_spectral_types:
			ax = self.plot_text_by_spectral_types(
				ax=ax,
				cls=cls,
				x_parameter=x_parameter,
				spectral_type_color=spectral_type_color,
				luminosity_class_at_spectral_type=luminosity_class_at_spectral_type,
				is_mark_spectral_sub_type_at_zero=is_mark_spectral_sub_type_at_zero)
		fig, ax, leg = self.plot_legend(
			fig=fig,
			ax=ax,
			cls=cls,
			search_args=search_args)
		if color_by is None:
			plot_name = "Undifferentiated"
		else:
			plot_name = color_by.title()
		save_name = self.get_save_name(
			plot_name=plot_name,
			x_parameter=x_parameter,
			y_parameter=y_parameter,
			is_show_isoradius=is_show_isoradius,
			is_show_sun=is_show_sun,
			is_show_spectral_types=is_show_spectral_types,
			is_mark_spectral_sub_type_at_zero=is_mark_spectral_sub_type_at_zero,
			is_save=is_save,
			is_show_unclassified=is_show_unclassified,
			is_show_particular=is_show_particular,
			luminosity_class_at_spectral_type=luminosity_class_at_spectral_type)
		self.visual_settings.display_image(
			fig=fig,
			save_name=save_name,
			space_replacement="_")

	def view_two_dimensional_histogram(self, cls, x_bin_selection_method, y_bin_selection_method, *args, search_parameters=None, search_conditions=None, search_values=None, is_any=False, x_parameter="Color Index (B-V)", y_parameter="Absolute Magnitude", cmap="plasma", is_color_scale_log=False, invalid_color="black", is_show_isoradius=False, is_show_sun=False, is_show_spectral_types=False, is_mark_spectral_sub_type_at_zero=False, luminosity_class_at_spectral_type=None, isoradius_color="white", sun_color="gold", spectral_type_color="white", figsize=None, is_save=False, **kwargs):
		self.verify_xy_parameters(
			x_parameter=x_parameter,
			y_parameter=y_parameter)
		is_show_legend = False
		if search_parameters is None:
			data = dict(
				cls.data)
			search_args = None
		else:
			searcher = DataSearcherConfiguration()
			data, is_matches, search_args = searcher.get_data(
				original_data=cls.data,
				search_parameters=search_parameters,
				search_conditions=search_conditions,
				search_values=search_values,
				is_any=is_any)
		histogram = TwoDimensionalHistogramConfiguration()
		histogram.initialize(
			data[x_parameter],
			data[y_parameter],
			x_bin_selection_method,
			y_bin_selection_method,
			*args,
			**kwargs)
		fig, ax = plt.subplots(
			figsize=figsize)
		ax = self.autoformat_plot(
			ax=ax,
			cls=cls,
			x_parameter=x_parameter,
			y_parameter=y_parameter,
			is_dim2_histogram=True)
		ax, handle, true_cmap, norm = self.plot_two_dimensional_histogram(
			ax=ax,
			histogram=histogram,
			cmap=cmap,
			is_color_scale_log=is_color_scale_log,
			invalid_color=invalid_color)
		fig, ax, cbar, invalid_cbar = self.plot_color_bar(
			fig=fig,
			ax=ax,
			handle=handle,
			true_cmap=true_cmap,
			is_color_scale_log=is_color_scale_log,
			invalid_color=invalid_color,
			title=None)
		if is_show_isoradius:
			ax = self.plot_isoradius(
				ax=ax,
				cls=cls,
				x_parameter=x_parameter,
				y_parameter=y_parameter,
				isoradius_color=isoradius_color)
		if is_show_sun:
			is_show_legend = True
			ax = self.plot_sun(
				ax=ax,
				cls=cls,
				x_parameter=x_parameter,
				y_parameter=y_parameter,
				sun_color=sun_color)
		if is_show_spectral_types:
			ax = self.plot_text_by_spectral_types(
				ax=ax,
				cls=cls,
				x_parameter=x_parameter,
				spectral_type_color=spectral_type_color,
				luminosity_class_at_spectral_type=luminosity_class_at_spectral_type,
				is_mark_spectral_sub_type_at_zero=is_mark_spectral_sub_type_at_zero)
		if (not is_show_legend) and (search_args is not None):
			ax.scatter(
				list(),
				list(),
				color="none",
				label=" ",
				alpha=0)
			is_show_legend = True
		if is_show_legend:
			fig, ax, leg = self.plot_legend(
				fig=fig,
				ax=ax,
				cls=cls,
				search_args=search_args)
		plot_name = "2-D Histogram"
		save_name = self.get_save_name(
			plot_name=plot_name,
			x_parameter=x_parameter,
			y_parameter=y_parameter,
			is_show_isoradius=is_show_isoradius,
			is_show_sun=is_show_sun,
			is_show_spectral_types=is_show_spectral_types,
			is_mark_spectral_sub_type_at_zero=is_mark_spectral_sub_type_at_zero,
			is_save=is_save,
			luminosity_class_at_spectral_type=luminosity_class_at_spectral_type)
		self.visual_settings.display_image(
			fig=fig,
			save_name=save_name,
			space_replacement="_")

##