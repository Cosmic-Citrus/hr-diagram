import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm


class BaseVisualSettingsConfiguration():

	def __init__(self):
		super().__init__()
		self._path_to_save_directory = None
		self._tick_size = None
		self._label_size = None
		self._text_size = None
		self._cell_size = None
		self._title_size = None

	@property
	def path_to_save_directory(self):
		return self._path_to_save_directory
	
	@property
	def tick_size(self):
		return self._tick_size
	
	@property
	def label_size(self):
		return self._label_size
	
	@property
	def text_size(self):
		return self._text_size
	
	@property
	def cell_size(self):
		return self._cell_size
	
	@property
	def title_size(self):
		return self._title_size

	@staticmethod
	def autocorrect_string_spaces(s, space_replacement=None):
		if not isinstance(s, str):
			raise ValueError("invalid type(s): {}".format(type(s)))
		modified_s = s[:]
		if space_replacement is not None:
			if not isinstance(space_replacement, str):
				raise ValueError("invalid type(space_replacement): {}".format(type(space_replacement)))
			modified_s = s.replace(
				" ",
				space_replacement)
		return modified_s

	@staticmethod
	def verify_container_is_flat_numerical_array(container):
		if not isinstance(container, np.ndarray):
			raise ValueError("invalid type(container): {}".format(type(container)))
		if not (np.issubdtype(container.dtype, np.integer) or np.issubdtype(container.dtype, np.floating)):
			raise ValueError("invalid container.dtype: {}".format(container.dtype))
		size_at_shape = len(
			container.shape)
		if size_at_shape != 1:
			raise ValueError("invalid container.shape: {}".format(container.shape))

	@staticmethod
	def verify_element_is_numerical(element):
		if not isinstance(element, (int, float)):
			raise ValueError("invalid type(element): {}".format(type(element)))

	def verify_element_is_strictly_positive(self, element):
		self.verify_element_is_numerical(
			element=element)
		if element <= 0:
			raise ValueError("invalid element: {}".format(element))

	def verify_element_is_non_negative(self, element):
		self.verify_element_is_numerical(
			element=element)
		if element < 0:
			raise ValueError("invalid element: {}".format(element))

class BaseVisualColorsConfiguration(BaseVisualSettingsConfiguration):

	def __init__(self):
		super().__init__()

	@staticmethod
	def get_rgb_facecolors_from_cmap(cmap, norm, vector):
		f_cmap = plt.cm.get_cmap(
			cmap)
		rgb_facecolors = f_cmap(
			norm(
				vector))
		return rgb_facecolors

class VisualColorsConfiguration(BaseVisualColorsConfiguration):

	def __init__(self):
		super().__init__()

	def get_rgb_facecolors(self, number_colors, facecolors=None, cmap=None):
		if (cmap is None) and (facecolors is None):
			raise ValueError("cmap=None and facecolors=None is not a valid combination")
		if (cmap is not None) and (facecolors is not None):
			raise ValueError("input cmap or facecolors (not both)")
		if number_colors == 0:
			vector = np.array([
				0,
				1])
		else:
			vector = np.arange(
				number_colors,
				dtype=int)
		if facecolors is not None:
			if not isinstance(facecolors, (tuple, list)):
				raise ValueError("invalid type(facecolors): {}".format(type(facecolors)))
			number_facecolors = len(
				facecolors)
			if number_colors != number_facecolors:
				raise ValueError("len(facecolors)={} is not compatible with number_colors={}".format(number_facecolors, number_colors))
			cmap = ListedColormap(
				facecolors)
			color_edges = np.arange(
				-0.5,
				number_colors,
				1)
			norm = BoundaryNorm(
				color_edges,
				number_colors)
		else:
			norm = Normalize(
				vmin=0,
				vmax=vector[-1])
		rgb_facecolors = self.get_rgb_facecolors_from_cmap(
			cmap=cmap,
			norm=norm,
			vector=vector)
		return rgb_facecolors, cmap, norm

	def get_color_bar(self, fig, ax, handle, title=None, orientation="vertical", ticks_position="left", label_position="left", **kwargs):
		cbar = fig.colorbar(
			handle,
			ax=ax,
			orientation=orientation,
			**kwargs)
		cbar.ax.yaxis.set_ticks_position(
			ticks_position)
		cbar.ax.yaxis.set_label_position(
			label_position)
		cbar.ax.tick_params(
			labelsize=self.tick_size)
		if title is not None:
			cbar.ax.set_title(
				title,
				fontsize=self.label_size)
		return cbar

class BaseVisualAxesConfiguration(VisualColorsConfiguration):

	def __init__(self):
		super().__init__()

	@staticmethod
	def autoformat_ticks_and_ticklabels_at_dimension(ax, axis_name, data_mapping, parameterization_mapping):

		def get_autoformatted_ticklabels(ticklabels, fmt):
			if fmt is None:
				modified_ticklabels = ticklabels
			else:
				if isinstance(fmt, str):
					modified_ticklabels = [
						fmt.format(ticklabel)
							for ticklabel in ticklabels]
				elif callable(fmt):
					modified_ticklabels = [
						fmt(ticklabel)
							for ticklabel in ticklabels]
				else:
					raise ValueError("invalid type(fmt): {}".format(type(fmt)))
			return modified_ticklabels

		def set_default_ticklabels(ticks, ticklabels, fmt, get_ticks, set_ticklabels, is_major, **kwargs):
			if ticklabels is not None:
				if isinstance(ticklabels, bool):
					if ticklabels:
						if fmt is None:
							set_ticklabels(
								ticks,
								minor=np.invert(
									is_major),
								**kwargs)
						else:
							ticklabels = get_autoformatted_ticklabels(
								ticklabels=ticks,
								fmt=fmt)
							set_ticklabels(
								ticklabels,
								minor=np.invert(
									is_major),
								**kwargs)							
					else:
						set_ticklabels(
							list(),
							minor=np.invert(
								is_major),
							**kwargs)
				elif isinstance(ticklabels, (tuple, list, np.ndarray)):
					set_ticklabels(
						ticklabels,
						minor=np.invert(
							is_major),
						**kwargs)							
				else:
					raise ValueError("invalid type(ticklabels): {}".format(type(ticklabels)))
			return ax

		def set_default_ticks_and_ticklabels(ax, ticks, ticklabels, fmt, set_locator, get_ticks, set_ticks, set_ticklabels, is_major, **kwargs):
			if not isinstance(is_major, bool):
				raise ValueError("invalid type(is_major): {}".format(type(is_major)))
			if ticks is not None:
				if isinstance(ticks, bool):
					if ticks:
						if is_major:
							set_locator(
								ticker.AutoLocator())
						else:
							set_locator(
								ticker.AutoMinorLocator())							
						ticks = get_ticks(
							minor=np.invert(
								is_major))
						set_ticks(
							ticks,
							minor=np.invert(
								is_major))
						set_default_ticklabels(
							ticks=ticks,
							ticklabels=ticklabels,
							fmt=fmt,
							get_ticks=get_ticks,
							set_ticklabels=set_ticklabels,
							is_major=is_major,
							**kwargs)
					else:
						set_ticks(
							list(),
							minor=np.invert(
								is_major))
						if ticklabels is not None:
							if isinstance(ticklabels, bool):
								if ticklabels:
									raise ValueError("ticks=False and ticklabels=True is not a valid combination")
							elif isinstance(ticklabels, (tuple, list, np.ndarray)):
								raise ValueError("ticks=False and ticklabels=type({}) is not a valid combination".format(type(ticklabels)))
				elif isinstance(ticks, (tuple, list, np.ndarray)):
					self.verify_container_is_flat_numerical_array(
						container=np.array(
							ticks))
					set_ticks(
						ticks,
						minor=np.invert(
							is_major))
					set_default_ticklabels(
						ticks=ticks,
						ticklabels=ticklabels,
						fmt=fmt,
						get_ticks=get_ticks,
						set_ticklabels=set_ticklabels,
						is_major=is_major,
						**kwargs)
				else:
					raise ValueError("invalid type(ticks): {}".format(type(ticks)))
			return ax

		# def set_logarithmic_ticks_and_ticklabels(ax, log_base, set_locator):
		# 	set_locator(
		# 		ticker.LogLocator(
		# 			base=float(
		# 				log_base)))
		# 	return ax

		major_ticks = data_mapping[axis_name]["major ticks"]
		minor_ticks = data_mapping[axis_name]["minor ticks"]
		major_ticklabels = data_mapping[axis_name]["major ticklabels"]
		minor_ticklabels = data_mapping[axis_name]["minor ticklabels"]
		major_fmt = data_mapping[axis_name]["major fmt"]
		minor_fmt = data_mapping[axis_name]["minor fmt"]
		log_base = data_mapping[axis_name]["log base"]
		get_default_ticks = parameterization_mapping[axis_name]["tick getter"]
		get_default_ticklabels = parameterization_mapping[axis_name]["ticklabel getter"]
		set_ticks = parameterization_mapping[axis_name]["tick setter"]
		set_ticklabels = parameterization_mapping[axis_name]["ticklabel setter"]
		set_major_locator = parameterization_mapping[axis_name]["major locator setter"]
		set_minor_locator = parameterization_mapping[axis_name]["minor locator setter"]
		set_scale = parameterization_mapping[axis_name]["scale setter"]
		if log_base is None:
			ax = set_default_ticks_and_ticklabels(
				ax=ax,
				ticks=major_ticks,
				ticklabels=major_ticklabels,
				fmt=major_fmt,
				set_locator=set_major_locator,
				get_ticks=get_default_ticks,
				set_ticks=set_ticks,
				set_ticklabels=set_ticklabels,
				is_major=True)
			ax = set_default_ticks_and_ticklabels(
				ax=ax,
				ticks=minor_ticks,
				ticklabels=minor_ticklabels,
				fmt=minor_fmt,
				set_locator=set_minor_locator,
				get_ticks=get_default_ticks,
				set_ticks=set_ticks,
				set_ticklabels=set_ticklabels,
				is_major=False)
		else:
			set_scale(
				"log",
				base=log_base)
			set_major_locator(
				ticker.LogLocator(
					base=float(
						log_base)))
			# ax = set_logarithmic_ticks_and_ticklabels(
			# 	ax=ax,
			# 	log_base=log_base,
			# 	set_locator=set_major_locator)
			# ax = set_logarithmic_ticks_and_ticklabels(
			# 	ax=ax,
			# 	log_base=log_base,
			# 	set_locator=set_minor_locator)
		return ax

class VisualAxesConfiguration(BaseVisualAxesConfiguration):

	def __init__(self):
		super().__init__()

	@staticmethod
	def get_mirror_ax(ax, frameon=False):
		mirror_ax = ax.figure.add_subplot(
			ax.get_subplotspec(),
			frameon=frameon)
		mirror_ax.xaxis.set_label_position(
			"top")
		mirror_ax.xaxis.tick_top()
		mirror_ax.yaxis.set_label_position(
			"right")
		mirror_ax.yaxis.tick_right()
		return mirror_ax

	@staticmethod
	def autoformat_axis_limits(ax, xlim=None, ylim=None, zlim=None):
		if xlim is not None:
			ax.set_xlim(
				xlim)
		if ylim is not None:
			ax.set_ylim(
				ylim)
		if zlim is not None:
			if not hasattr(ax, "get_zlim"):
				raise ValueError("cannot apply zlim to non-3D ax")
			ax.set_zlim(
				zlim)
		return ax

	@staticmethod
	def autoformat_grid(ax, grid_color=None, grid_alpha=0.3, grid_linestyle=":", **kwargs):
		if grid_color is not None:
			ax.grid(
				color=grid_color,
				alpha=grid_alpha,
				linestyle=grid_linestyle,
				**kwargs)
		return ax

	def autoformat_axis_labels(self, ax, xlabel=None, ylabel=None, zlabel=None, title=None):
		if xlabel is not None:
			ax.set_xlabel(
				xlabel,
				fontsize=self.label_size)
		if ylabel is not None:
			ax.set_ylabel(
				ylabel,
				fontsize=self.label_size)
		if zlabel is not None:
			if not hasattr(ax, "get_zlim"):
				raise ValueError("cannot apply zlabel to non-3D ax")
			ax.set_zlabel(
				zlabel,
				fontsize=self.label_size)
		if title is not None:
			ax.set_title(
				title,
				fontsize=self.title_size)
		return ax

	def autoformat_axis_ticks_and_ticklabels(self, ax, x_major_ticks=None, y_major_ticks=None, z_major_ticks=None, x_minor_ticks=None, y_minor_ticks=None, z_minor_ticks=None, x_major_ticklabels=None, y_major_ticklabels=None, z_major_ticklabels=None, x_minor_ticklabels=None, y_minor_ticklabels=None, z_minor_ticklabels=None, x_major_fmt=None, x_minor_fmt=None, y_major_fmt=None, y_minor_fmt=None, z_major_fmt=None, z_minor_fmt=None, x_log_base=None, y_log_base=None, z_log_base=None):
		axis_names = [
			"x",
			"y"]
		data_mapping = {
			"x" : {
				"major ticks" : x_major_ticks,
				"minor ticks" : x_minor_ticks,
				"major ticklabels" : x_major_ticklabels,
				"minor ticklabels" : x_minor_ticklabels,
				"major fmt" : x_major_fmt,
				"minor fmt" : x_minor_fmt,
				"log base" : x_log_base},
			"y" : {
				"major ticks" : y_major_ticks,
				"minor ticks" : y_minor_ticks,
				"major ticklabels" : y_major_ticklabels,
				"minor ticklabels" : y_minor_ticklabels,
				"major fmt" : y_major_fmt,
				"minor fmt" : y_minor_fmt,
				"log base" : y_log_base}}
		parameterization_mapping = {
			"x" : {
				"tick getter" : ax.get_xticks,
				"tick setter" : ax.set_xticks,
				"ticklabel getter" : ax.get_xticklabels,
				"ticklabel setter" : ax.set_xticklabels,
				"major locator setter" : ax.xaxis.set_major_locator,
				"minor locator setter" : ax.xaxis.set_minor_locator,
				"scale setter" : ax.set_xscale},
			"y" : {
				"tick getter" : ax.get_yticks,
				"tick setter" : ax.set_yticks,
				"ticklabel getter" : ax.get_yticklabels,
				"ticklabel setter" : ax.set_yticklabels,
				"major locator setter" : ax.yaxis.set_major_locator,
				"minor locator setter" : ax.yaxis.set_minor_locator,
				"scale setter" : ax.set_yscale}}
		if hasattr(ax, "get_zlim"):
			axis_names.append(
				"z")
			data_mapping["z"] = {
				"major ticks" : z_major_ticks,
				"minor ticks" : z_minor_ticks,
				"major ticklabels" : z_major_ticklabels,
				"minor ticklabels" : z_minor_ticklabels,
				"major fmt" : z_major_fmt,
				"minor fmt" : z_minor_fmt,
				"log base" : z_log_base}
			parameterization_mapping["z"] = {
				"tick getter" : ax.get_zticks,
				"tick setter" : ax.set_zticks,
				"ticklabel getter" : ax.get_zticklabels,
				"ticklabel setter" : ax.set_zticklabels,
				"major locator setter" : ax.zaxis.set_major_locator,
				"minor locator setter" : ax.zaxis.set_minor_locator,
				"scale setter" : ax.set_zscale}
		for axis_name in axis_names:
			ax = self.autoformat_ticks_and_ticklabels_at_dimension(
				ax=ax,
				axis_name=axis_name,
				data_mapping=data_mapping,
				parameterization_mapping=parameterization_mapping)
			ax.tick_params(
				axis=axis_name,
				labelsize=self.tick_size,
				which="both")
		return ax

class BaseVisualLegendConfiguration(VisualAxesConfiguration):

	def __init__(self):
		super().__init__()

	@staticmethod
	def get_empty_label():
		empty_label = " "
		return empty_label

	@staticmethod
	def get_empty_scatter_handle(ax):
		handle = ax.scatter(
			[np.nan], 
			[np.nan], 
			color="none", 
			alpha=0)
		return handle

	@staticmethod
	def get_combined_legend_handles(ax, rgb_facecolors, alphas, **kwargs):
		number_colors = len(
			rgb_facecolors)
		number_alphas = len(
			alphas)
		if number_colors != number_alphas:
			raise ValueError("{} rgb_facecolors and {} alphas are not compatible".format(number_colors, number_alphas))
		handles = list()
		for rgb_facecolor, alpha in zip(rgb_facecolors, alphas):
			handle = ax.scatter(
				list(),
				list(),
				c=rgb_facecolor,
				alpha=alpha,
				**kwargs)
			handles.append(
				handle)
		return handles
	
	def get_base_legend(self, fig, handles, labels, ax=None, number_columns=None, **kwargs):
		if not isinstance(handles, (tuple, list)):
			raise ValueError("invalid type(handles): {}".format(type(handles)))
		if not isinstance(labels, (tuple, list)):
			raise ValueError("invalid type(labels): {}".format(type(labels)))
		number_handles = len(
			handles)
		number_labels = len(
			labels)
		if number_handles != number_labels:
			raise ValueError("{} handles and {} labels are not compatible".format(number_handles, number_labels))
		if number_labels == 0:
			raise ValueError("zero labels found")
		is_add_empty_columns = False
		if number_labels == 1:
			if ax is None:
				raise ValueError("ax is required to get empty_handle")
			is_add_empty_columns = True
			empty_handle = self.get_empty_scatter_handle(
				ax=ax)
			empty_label = self.get_empty_label()
			modified_handles = [
				empty_handle,
				handles[0],
				empty_handle]
			modified_labels = [
				empty_label,
				labels[0],
				empty_label]
			modified_number_columns = len(
				modified_labels)
		else:
			if number_columns is None:
				modified_number_columns = int(
					number_labels)
			else:
				if not isinstance(number_columns, int):
					raise ValueError("invalid type(number_columns): {}".format(type(number_columns)))
				if number_columns <= 0:
					raise ValueError("invalid number_columns: {}".format(number_columns))
				modified_number_columns = int(
					number_columns)
			modified_handles = handles
			modified_labels = labels
		fig.subplots_adjust(
			bottom=0.2)
		leg = fig.legend(
			handles=modified_handles,
			labels=modified_labels,
			ncol=modified_number_columns,
			**kwargs)
		return leg, modified_handles, modified_labels, is_add_empty_columns

	def autoformat_legend(self, leg, labels, title=None, title_color="black", text_colors="black", facecolor="lightgray", edgecolor="gray", is_add_empty_columns=False):
		if not isinstance(is_add_empty_columns, bool):
			raise ValueError("invalid type(is_add_empty_columns): {}".format(type(is_add_empty_columns)))
		number_total_labels = len(
			labels)
		is_labels_non_empty = np.full(
			fill_value=True,
			shape=number_total_labels,
			dtype=bool)
		if is_add_empty_columns:
			is_labels_non_empty[0] = False
			is_labels_non_empty[-1] = False
		number_true_labels = int(
			np.sum(
				is_labels_non_empty))
		if title is not None:
			if not isinstance(title, str):
				raise ValueError("invalid type(title): {}".format(type(title)))
			leg.set_title(
				title,
				prop={
					"size": self.label_size, 
					# "weight" : "semibold",
					})
			if title_color is not None:
				leg.get_title().set_color(
					title_color)
		leg._legend_box.align = "center"
		frame = leg.get_frame()
		if facecolor is not None:
			if not isinstance(facecolor, str):
				raise ValueError("invalid type(facecolor): {}".format(type(facecolor)))
			frame.set_facecolor(
				facecolor)
		if edgecolor is not None:
			if not isinstance(edgecolor, str):
				raise ValueError("invalid type(edgecolor): {}".format(type(edgecolor)))
			frame.set_edgecolor(
				edgecolor)
		if text_colors is not None:
			if isinstance(text_colors, str):
				modified_text_colors = [
					text_colors
						for _ in range(
							number_true_labels)]
			elif isinstance(text_colors, (tuple, list)):
				number_text_colors = len(
					text_colors)
				if number_text_colors != number_true_labels:
					raise ValueError("{} colors and {} labels are not compatible".format(number_text_colors, number_true_labels))
				modified_text_colors = list(
					text_colors)
			else:
				raise ValueError("invalid type(text_colors): {}".format(type(text_colors)))
			for index_at_label, (text, text_color) in enumerate(zip(leg.get_texts(), modified_text_colors)):
				if is_labels_non_empty[index_at_label]:
					text.set_color(
						text_color)
		return leg

class VisualLegendConfiguration(BaseVisualLegendConfiguration):

	def __init__(self):
		super().__init__()

	def get_legend(self, fig, handles, labels, ax=None, number_columns=None, title=None, title_color="black", text_colors="black", facecolor="lightgray", edgecolor="gray", **kwargs):
		leg, modified_handles, modified_labels, is_add_empty_columns = self.get_base_legend(
			fig=fig,
			handles=handles,
			labels=labels,
			ax=ax,
			number_columns=number_columns,
			loc="lower center",
			mode="expand",
			fontsize=self.label_size,
			borderaxespad=0.1,
			**kwargs)
		leg = self.autoformat_legend(
			leg=leg,
			labels=modified_labels,
			title=title,
			title_color=title_color,
			text_colors=text_colors,
			facecolor=facecolor,
			edgecolor=edgecolor,
			is_add_empty_columns=is_add_empty_columns)
		return leg

class VisualSettingsConfiguration(VisualLegendConfiguration):

	def __init__(self, tick_size=7, label_size=10, text_size=6, cell_size=6, title_size=12):
		super().__init__()
		self.initialize_font_sizes(
			tick_size=tick_size,
			label_size=label_size,
			text_size=text_size,
			cell_size=cell_size,
			title_size=title_size)

	def initialize_font_sizes(self, tick_size, label_size, text_size, cell_size, title_size):
		self.verify_element_is_strictly_positive(
			element=tick_size)
		self.verify_element_is_strictly_positive(
			element=label_size)
		self.verify_element_is_strictly_positive(
			element=text_size)
		self.verify_element_is_strictly_positive(
			element=cell_size)
		self.verify_element_is_strictly_positive(
			element=title_size)
		self._tick_size = tick_size
		self._label_size = label_size
		self._text_size = text_size
		self._cell_size = cell_size
		self._title_size = title_size

	def get_save_path(self, save_name, default_extension, extension=None, space_replacement=None):
		if self.path_to_save_directory is None:
			raise ValueError("cannot save plot; self.path_to_save_directory is None")
		if extension is None:
			modified_extension = default_extension[:]
		elif isinstance(extension, str):
			modified_extension = extension[:]
		else:
			raise ValueError("invalid type(extension): {}".format(type(extension)))
		save_path = self.autocorrect_string_spaces(
			s="{}{}{}".format(
				self.path_to_save_directory,
				save_name,
				modified_extension),
			space_replacement=space_replacement)
		return save_path

	def update_save_directory(self, path_to_save_directory=None):
		if path_to_save_directory is not None:
			if not isinstance(path_to_save_directory, str):
				raise ValueError("invalid type(path_to_save_directory): {}".format(type(path_to_save_directory)))
		self._path_to_save_directory = path_to_save_directory

	def display_image(self, fig, save_name=None, dpi=800, bbox_inches="tight", pad_inches=0.1, extension=None, space_replacement=None, **kwargs):
		if save_name is None:
			plt.show()
		elif isinstance(save_name, str):
			save_path = self.get_save_path(
				save_name=save_name,
				default_extension=".png",
				extension=extension,
				space_replacement=space_replacement)
			fig.savefig(
				save_path,
				dpi=dpi,
				bbox_inches=bbox_inches,
				pad_inches=pad_inches,
				**kwargs)
		else:
			raise ValueError("invalid type(save_name): {}".format(type(save_name)))
		plt.close(
			fig)

##