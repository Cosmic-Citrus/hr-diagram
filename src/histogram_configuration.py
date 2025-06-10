import numpy as np


class BasestHistogramConfiguration():

	def __init__(self):
		super().__init__()
		self._distribution_values = None
		self._side_bias = None
		self._number_bins = None
		self._bin_edges = None
		self._bin_widths = None
		self._bin_midpoints = None
		self._bin_counts = None
		self._cumulative_bin_counts = None
		self._is_bins_equivalent_widths = None

	@property
	def distribution_values(self):
		return self._distribution_values
	
	@property
	def side_bias(self):
		return self._side_bias
	
	@property
	def number_bins(self):
		return self._number_bins
	
	@property
	def bin_edges(self):
		return self._bin_edges
	
	@property
	def bin_widths(self):
		return self._bin_widths
	
	@property
	def bin_midpoints(self):
		return self._bin_midpoints
		
	@property
	def bin_counts(self):
		return self._bin_counts
	
	@property
	def cumulative_bin_counts(self):
		return self._cumulative_bin_counts
	
	@property
	def is_bins_equivalent_widths(self):
		return self._is_bins_equivalent_widths

	@staticmethod
	def get_rounded_number(number, base=10, f=None):
		if f is None:
			rounded_number = base * round(number / base)
		else:
			rounded_number = base * int(
				f(number / base))
		return rounded_number

	@staticmethod
	def verify_log_base(log_base):
		if not isinstance(log_base, (int, float)):
			raise ValueError("invalid type(log_base): {}".format(type(log_base)))
		if log_base <= 0:
			raise ValueError("invalid log_base: {}".format(log_base))

	@staticmethod
	def verify_container_is_flat_numerical_array(container):
		if not isinstance(container, np.ndarray):
			raise ValueError("invalid type(container): {}".format(type(container)))
		if not (np.issubdtype(container.dtype, np.integer) or np.issubdtype(container.dtype, np.floating)):
			raise ValueError("invalid container.dtype: {}".format(container.dtype))
		size_at_shape = len(
			container.shape)
		if size_at_shape != 1:
			raise ValueError("invalid container.shape: {}".format(arr.container))

	def verify_container_is_shifting_monotonically(self, container):
		modified_container = np.array(
			container)
		self.verify_container_is_flat_numerical_array(
			container=modified_container)
		consecutive_differences = np.diff(
			modified_container)
		signed_consecutive_differences = np.sign(
			consecutive_differences)
		if not np.all(signed_consecutive_differences == signed_consecutive_differences[0]):
			raise ValueError("container is not shifting monotonically")

	def verify_container_is_strictly_positive(self, container):
		modified_container = np.array(
			container)
		self.verify_container_is_flat_numerical_array(
			container=modified_container)
		if np.any(modified_container <= 0):
			raise ValueError("container contains value less than or equal to zero")

	def get_up_rounded_number(self, number, base=10):
		rounded_number = self.get_rounded_number(
			number=number,
			base=base,
			f=np.ceil)
		return rounded_number

	def get_down_rounded_number(self, number, base=10):
		rounded_number = self.get_rounded_number(
			number=number,
			base=base,
			f=np.floor)
		return rounded_number

class BaseHistogramBinsConfiguration(BasestHistogramConfiguration):

	def __init__(self):
		super().__init__()

	@staticmethod
	def get_midpoints(bin_edges):
		bin_midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2
		return bin_midpoints

	def get_leftmost_and_rightmost_bin_edges(self, leftmost_edge=None, rightmost_edge=None, round_to_base=None):
		is_modified = False
		if leftmost_edge is None:
			leftmost_edge = np.nanmin(
				self.distribution_values)
		if rightmost_edge is None:
			rightmost_edge = np.nanmax(
				self.distribution_values)
		if not isinstance(leftmost_edge, (int, float)):
			raise ValueError("invalid type(leftmost_edge): {}".format(type(leftmost_edge)))
		if not isinstance(rightmost_edge, (int, float)):
			raise ValueError("invalid type(rightmost_edge): {}".format(type(rightmost_edge)))
		if (round_to_base is not None) and (leftmost_edge != rightmost_edge):
			leftmost_edge = self.get_down_rounded_number(
				leftmost_edge,
				base=round_to_base)
			rightmost_edge = self.get_up_rounded_number(
				rightmost_edge,
				base=round_to_base)
		# if rightmost_edge <= leftmost_edge:
		# 	raise ValueError("leftmost_edge={} is not less than rightmost_edge={}".format(leftmost_edge, rightmost_edge))
		if rightmost_edge < leftmost_edge:
			leftmost_edge, rightmost_edge = rightmost_edge, leftmost_edge
		if leftmost_edge == rightmost_edge:
			if round_to_base is None:
				leftmost_edge -= 1
				rightmost_edge += 1
			else:
				leftmost_edge -= float(
					round_to_base)
				rightmost_edge += float(
					round_to_base)
			is_modified = True
		return leftmost_edge, rightmost_edge, is_modified

	def get_bins_by_number(self, number_bins=None, leftmost_edge=None, rightmost_edge=None, round_to_base=None, log_base=None):
		modified_leftmost_edge, modified_rightmost_edge, is_modified = self.get_leftmost_and_rightmost_bin_edges(
			leftmost_edge=leftmost_edge,
			rightmost_edge=rightmost_edge,
			round_to_base=round_to_base)
		if is_modified:
			modified_bin_edges = np.array([
				modified_leftmost_edge,
				modified_rightmost_edge])
		else:
			if log_base is None:
				if number_bins is None:
					raise ValueError("invalid number_bins: {}".format(number_bins))
				modified_bin_edges = np.linspace(
					modified_leftmost_edge,
					modified_rightmost_edge,
					number_bins)
			else:
				self.verify_log_base(
					log_base=log_base)
				log_at_leftmost_edge = np.log(modified_leftmost_edge) / np.log(log_base)
				log_at_rightmost_edge = np.log(modified_rightmost_edge) / np.log(log_base)
				log_at_upper_bound = max([
					log_at_leftmost_edge,
					log_at_rightmost_edge])
				log_at_lower_bound = min([
					log_at_leftmost_edge,
					log_at_rightmost_edge])
				if number_bins is None:
					number_bins = int(log_at_upper_bound - log_at_lower_bound) + 2
				modified_bin_edges = np.logspace(
					log_at_leftmost_edge,
					log_at_rightmost_edge,
					number_bins,
					base=log_base)
		return modified_bin_edges

	def get_bins_by_edges(self, bin_edges):
		self.verify_container_is_flat_numerical_array(
			container=bin_edges)
		self.verify_container_is_shifting_monotonically(
			container=bin_edges)
		modified_bin_edges = np.copy(
			bin_edges)
		return modified_bin_edges

	def get_bins_by_equivalent_width(self, bin_widths, leftmost_edge=None, rightmost_edge=None, round_to_base=None):
		if isinstance(bin_widths, (int, float)):
			if bin_widths <= 0:
				raise ValueError("invalid bin_widths: {}".format(bin_widths))
		else:
			raise ValueError("invalid type(bin_widths): {}".format(type(bin_widths)))
		modified_leftmost_edge, modified_rightmost_edge, is_modified = self.get_leftmost_and_rightmost_bin_edges(
			leftmost_edge=leftmost_edge,
			rightmost_edge=rightmost_edge,
			round_to_base=round_to_base)
		if is_modified:
			modified_bin_edges = np.array([
				modified_leftmost_edge,
				modified_rightmost_edge])
		else:
			modified_bin_edges = np.arange(
				modified_leftmost_edge,
				modified_rightmost_edge + bin_widths,
				bin_widths)
		return modified_bin_edges

	def get_bins_by_midpoints(self, bin_midpoints, bin_widths, is_reverse=False):
		self.verify_container_is_flat_numerical_array(
			container=bin_midpoints)
		self.verify_container_is_shifting_monotonically(
			container=bin_midpoints)
		modified_bin_midpoints = np.copy(
			bin_midpoints)
		if isinstance(bin_widths, (tuple, list, np.ndarray)):
			modified_bin_widths = np.copy(
				bin_widths)
		elif isinstance(bin_widths, (int, float)):
			modified_bin_widths = np.array([
				bin_widths
					for _ in range(
						modified_bin_midpoints.size)])
		else:
			raise ValueError("invalid type(bin_widths): {}".format(type(bin_widths)))
		self.verify_container_is_flat_numerical_array(
			container=modified_bin_widths)
		self.verify_container_is_strictly_positive(
			container=modified_bin_widths)
		if modified_bin_widths.shape != modified_bin_midpoints.shape:
			raise ValueError("modified_bin_widths.shape={} and modified_bin_midpoints.shape={} are not compatible".format(modified_bin_widths.shape, modified_bin_midpoints.shape))
		smallest_edge = np.min(modified_bin_midpoints) - modified_bin_widths[0] / 2
		bin_edges = [smallest_edge]
		for midpoint, width in zip(modified_bin_midpoints, modified_bin_widths):
			new_edge = midpoint + width / 2
			bin_edges.append(
				new_edge)
		if is_reverse:
			bin_edges.reverse()
		bin_edges = np.array(
			bin_edges)
		return bin_edges

class BaseHistogramBinCountsConfiguration(BaseHistogramBinsConfiguration):

	def __init__(self):
		super().__init__()

	def get_bin_counts_by_left_side_bias(self, bin_edges):
		bin_counts, _ = np.histogram(
			self.distribution_values,
			bins=bin_edges)
		return bin_counts

	def get_bin_counts_by_right_side_bias(self, bin_edges):
		bin_counts = np.full(
			fill_value=0,
			shape=bin_edges.size - 1,
			dtype=int)
		count_value_indices, count_values = np.unique(
			np.searchsorted(
				bin_edges,
				self.distribution_values,
				side="left"),
			return_counts=True)
		for index_at_count_value, count_value in zip(count_value_indices, count_values):
			bin_counts[index_at_count_value - 1] = count_value
		return bin_counts

class HistogramBinsConfiguration(BaseHistogramBinCountsConfiguration):

	def __init__(self):
		super().__init__()

	def get_bin_edges(self, bin_selection_method, *args, **kwargs):
		mapping_at_bin_selection = {
			"number bins" : self.get_bins_by_number,
			"bin edges" : self.get_bins_by_edges,
			"equivalent bin widths" : self.get_bins_by_equivalent_width,
			"bin midpoints" : self.get_bins_by_midpoints}
		if bin_selection_method not in mapping_at_bin_selection.keys():
			raise ValueError("invalid bin_selection_method: {}".format(bin_selection_method))
		get_bin_edges = mapping_at_bin_selection[bin_selection_method]
		bin_edges = get_bin_edges(
			*args,
			**kwargs)
		number_bins = bin_edges.size - 1
		if number_bins == 0:
			raise ValueError("invalid len(bin_edges): {}".format(bin_edges.size))
		bin_widths = np.diff(
			bin_edges)
		bin_midpoints = self.get_midpoints(
			bin_edges=bin_edges)
		is_bins_equivalent_widths = np.all(
			bin_widths == bin_widths[0])
		return bin_edges, number_bins, bin_widths, bin_midpoints, is_bins_equivalent_widths

	def get_bin_counts(self, bin_counts=None):
		mapping_at_side_bias = {
			"left" : self.get_bin_counts_by_left_side_bias,
			"right" : self.get_bin_counts_by_right_side_bias}
		if self.side_bias not in mapping_at_side_bias.keys():
			raise ValueError("invalid self.side_bias: {}".format(self.side_bias))
		if bin_counts is None:
			get_bin_counts = mapping_at_side_bias[self.side_bias]
			bin_counts = get_bin_counts(
				bin_edges=self.bin_edges)
		else:
			self.verify_container_is_flat_numerical_array(
				container=bin_counts)
			if self.number_bins != bin_counts.size:
				raise ValueError("{} bins and {} bin-counts are not compatible".format(self.number_bins, bin_counts))
		cumulative_bin_counts = np.cumsum(
			bin_counts)
		return bin_counts, cumulative_bin_counts

class BaseHistogramConfiguration(HistogramBinsConfiguration):

	def __init__(self):
		super().__init__()

	def initialize_distribution_values(self, distribution_values):
		self.verify_container_is_flat_numerical_array(
			container=distribution_values)
		self._distribution_values = distribution_values

	def initialize_side_bias(self, side_bias):
		if side_bias not in ("left", "right"):
			raise ValueError("invalid side_bias: {}".format(side_bias))
		self._side_bias = side_bias

	def initialize_bin_edges(self, bin_selection_method, *args, **kwargs):
		bin_edges, number_bins, bin_widths, bin_midpoints, is_bins_equivalent_widths = self.get_bin_edges(
			bin_selection_method,
			*args,
			**kwargs)
		self._number_bins = number_bins
		self._bin_edges = bin_edges
		self._bin_widths = bin_widths
		self._bin_midpoints = bin_midpoints
		self._is_bins_equivalent_widths = is_bins_equivalent_widths

	def initialize_bin_counts(self, bin_counts=None):
		bin_counts, cumulative_bin_counts = self.get_bin_counts(
			bin_counts=bin_counts)
		self._bin_counts = bin_counts
		self._cumulative_bin_counts = cumulative_bin_counts

class HistogramConfiguration(BaseHistogramConfiguration):

	def __init__(self):
		super().__init__()

	def __repr__(self):
		histogram = f"HistogramConfiguration()"
		return histogram

	def __str__(self):
		title = "\n Histogram of {:,} Data-Points\n".format(
			self.distribution_values.size)
		if self.bin_edges[0] > self.bin_edges[-1]:
			operator_symbol = "≥"
		else:
			operator_symbol = "≤"
		bin_edges = "\n .. {:,.2f} {} {:,} bin edges {} {:,.2f}\n".format(
			self.bin_edges[0],
			operator_symbol,
			self.bin_edges.size,
			operator_symbol,
			self.bin_edges[-1])
		bin_midpoints = "\n .. {:,.2f} {} {:,} bin midpoints {} {:,.2f}\n".format(
			self.bin_midpoints[0],
			operator_symbol,
			self.bin_midpoints.size,
			operator_symbol,
			self.bin_midpoints[-1])
		bin_counts = "\n .. {:,.2f} ≤ {:,} bin counts ≤ {:,.2f}\n".format(
			np.min(
				self.bin_counts),
			self.bin_counts.size,
			np.max(
				self.bin_counts))
		s = "{}{}{}{}".format(
			title,
			bin_edges,
			bin_midpoints,
			bin_counts)
		return s

	def initialize(self, distribution_values, bin_selection_method, *args, side_bias="left", bin_counts=None, **kwargs):
		self.initialize_distribution_values(
			distribution_values=distribution_values)
		self.initialize_side_bias(
			side_bias=side_bias)
		self.initialize_bin_edges(
			bin_selection_method,
			*args,
			**kwargs)
		self.initialize_bin_counts(
			bin_counts=bin_counts)

class BaseTwoDimensionalHistogramConfiguration():

	def __init__(self):
		super().__init__()
		self._x_histogram = None
		self._y_histogram = None
		self._bin_counts = None

	@property
	def x_histogram(self):
		return self._x_histogram
	
	@property
	def y_histogram(self):
		return self._y_histogram

	@property
	def bin_counts(self):
		return self._bin_counts
	
	def initialize_x(self, x_values, x_bin_selection_method, *args, x_bin_counts=None, **kwargs):
		x_histogram = HistogramConfiguration()
		x_histogram.initialize_distribution_values(
			distribution_values=x_values)
		x_histogram.initialize_side_bias(
			side_bias="left") # "right"
		x_histogram.initialize_bin_edges(
			x_bin_selection_method,
			*args,
			**kwargs)
		x_histogram.initialize_bin_counts(
			bin_counts=x_bin_counts)
		self._x_histogram = x_histogram

	def initialize_y(self, y_values, y_bin_selection_method, *args, y_bin_counts=None, **kwargs):
		y_histogram = HistogramConfiguration()
		y_histogram.initialize_distribution_values(
			distribution_values=y_values)
		y_histogram.initialize_side_bias(
			side_bias="left") # "right"
		y_histogram.initialize_bin_edges(
			y_bin_selection_method,
			*args,
			**kwargs)
		y_histogram.initialize_bin_counts(
			bin_counts=y_bin_counts)
		self._y_histogram = y_histogram

	def initialize_bin_counts(self):
		bin_counts, x_edges, y_edges = np.histogram2d(
			x=self.x_histogram.distribution_values,
			y=self.y_histogram.distribution_values,
			bins=(
				self.x_histogram.bin_edges,
				self.y_histogram.bin_edges))
		self._bin_counts = bin_counts

class TwoDimensionalHistogramConfiguration(BaseTwoDimensionalHistogramConfiguration):

	def __init__(self):
		super().__init__()

	def __repr__(self):
		histogram = f"TwoDimensionalHistogramConfiguration()"
		return histogram

	def __str__(self):
		singular_title = "\n 2-D Histogram of {:,} x {:,} Data-Points\n".format(
			self.x_histogram.distribution_values.size,
			self.y_histogram.distribution_values.size)
		bin_edges, bin_midpoints = list(), list()
		for histogram, axis_label in zip((self.x_histogram, self.y_histogram), ("x", "y")):
			if histogram.bin_edges[0] > histogram.bin_edges[-1]:
				operator_symbol = "≥"
			else:
				operator_symbol = "≤"
			partial_bin_edges = "\n .. {:,.2f} {} {:,} {}- bin edges {} {:,.2f}\n".format(
				histogram.bin_edges[0],
				operator_symbol,
				histogram.bin_edges.size,
				axis_label,
				operator_symbol,
				histogram.bin_edges[-1])
			partial_bin_midpoints = "\n .. {:,.2f} {} {:,} {}- bin midpoints {} {:,.2f}\n".format(
				histogram.bin_midpoints[0],
				operator_symbol,
				histogram.bin_midpoints.size,
				axis_label,
				operator_symbol,
				histogram.bin_midpoints[-1])
			bin_edges.append(
				partial_bin_edges)
			bin_midpoints.append(
				partial_bin_midpoints)
		singular_bin_edges = "".join(
			bin_edges)
		singular_bin_midpoints = "".join(
			bin_midpoints)
		singular_bin_counts = "\n .. {:,.2f} ≤ ({:,} x {:,}) bin counts ≤ {:,.2f}\n".format(
			np.min(
				self.bin_counts),
			*self.bin_counts.shape,
			np.max(
				self.bin_counts))
		s = "{}{}{}{}".format(
			singular_title,
			singular_bin_edges,
			singular_bin_midpoints,
			singular_bin_counts)
		return s

	def initialize(self, x_values, y_values, x_bin_selection_method, y_bin_selection_method, x_args=None, x_kwargs=None, y_args=None, y_kwargs=None, x_bin_counts=None, y_bin_counts=None):

		def get_modified_args_and_kwargs(args, kwargs):
			if args is None:
				modified_args = tuple()
			elif isinstance(args, (tuple, list)):
				modified_args = tuple(
					modified_args)
			else:
				raise ValueError("invalid type(args): {}".format(type(args)))
			if kwargs is None:
				modified_kwargs = dict()
			elif isinstance(kwargs, dict):
				modified_kwargs = dict(
					kwargs)
			else:
				raise ValueError("invalid type(kwargs): {}".format(type(kwargs)))
			return modified_args, modified_kwargs

		modified_x_args, modified_x_kwargs = get_modified_args_and_kwargs(
			x_args,
			x_kwargs)
		modified_y_args, modified_y_kwargs = get_modified_args_and_kwargs(
			y_args,
			y_kwargs)
		self.initialize_x(
			x_values,
			x_bin_selection_method,
			*modified_x_args,
			x_bin_counts=x_bin_counts,
			**modified_x_kwargs)
		self.initialize_y(
			y_values,
			y_bin_selection_method,
			*modified_y_args,
			y_bin_counts=y_bin_counts,
			**modified_y_kwargs)
		self.initialize_bin_counts()

##