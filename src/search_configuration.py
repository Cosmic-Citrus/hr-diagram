import numpy as np
# import operator


class BaseDataSearcherConfiguration():

	def __init__(self):
		super().__init__()
		self._mapping = None
	
	@property
	def mapping(self):
		return self._mapping

	@staticmethod
	def verify_inputs(original_data, is_any, is_suppress_error):
		if not isinstance(is_any, bool):
			raise ValueError("invalid type(is_any): {}".format(type(is_any)))
		if not isinstance(is_suppress_error, bool):
			raise ValueError("invalid type(is_suppress_error): {}".format(type(is_suppress_error)))
		if not isinstance(original_data, dict):
			raise ValueError("invalid type(original_data): {}".format(type(original_data)))
		sizes = list()
		for key, value in original_data.items():
			if not isinstance(value, np.ndarray):
				raise ValueError("invalid type(data[{}]): {}".format(key, type(value)))
			size_of_shape = len(
				value.shape)
			if size_of_shape != 1:
				raise ValueError("invalid data[{}].shape: {}".format(key, value.shape))
			sizes.append(
				value.size)
		consecutive_size_differences = np.diff(
			np.array(
				sizes))
		if np.any(consecutive_size_differences != 0):
			raise ValueError("size of each value in original_data.values() must be the same")

	@staticmethod
	def get_autocorrected_inputs(search_parameters, search_conditions, search_values):

		def get_initial_parameters(search_parameters):
			if isinstance(search_parameters, str):
				modified_search_parameters = [search_parameters]
			elif isinstance(search_parameters, (tuple, list)):
				modified_search_parameters = tuple(
					search_parameters)
			else:
				raise ValueError("invalid type(search_parameters): {}".format(type(search_parameters)))
			return modified_search_parameters

		def get_initial_conditions(search_conditions):
			if isinstance(search_conditions, str):
				modified_search_conditions = [search_conditions]
			elif isinstance(search_conditions, (tuple, list)):
				modified_search_conditions = tuple(
					search_conditions)
			else:
				raise ValueError("invalid type(search_conditions): {}".format(type(search_conditions)))
			return modified_search_conditions

		def get_initial_values(search_values):
			if isinstance(search_values, (float, int, str, np.str_)):
				modified_search_values = [search_values]
			elif isinstance(search_values, (tuple, list, np.ndarray)):
				modified_search_values = np.array(
					search_values)
			else:
				raise ValueError("invalid type(search_values): {}".format(type(search_values)))
			return modified_search_values

		def get_normalized_search_parameters(modified_search_parameters, modified_search_conditions, modified_search_values):
			number_search_parameters = len(
				modified_search_parameters)
			number_search_conditions = len(
				modified_search_conditions)
			number_search_values = len(
				modified_search_values)
			if not ((number_search_parameters == number_search_conditions) and (number_search_conditions == number_search_values)):
				if not ((number_search_parameters == 1) or (number_search_conditions == 1) or (number_search_values == 1)):
					raise ValueError("{} search_parameters, {} search_conditions, and {} search_values are not compatible".format(number_search_parameters, number_search_conditions, number_search_values))
				largest_number_inputs = max([
					number_search_parameters,
					number_search_conditions,
					number_search_values])
				modified_search_parameters = (modified_search_parameters * largest_number_inputs)[:largest_number_inputs]
				modified_search_conditions = (modified_search_conditions * largest_number_inputs)[:largest_number_inputs]
				if number_search_values < largest_number_inputs:
					modified_search_values = np.resize(
						modified_search_values,
						largest_number_inputs)
			return modified_search_parameters, modified_search_conditions, modified_search_values

		if (search_parameters is None) or (search_conditions is None) or (search_values is None):
			raise ValueError("invalid combination of search_parameters, search_conditions, and search_values")
		modified_search_parameters = get_initial_parameters(
			search_parameters=search_parameters)
		modified_search_conditions = get_initial_conditions(
			search_conditions=search_conditions)
		modified_search_values = get_initial_values(
			search_values=search_values)
		modified_search_parameters, modified_search_conditions, modified_search_values = get_normalized_search_parameters(
			modified_search_parameters=modified_search_parameters,
			modified_search_conditions=modified_search_conditions,
			modified_search_values=modified_search_values)
		return modified_search_parameters, modified_search_conditions, modified_search_values

	def initialize_mapping(self):
		mapping = {
			"equal" : (
				np.equal, # operator.eq,
				r"$\eq$"),
			"not equal" : (
				np.not_equal, # operator.ne,
				r"$\neq$"),
			"greater than" : (
				np.greater, # operator.gt,
				r"$\gt$"),
			"greater than or equal" : (
				np.greater_equal, # operator.ge,
				r"$\geq$"),
			"less than" : (
				np.less, # operator.lt,
				r"$\lt$"),
			"less than or equal" : (
				np.less_equal, # operator.l,
				r"$\leq$")}
		self._mapping = mapping

	def get_match_states(self, original_data, search_args):
		(search_parameters, search_conditions, search_values, is_any) = search_args
		multiple_match_states = list()
		for parameter, condition, value in zip(search_parameters, search_conditions, search_values):
			arr = original_data[parameter]
			get_comparison = self.mapping[condition][0]
			match_states = get_comparison(
				arr,
				value)
			multiple_match_states.append(
				match_states)
		number_groups = len(
			multiple_match_states)
		multiple_match_states = np.array(
			multiple_match_states)
		if number_groups == 0:
			raise ValueError("empty quantity in search_args")
		if number_groups == 1:
			is_matches = multiple_match_states.flatten()
		else: # elif number_groups > 1:
			if is_any:
				is_matches = np.any(
					multiple_match_states,
					axis=0)
			else:
				is_matches = np.all(
					multiple_match_states,
					axis=0)
		return is_matches

class DataSearcherConfiguration(BaseDataSearcherConfiguration):

	def __init__(self):
		super().__init__()
		self.initialize_mapping()

	def get_data(self, original_data, search_parameters, search_conditions, search_values, is_any=False, is_suppress_error=False):
		self.verify_inputs(
			original_data=original_data,
			is_any=is_any,
			is_suppress_error=is_suppress_error)
		modified_search_parameters, modified_search_conditions, modified_search_values = self.get_autocorrected_inputs(
			search_parameters=search_parameters,
			search_conditions=search_conditions,
			search_values=search_values)
		search_args = (
			modified_search_parameters,
			modified_search_conditions,
			modified_search_values,
			is_any)
		is_matches = self.get_match_states(
			original_data=original_data,
			search_args=search_args)
		if (not np.any(is_matches)) and (not is_suppress_error):
			raise ValueError("no matches found")
		data = {
			key : np.copy(value[is_matches])
				for key, value in original_data.items()}
		return data, is_matches, search_args


if __name__ == "__main__":


	search_parameters = "x"
	search_conditions = "greater than"
	search_values = 1

	# search_parameters = (
	# 	"x",)
	# search_conditions = (
	# 	"greater than",
	# 	"less than")
	# search_values = (
	# 	1,
	# 	4)

	# search_parameters = (
	# 	"x",
	# 	"y")
	# search_conditions = (
	# 	"greater than",
	# 	"less than")
	# search_values = (
	# 	5,)

	# search_parameters = (
	# 	"x",
	# 	"y")
	# search_conditions = (
	# 	"greater than",)
	# search_values = (
	# 	1,
	# 	200)

	# search_parameters = (
	# 	"x",
	# 	"y")
	# search_conditions = (
	# 	"greater than",
	# 	"less than",
	# 	"equal to")
	# search_values = (
	# 	1,
	# 	105)

	original_data = {
		"x" : np.arange(5),
		"y" : np.arange(5) * 100,
		"z" : np.arange(5) - 10}
	searcher = DataSearcherConfiguration()
	data, is_matches, search_args = searcher.get_data(
		original_data=original_data,
		search_parameters=search_parameters,
		search_conditions=search_conditions,
		search_values=search_values)

	(modified_search_parameters, modified_search_conditions, modified_search_values) = search_args
	print("\n .. SEARCH PARAMETERS:\n{}\n".format(
		modified_search_parameters))
	print("\n .. SEARCH CONDITIONS:\n{}\n".format(
		modified_search_conditions))
	print("\n .. SEARCH VALUES:\n{}\n".format(
		modified_search_values))
	print("\n .. IS MATCHES:\n{}\n".format(
		is_matches))
	print("\n .. ORIGINAL DATA:\n{}\n".format(
		original_data))
	print("\n .. DATA:\n{}\n".format(
		data))


##